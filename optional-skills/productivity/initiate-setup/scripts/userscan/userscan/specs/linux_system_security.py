"""Linux system-level probes: security, network.

Registration only at import time. Shared helpers live in linux_system.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import grp

from .linux_system import (_enabled_units, _fv, _home, _ini, _kv, _me, _nm_kind, _nm_profiles,
    _read, _read1, _run, _which, lp)

# ================================================================ security

def _sshd_directives():
    out, unreadable = {}, 0
    files = ["/etc/ssh/sshd_config"] + sorted(glob.glob("/etc/ssh/sshd_config.d/*.conf"))[:50]
    for p in files:
        txt = _read(p, 200_000)
        if txt is None:
            unreadable += os.path.exists(p)
            continue
        for line in txt.splitlines()[:2000]:
            parts = line.strip().split(None, 1)
            if not parts or parts[0].startswith("#"):
                continue
            k = parts[0].lower()
            if k in ("port", "passwordauthentication", "pubkeyauthentication", "permitrootlogin",
                     "kbdinteractiveauthentication", "x11forwarding", "usepam"):
                out.setdefault(k, parts[1] if len(parts) > 1 else "")
            elif k in ("allowusers", "allowgroups", "match", "denyusers"):
                out[k] = "<set>"
    return out, len(files), unreadable


@lp("security.openssh_server", level="L1", family="security", tier="T3")
def openssh_server(h, facts):
    """sshd installed and enabled (ssh.service/ssh.socket wants-links), non-secret sshd_config directives."""
    exe = _which("sshd")
    if not exe:
        return None
    en = _enabled_units()
    d, nfiles, unreadable = _sshd_directives()
    svc = "ssh.service" in en or "sshd.service" in en
    sock = "ssh.socket" in en
    return {"present": True, "installed": True, "sshd_auto": svc or sock, "service_enabled": svc, "socket_enabled": sock,
            "sshd_config": d, "config_files": nfiles, "config_files_unreadable": unreadable}


@lp("dev.ssh_dir_presence", level="L1", family="security", tier="T3")
def ssh_dir_presence(h, facts):
    """~/.ssh file kinds by name/stat only (keys never opened); known_hosts line count, config Host count."""
    d = os.path.join(_home(), ".ssh")
    names = h.list_dir(d, 200)
    if not names:
        return None
    pub = [n for n in names if n.endswith(".pub")]
    priv = [n for n in names if n.startswith("id_") and not n.endswith(".pub")]
    out = {"present": True, "files": len(names), "private_keys": len(priv), "public_keys": len(pub),
           "config": os.path.exists(os.path.join(d, "config")),
           "authorized_keys": h.meta(os.path.join(d, "authorized_keys")).get("bytes"),
           "known_hosts_lines": None, "config_host_entries": None, "agent_socket": bool(os.environ.get("SSH_AUTH_SOCK"))}
    try:
        with open(os.path.join(d, "known_hosts"), "rb") as f:
            out["known_hosts_lines"] = sum(1 for ln in f if ln.strip())
    except OSError:
        pass
    hosts = _ssh_hosts()
    if hosts is not None:
        out["config_host_entries"] = len(hosts)
    return out


def _ssh_hosts():
    """Host aliases from ~/.ssh/config and ~/.ssh/config.d/*; wildcard patterns dropped; no other keys read."""
    d = os.path.join(_home(), ".ssh")
    files = [os.path.join(d, "config")] + sorted(glob.glob(os.path.join(d, "config.d", "*")))[:20]
    seen, any_file = [], False
    for p in files:
        txt = _read(p, 500_000)
        if txt is None:
            continue
        any_file = True
        for line in txt.splitlines()[:5000]:
            m = re.match(r"\s*Host\s+(.+)", line, re.I)
            if m:
                for a in m.group(1).split():
                    if not any(c in a for c in "*?!") and a not in seen:
                        seen.append(a)
    return seen if any_file else None


@lp("security.ssh_config_hosts", level="L2", family="security", tier="T2", gate="dev.ssh_dir_presence")
def ssh_config_hosts(h, facts):
    """ssh config Host aliases (names only; HostName/User/IdentityFile never read). T2: machine names."""
    hosts = _ssh_hosts()
    if not hosts:
        return None
    return {"present": True, "count": len(hosts), "aliases": hosts[:50]}


@lp("security.firewall", level="L1", family="security")
def security_firewall(h, facts):
    """Host firewall: ufw ENABLED in ufw.conf, firewalld/nftables/iptables-persistent enabled."""
    en = _enabled_units()
    ufw_conf = _kv(_read("/etc/ufw/ufw.conf") or "")
    out = {"ufw_installed": os.path.exists("/usr/sbin/ufw"), "ufw": ufw_conf.get("ENABLED") == "yes" if ufw_conf else None,
           "firewalld": "firewalld.service" in en, "nftables": "nftables.service" in en,
           "iptables_persistent": "netfilter-persistent.service" in en}
    out["present"] = any(v for v in out.values())
    out["all_on"] = bool(out["ufw"] or out["firewalld"] or out["nftables"] or out["iptables_persistent"])
    return out


@lp("security.secureboot", level="L1", family="security")
def security_secureboot(h, facts):
    """UEFI boot and Secure Boot state from efivars (SecureBoot, SetupMode)."""
    uefi = os.path.isdir("/sys/firmware/efi")
    sb = None
    for p in glob.glob("/sys/firmware/efi/efivars/SecureBoot-*")[:1]:
        try:
            with open(p, "rb") as f:
                b = f.read(5)
            sb = len(b) == 5 and b[4] == 1
        except OSError:
            pass
    return {"present": True, "uefi": uefi, "secure_boot": sb, "enabled": bool(sb),
            "mok_enrolled": os.path.exists("/var/lib/shim-signed/mok/MOK.der")}


@lp("security.uac", level="L1", family="security")
def security_uac(h, facts):
    """Elevation path: invoking user in sudo/admin/wheel, sudoers.d drop-ins count, polkit rules count."""
    try:
        groups = sorted({grp.getgrgid(g).gr_name for g in os.getgroups()})
    except KeyError:
        groups = []
    admin = [g for g in groups if g in ("sudo", "admin", "wheel", "root")]
    return {"present": True, "mechanism": "sudo" if _which("sudo") else None, "in_admin_group": bool(admin),
            "admin_groups": admin, "is_root": os.geteuid() == 0,
            "sudo_used_before": os.path.exists(os.path.join(_home(), ".sudo_as_admin_successful")),
            "sudoers_d_files": len([n for n in h.list_dir("/etc/sudoers.d", 200) if n != "README"]),
            "polkit_rules": len(h.list_dir("/etc/polkit-1/rules.d", 200)),
            "other_groups": [g for g in groups if g in ("docker", "lxd", "libvirt", "kvm", "adm", "wireshark")]}


@lp("security.sudo_nopasswd", level="L2", family="security", gate="security.uac", collect="extended")
def sudo_nopasswd(h, facts):
    """sudo -n -l classification: password required / NOPASSWD rules / not allowed. Commands never emitted.
    Side effect: when a password is required sudo logs one auth-log line per run; skipped for non-admins."""
    uac = _fv(facts, "security.uac") or {}
    if not _which("sudo") or not uac.get("in_admin_group"):
        return None
    out = _run(h, ["sudo", "-n", "-l"], 3000)
    if out is None:
        return {"present": True, "result": "error"}
    low = out.lower()
    if "a password is required" in low or not out.strip():
        res = "password_required"
    elif "may not run sudo" in low or "not allowed" in low:
        res = "not_allowed"
    else:
        res = "listed"
    nop = len(re.findall(r"NOPASSWD:", out))
    nop_all = bool(re.search(r"NOPASSWD:\s*ALL\b", out))
    return {"present": True, "result": res, "nopasswd_rules": nop, "nopasswd_all": nop_all}


@lp("acct.admins", level="L2", family="security", gate="security.uac")
def acct_admins(h, facts):
    """Members of sudo/admin/wheel from /etc/group (counts only; names never emitted)."""
    out = {}
    for g in ("sudo", "admin", "wheel"):
        try:
            out[g] = len(grp.getgrnam(g).gr_mem)
        except KeyError:
            continue
    if not out:
        return None
    return {"present": True, "members": out, "count": max(out.values())}


@lp("security.telemetry", level="L1", family="security")
def security_telemetry(h, facts):
    """Crash/usage reporting: apport enabled flag, whoopsie, popularity-contest, ubuntu-report consent file."""
    ap = _kv(_read("/etc/default/apport") or "")
    pop = _kv(_read("/etc/popularity-contest.conf") or "")
    en = _enabled_units()
    apport_on = ap.get("enabled") == "1" if ap else None
    return {"present": True, "apport_enabled": apport_on, "whoopsie": "whoopsie.service" in en,
            "popcon": pop.get("PARTICIPATE") == "yes" if pop else None,
            "ubuntu_report_file": os.path.isdir(os.path.join(_home(), ".cache/ubuntu-report")),
            "ubuntu_pro_status_file": os.path.exists("/var/lib/ubuntu-advantage/status.json"),
            "reduced": apport_on is False}


_REMOTE = {"xrdp": ["/usr/sbin/xrdp"], "krdp": ["/usr/bin/krdpserver"], "gnome-remote-desktop": ["/usr/libexec/gnome-remote-desktop-daemon"],
           "krfb": ["/usr/bin/krfb"], "x11vnc": ["/usr/bin/x11vnc"], "tigervnc": ["/usr/bin/x0vncserver", "/usr/bin/Xtigervnc"],
           "anydesk": ["/usr/bin/anydesk"], "teamviewer": ["/opt/teamviewer"], "rustdesk": ["/usr/bin/rustdesk"],
           "sunshine": ["/usr/bin/sunshine"], "nomachine": ["/usr/NX"], "parsec": ["/usr/bin/parsecd"],
           "chrome-remote-desktop": ["/opt/google/chrome-remote-desktop"], "et": ["/usr/bin/etserver"],
           "mosh": ["/usr/bin/mosh-server"]}


@lp("security.rdp", level="L1", family="security")
def security_rdp(h, facts):
    """RDP server (xrdp / KDE krdp / gnome-remote-desktop) installed and enabled."""
    en = _enabled_units()
    inst = [k for k in ("xrdp", "krdp", "gnome-remote-desktop") if any(os.path.exists(p) for p in _REMOTE[k])]
    enabled = "xrdp.service" in en
    return {"present": True, "installed": inst, "rdp_enabled": enabled or None}


@lp("security.remote_tools", level="L1", family="security")
def security_remote_tools(h, facts):
    """Remote access tools present: VNC servers, AnyDesk, TeamViewer, RustDesk, Sunshine, EternalTerminal, mosh."""
    found = sorted(k for k, ps in _REMOTE.items() if any(os.path.exists(p) for p in ps))
    vnc_dir = h.meta(os.path.join(_home(), ".vnc"))
    return {"present": True, "tools": found, "count": len(found), "user_vnc_dir": vnc_dir.get("present", False),
            "user_vnc_passwd_bytes": h.meta(os.path.join(_home(), ".vnc/passwd")).get("bytes")}


@lp("security.apparmor", level="L1", family="security")
def security_apparmor(h, facts):
    """Active LSMs (AppArmor/SELinux), kernel lockdown mode."""
    lsm = (_read1("/sys/kernel/security/lsm") or "").split(",")
    aa = _read1("/sys/module/apparmor/parameters/enabled")
    lock = _read1("/sys/kernel/security/lockdown")
    m = re.search(r"\[(\w+)\]", lock or "")
    return {"present": True, "lsm": [x for x in lsm if x], "apparmor": aa == "Y" if aa else "apparmor" in lsm,
            "selinux": "selinux" in lsm, "lockdown": m.group(1) if m else None,
            "apparmor_profiles_dir": len(h.list_dir("/etc/apparmor.d", 2000))}


@lp("security.bitlocker", level="L1", family="security")
def security_disk_encryption(h, facts):
    """Disk encryption (Linux: LUKS dm-crypt mappings from /sys/block/dm-*/dm/uuid; BitLocker equivalent)."""
    crypt = 0
    for u in glob.glob("/sys/block/dm-*/dm/uuid")[:200]:
        if (_read1(u) or "").startswith("CRYPT-"):
            crypt += 1
    return {"present": True, "mechanism": "luks", "encrypted_volumes": crypt, "any_encrypted": crypt > 0,
            "crypttab_lines": sum(1 for l in (_read("/etc/crypttab") or "").splitlines() if l.strip() and not l.startswith("#"))}


@lp("security.tpm", level="L1", family="security")
def security_tpm(h, facts):
    """TPM present (/sys/class/tpm) and its major version; resource manager device for the user."""
    devs = [n for n in h.list_dir("/sys/class/tpm", 10) if n.startswith("tpm")]
    if not devs:
        return {"present": False, "tpm": False}
    ver = _read1(f"/sys/class/tpm/{devs[0]}/tpm_version_major")
    return {"present": True, "tpm": True, "version": ver, "rm_device": os.path.exists("/dev/tpmrm0")}


@lp("acct.autologon", level="L1", family="security")
def acct_autologon(h, facts):
    """Display-manager autologin (sddm/gdm/lightdm) and whether it logs in the invoking user. No names."""
    me = _me()
    user, dm = None, None
    for p in ["/etc/sddm.conf"] + sorted(glob.glob("/etc/sddm.conf.d/*.conf")):
        sec = _ini(_read(p) or "").get("Autologin") or {}
        if sec.get("User"):
            user, dm = sec["User"], "sddm"
    g = _ini(_read("/etc/gdm3/custom.conf") or "").get("daemon") or {}
    if g.get("AutomaticLoginEnable", "").lower() == "true":
        user, dm = g.get("AutomaticLogin"), "gdm"
    for p in ["/etc/lightdm/lightdm.conf"] + sorted(glob.glob("/etc/lightdm/lightdm.conf.d/*.conf")):
        for s in _ini(_read(p) or "").values():
            if s.get("autologin-user"):
                user, dm = s["autologin-user"], "lightdm"
    return {"present": True, "enabled": bool(user), "display_manager": dm,
            "is_current_user": bool(user and me and user == me.pw_name)}


_PM = {"KeePassXC": ["/usr/bin/keepassxc", "~/.config/keepassxc"], "Bitwarden": ["/usr/bin/bitwarden", "/opt/Bitwarden", "~/.config/Bitwarden"],
       "1Password": ["/opt/1Password", "/usr/bin/1password"], "Proton Pass": ["/usr/bin/proton-pass", "/opt/proton-pass"],
       "Enpass": ["/opt/enpass"], "pass": ["~/.password-store"], "gopass": ["/usr/bin/gopass", "~/.config/gopass"],
       "KWallet": ["~/.local/share/kwalletd"], "GNOME Keyring": ["~/.local/share/keyrings"]}


@lp("pm.desktop", level="L1", family="security")
def pm_desktop(h, facts):
    """Password managers and OS keyrings by path presence (vault/keyring files never opened)."""
    found = sorted(k for k, ps in _PM.items() if any(os.path.exists(os.path.expanduser(p)) for p in ps))
    managers = [f for f in found if f not in ("KWallet", "GNOME Keyring")]
    keyrings = [f for f in found if f not in managers]
    return {"present": bool(managers or keyrings), "managers": managers, "keyrings": keyrings,
            "count": len(managers)}


@lp("security.av_products", level="L2", family="security", gate="security.uac")
def av_products(h, facts):
    """Anti-malware / EDR agents by path: ClamAV, Sophos, CrowdStrike, SentinelOne, Defender for Endpoint, Wazuh."""
    paths = {"clamav": "/usr/bin/clamscan", "sophos": "/opt/sophos-spl", "crowdstrike": "/opt/CrowdStrike",
             "sentinelone": "/opt/sentinelone", "mdatp": "/opt/microsoft/mdatp", "wazuh": "/var/ossec",
             "rkhunter": "/usr/bin/rkhunter", "chkrootkit": "/usr/sbin/chkrootkit", "fail2ban": "/usr/bin/fail2ban-server"}
    found = sorted(k for k, p in paths.items() if os.path.exists(p))
    return {"present": True, "products": found, "count": len(found)}


_NOTABLE_SVC = ("ssh", "tailscaled", "docker", "containerd", "xrdp", "cups", "bluetooth", "avahi-daemon", "ufw",
                "apparmor", "snapd", "libvirtd", "sddm", "gdm", "lightdm", "NetworkManager", "systemd-networkd",
                "unattended-upgrades", "fail2ban", "nginx", "apache2", "caddy", "postgresql", "mysql", "mariadb",
                "redis-server", "ollama", "smbd", "nfs-server", "cockpit", "zerotier-one", "wg-quick@wg0", "k3s",
                "nvidia-persistenced", "power-profiles-daemon", "tlp", "thermald")


@lp("security.services_state", level="L2", family="security", gate="host.systemd", collect="extended")
def services_state(h, facts):
    """systemd unit counts: enabled services/timers/sockets, failed units, notable services enabled/active."""
    txt = _run(h, ["systemctl", "list-unit-files", "--no-legend", "--no-pager", "--type=service,timer,socket"], 4000) or ""
    by = collections.Counter()
    enabled = set()
    for line in txt.splitlines()[:5000]:
        p = line.split()
        if len(p) >= 2:
            kind = p[0].rsplit(".", 1)[-1]
            by[f"{kind}:{p[1]}"] += 1
            if p[1] == "enabled":
                enabled.add(p[0].rsplit(".", 1)[0])
    failed = _run(h, ["systemctl", "list-units", "--no-legend", "--no-pager", "--state=failed", "--plain"], 3000) or ""
    act = _run(h, ["systemctl", "list-units", "--no-legend", "--no-pager", "--type=service", "--state=running", "--plain"], 3000) or ""
    running = {l.split()[0].rsplit(".", 1)[0] for l in act.splitlines() if l.strip()}
    if not by:
        return None
    return {"present": True, "unit_files": sum(by.values()), "by_kind_state": dict(sorted(by.items())),
            "enabled_services": by.get("service:enabled", 0), "running_services": len(running),
            "failed_units": sum(1 for l in failed.splitlines() if l.strip()),
            "notable_enabled": sorted(s for s in _NOTABLE_SVC if s in enabled),
            "notable_running": sorted(s for s in _NOTABLE_SVC if s in running)}


@lp("security.cert_roots", level="L2", family="security", gate="os.edition", collect="extended")
def cert_roots(h, facts):
    """Locally added CA certificates (/usr/local/share/ca-certificates) vs system bundle size."""
    local = [n for n in h.list_dir("/usr/local/share/ca-certificates", 500)]
    return {"present": True, "local_added": len(local), "etc_ssl_certs": max(h.count_dir("/etc/ssl/certs", 5000), 0)}


# ================================================================ network

_PROXY_ENV = ("http_proxy", "https_proxy", "all_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


@lp("net.hosts_proxy", level="L1", family="network")
def net_hosts_proxy(h, facts):
    """/etc/hosts custom entries count, proxy variables set (names only), apt proxy config."""
    lines = [l.split("#")[0].split() for l in (_read("/etc/hosts") or "").splitlines()]
    lines = [l for l in lines if len(l) >= 2]
    default = {"localhost", "ip6-localhost", "ip6-loopback", "ip6-allnodes", "ip6-allrouters", "ip6-localnet",
               "ip6-mcastprefix", os.uname().nodename}
    custom = [l for l in lines if not set(l[1:]) <= default and not l[0].startswith(("127.0.1.1", "::1", "fe00", "ff0"))]
    envf = _kv(_read("/etc/environment") or "")
    apt_proxy = any("proxy" in n.lower() for n in h.list_dir("/etc/apt/apt.conf.d", 200))
    return {"present": True, "hosts_custom_entries": len(custom), "hosts_entries": len(lines),
            "proxy_env": sorted({k.lower() for k in _PROXY_ENV if os.environ.get(k)}),
            "proxy_etc_environment": sorted({k.lower() for k in envf if k in _PROXY_ENV}), "apt_proxy": apt_proxy}


_VPN_BINS = {"tailscale": "tailscale", "wireguard": "wg", "openvpn": "openvpn", "zerotier": "zerotier-cli",
             "cloudflare-warp": "warp-cli", "netbird": "netbird", "mullvad": "mullvad", "nordvpn": "nordvpn",
             "protonvpn": "protonvpn-cli", "expressvpn": "expressvpn", "headscale": "headscale", "nebula": "nebula"}


@lp("net.vpn_clients", level="L1", family="network")
def vpn_clients(h, facts):
    """VPN/mesh clients by binary and enabled systemd service (tailscaled, wg-quick@, openvpn@, zerotier-one...)."""
    found = sorted(k for k, b in _VPN_BINS.items() if _which(b))
    en = _enabled_units()
    svc = sorted(u for u in en if re.match(r"(tailscaled|wg-quick@|openvpn|zerotier-one|warp-svc|netbird|mullvad)", u))
    return {"present": True, "clients": found, "dirs": found, "services": svc, "count": len(found),
            "wireguard_dir": os.path.isdir("/etc/wireguard")}


@lp("net.networkmanager", level="L1", family="network")
def net_networkmanager(h, facts):
    """Network stack: NetworkManager / systemd-networkd / netplan, profile file count (names never emitted)."""
    en = _enabled_units()
    nm = os.path.exists("/usr/sbin/NetworkManager")
    files = h.list_dir("/etc/NetworkManager/system-connections", 2000)
    netplan = [n for n in h.list_dir("/etc/netplan", 100) if n.endswith(".yaml")]
    if not nm and not netplan and "systemd-networkd.service" not in en:
        return None
    return {"present": True, "networkmanager": nm, "nm_enabled": "NetworkManager.service" in en,
            "nm_profile_files": len(files), "networkd_enabled": "systemd-networkd.service" in en,
            "netplan_files": len(netplan)}


@lp("net.wifi_profiles", level="L2", family="network", gate="net.networkmanager")
def wifi_profiles(h, facts):
    """Saved Wi-Fi profile count from nmcli TYPE column; SSIDs never read."""
    rows = _nm_profiles(h)
    if rows is None:
        return None
    n = sum(1 for r in rows if _nm_kind(r["type"]) == "wireless")
    wifi_hw = any(os.path.isdir(os.path.join("/sys/class/net", i, "wireless")) for i in h.list_dir("/sys/class/net", 100))
    return {"present": True, "count": n, "wifi_hardware": wifi_hw}


@lp("net.connection_profiles", level="L2", family="network", gate="net.networkmanager", collect="extended")
def connection_profiles(h, facts):
    """NetworkManager profiles by kind and autoconnect count (names never requested)."""
    rows = _nm_profiles(h)
    if not rows:
        return None
    return {"present": True, "count": len(rows), "by_kind": dict(collections.Counter(_nm_kind(r["type"]) for r in rows)),
            "autoconnect": sum(1 for r in rows if r["autoconnect"])}


def _if_kind(name):
    base = os.path.join("/sys/class/net", name)
    if name == "lo":
        return "loopback"
    if os.path.isdir(os.path.join(base, "wireless")) or os.path.exists(os.path.join(base, "phy80211")):
        return "wifi"
    if name.startswith(("tailscale", "wg", "tun", "zt", "nb-")):
        return "tunnel"
    if name.startswith(("docker", "br-", "veth", "virbr", "lxc", "lxd", "cni", "flannel", "vnet")) or \
            os.path.isdir(os.path.join(base, "bridge")):
        return "virtual"
    if name.startswith(("wwan", "ww")):
        return "mobile"
    if os.path.exists(os.path.join(base, "device")):
        return "ethernet"
    return "other"


@lp("net.adapters", level="L2", family="network", gate="net.hosts_proxy", collect="extended")
def net_adapters(h, facts):
    """Network interfaces from /sys/class/net: count by kind, up count, physical link speeds."""
    kinds, up, speeds = collections.Counter(), collections.Counter(), []
    for n in h.list_dir("/sys/class/net", 500):
        k = _if_kind(n)
        kinds[k] += 1
        if _read1(os.path.join("/sys/class/net", n, "operstate")) == "up":
            up[k] += 1
        if k in ("ethernet", "wifi"):
            s = _read1(os.path.join("/sys/class/net", n, "speed"))
            if s and s.lstrip("-").isdigit() and int(s) > 0:
                speeds.append(int(s))
    if not kinds:
        return None
    return {"present": True, "by_kind": dict(kinds), "up_by_kind": dict(up), "link_speeds_mbps": sorted(speeds)}


@lp("net.dns", level="L2", family="network", gate="net.hosts_proxy", collect="extended")
def net_dns(h, facts):
    """Resolver: resolv.conf nameserver count, systemd-resolved stub, Tailscale MagicDNS server, custom upstreams."""
    rc = _read("/etc/resolv.conf") or ""
    ns = re.findall(r"^\s*nameserver\s+(\S+)", rc, re.M)
    stub = "127.0.0.53" in ns
    try:
        resolved = "systemd/resolve" in os.readlink("/etc/resolv.conf")
    except OSError:
        resolved = False
    ups = []
    if stub and _which("resolvectl"):
        out = _run(h, ["resolvectl", "dns"], 2000) or ""
        ups = re.findall(r"(\d+\.\d+\.\d+\.\d+|[0-9a-f:]{6,})", out)
    public = {"1.1.1.1": "cloudflare", "1.0.0.1": "cloudflare", "8.8.8.8": "google", "8.8.4.4": "google",
              "9.9.9.9": "quad9", "208.67.222.222": "opendns"}
    return {"present": bool(ns), "nameservers": len(ns), "resolved_stub": stub, "resolved_managed": resolved,
            "upstreams": len(set(ups)), "magicdns": "100.100.100.100" in ups or "100.100.100.100" in ns,
            "public_resolvers": sorted({public[u] for u in set(ups) | set(ns) if u in public}),
            "search_domains": len(re.findall(r"^\s*search\s+(.+)", rc, re.M))}


@lp("net.tailscale", level="L2", family="network", tier="T1", gate="net.vpn_clients", collect="extended")
def net_tailscale(h, facts):
    """Tailscale state via 'tailscale status --json': peer counts by OS and online count; peer names never emitted."""
    exe = _which("tailscale")
    if not exe:
        return None
    out = _run(h, [exe, "status", "--json"], 3000)
    try:
        j = json.loads(out) if out else None
    except ValueError:
        j = None
    if not isinstance(j, dict):
        return {"present": True, "status": "unparsed"}
    peers = list((j.get("Peer") or {}).values())
    by_os = collections.Counter(p.get("OS") or "?" for p in peers)
    return {"present": True, "version": (j.get("Version") or "").split("-")[0], "backend": j.get("BackendState"),
            "peers": len(peers), "online": sum(1 for p in peers if p.get("Online")), "peers_by_os": dict(by_os),
            "exit_node_in_use": any(p.get("ExitNode") for p in peers),
            "magicdns": bool((j.get("CurrentTailnet") or {}).get("MagicDNSEnabled"))}
