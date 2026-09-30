"""Linux system-level probes: hardware.

Registration only at import time. Shared helpers live in linux_system.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re

from userscan.specs.linux_system import (_ARCH, _dpkg_names, _enabled_units, _home, _read, _read1, _run,
    _which, lp)

# ================================================================ hardware

_CHASSIS = {1: "other", 2: "unknown", 3: "desktop", 4: "low-profile desktop", 6: "mini tower", 7: "tower",
            8: "portable", 9: "laptop", 10: "notebook", 13: "all-in-one", 14: "sub-notebook", 17: "main server chassis",
            23: "rack mount", 30: "tablet", 31: "convertible", 32: "detachable", 35: "mini pc", 36: "stick pc"}


def _meminfo():
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1].isdigit():
            out[p[0].rstrip(":")] = int(p[1]) * 1024
    return out


_REAL_FS = {"ext4", "ext3", "ext2", "xfs", "btrfs", "zfs", "f2fs", "vfat", "exfat", "ntfs", "ntfs3", "fuseblk", "bcachefs"}


def _volumes():
    vols, seen = [], set()
    for line in (_read("/proc/mounts") or "").splitlines()[:2000]:
        p = line.split()
        if len(p) < 3 or p[2] not in _REAL_FS or p[0] in seen:
            continue
        seen.add(p[0])
        mp = p[1].replace("\\040", " ")
        try:
            st = os.statvfs(mp)
        except OSError:
            continue
        size = st.f_blocks * st.f_frsize
        if size <= 0:
            continue
        free = st.f_bavail * st.f_frsize
        vols.append({"name": mp, "fstype": p[2], "size_gb": round(size / 1e9, 1), "free_gb": round(free / 1e9, 1),
                     "free_pct": round(100 * free / size, 1)})
    return vols


@lp("hw.system", level="L1", family="hardware")
def hw_system(h, facts):
    """DMI system/board/BIOS, chassis type, visible RAM, fixed volumes with free space."""
    d = "/sys/class/dmi/id"
    r = lambda n: _read1(os.path.join(d, n)) or None
    mi = _meminfo()
    ct = r("chassis_type")
    ram = round(mi.get("MemTotal", 0) / 2 ** 30, 1) or None
    vols = _volumes()
    return {"present": True, "manufacturer": r("sys_vendor"), "product": r("product_name"), "family": r("product_family"),
            "version": r("product_version"), "board_vendor": r("board_vendor"), "board": r("board_name"),
            "bios_vendor": r("bios_vendor"), "bios_version": r("bios_version"), "bios_date": r("bios_date"),
            "chassis": _CHASSIS.get(int(ct), ct) if ct and ct.isdigit() else ct, "visible_ram_gb": ram,
            "memory_load_pct": round(100 * (1 - mi["MemAvailable"] / mi["MemTotal"])) if mi.get("MemTotal") and "MemAvailable" in mi else None,
            "fixed_drives_total_free_gb_pct": [[v["name"], v["size_gb"], v["free_gb"], v["free_pct"]] for v in vols]}


@lp("hw.ram", level="L1", family="hardware")
def hw_ram(h, facts):
    """RAM and swap from /proc/meminfo (module count and speed need root dmidecode: not collected)."""
    mi = _meminfo()
    if not mi.get("MemTotal"):
        return None
    return {"present": True, "total_gb": round(mi["MemTotal"] / 2 ** 30, 1),
            "available_gb": round(mi.get("MemAvailable", 0) / 2 ** 30, 1),
            "swap_gb": round(mi.get("SwapTotal", 0) / 2 ** 30, 1), "swap_used_gb": round((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 2 ** 30, 1),
            "zram": any(n.startswith("zram") for n in h.list_dir("/sys/block", 500))}


@lp("hw.disks", level="L2", family="hardware", gate="hw.system")
def hw_disks(h, facts):
    """Physical disks (lsblk: size, rotational, transport, model, LUKS) and mounted volumes with free space."""
    disks = []
    out = _run(h, ["lsblk", "-J", "-b", "-d", "-o", "NAME,TYPE,SIZE,ROTA,RM,TRAN,MODEL"], 3000) if _which("lsblk") else None
    try:
        for b in (json.loads(out).get("blockdevices") or []) if out else []:
            if b.get("type") != "disk" or str(b.get("name", "")).startswith(("loop", "zram", "ram")):
                continue
            disks.append({"name": b.get("name"), "size_gb": round(int(b.get("size") or 0) / 1e9, 1),
                          "ssd": not _truthy_flag(b.get("rota")), "removable": _truthy_flag(b.get("rm")),
                          "transport": b.get("tran"), "model": (b.get("model") or "").strip() or None})
    except (ValueError, AttributeError):
        disks = []
    if not disks:
        for n in h.list_dir("/sys/block", 200):
            if n.startswith(("loop", "zram", "ram", "dm-", "md", "sr")):
                continue
            sz = _read1(f"/sys/block/{n}/size")
            disks.append({"name": n, "size_gb": round(int(sz) * 512 / 1e9, 1) if sz and sz.isdigit() else None,
                          "ssd": _read1(f"/sys/block/{n}/queue/rotational") == "0",
                          "removable": _read1(f"/sys/block/{n}/removable") == "1",
                          "model": _read1(f"/sys/block/{n}/device/model")})
    vols = _volumes()
    return {"present": bool(disks or vols), "disks": disks, "volumes": vols,
            "virtio": any(d.get("transport") == "virtio" or str(d.get("name", "")).startswith("vd") for d in disks)}


def _truthy_flag(v):
    return v in (True, 1, "1", "true")


@lp("hw.cpu", level="L2", family="hardware", gate="hw.system")
def hw_cpu(h, facts):
    """CPU model, vendor, sockets/cores/threads, max MHz, virtualization from lscpu -J (cpuinfo fallback)."""
    out = _run(h, ["lscpu", "-J"], 3000) if _which("lscpu") else None
    f = {}
    try:
        for e in (json.loads(out).get("lscpu") or []) if out else []:
            f[e.get("field", "").rstrip(":")] = e.get("data")
            for c in e.get("children") or []:
                f[c.get("field", "").rstrip(":")] = c.get("data")
    except (ValueError, AttributeError):
        f = {}
    if not f:
        ci = _read("/proc/cpuinfo", 1 << 20) or ""
        m = re.search(r"^model name\s*:\s*(.+)$", ci, re.M)
        return {"present": bool(ci), "name": m.group(1).strip() if m else None,
                "logical_processors": len(re.findall(r"^processor\s*:", ci, re.M)), "via": "cpuinfo"}
    num = lambda k: int(f[k]) if str(f.get(k, "")).isdigit() else None
    return {"present": True, "name": f.get("Model name"), "vendor": f.get("Vendor ID"),
            "logical_processors": num("CPU(s)"), "sockets": num("Socket(s)"), "cores_per_socket": num("Core(s) per socket"),
            "threads_per_core": num("Thread(s) per core"), "max_mhz": f.get("CPU max MHz"),
            "virtualization": f.get("Virtualization"), "hypervisor": f.get("Hypervisor vendor"),
            "virt_type": f.get("Virtualization type"), "native_arch": _ARCH.get(os.uname().machine, os.uname().machine)}


_PCI_VENDOR = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel", "0x1af4": "Red Hat (virtio)", "0x1234": "QEMU",
               "0x15ad": "VMware", "0x80ee": "VirtualBox", "0x1414": "Microsoft", "0x1a03": "ASPEED", "0x5143": "Qualcomm"}


@lp("hw.gpu", level="L2", family="hardware", gate="hw.system")
def hw_gpu(h, facts):
    """Display adapters: PCI class 03 devices from sysfs (vendor, driver, VRAM for amdgpu), names from lspci -mm."""
    adapters = []
    for dev in glob.glob("/sys/bus/pci/devices/*")[:512]:
        cls = _read1(os.path.join(dev, "class")) or ""
        if not cls.startswith("0x03"):
            continue
        ven = _read1(os.path.join(dev, "vendor")) or ""
        drv = None
        try:
            drv = os.path.basename(os.readlink(os.path.join(dev, "driver")))
        except OSError:
            pass
        vram = _read1(os.path.join(dev, "mem_info_vram_total"))
        adapters.append({"slot": os.path.basename(dev), "vendor": _PCI_VENDOR.get(ven, ven), "device_id": _read1(os.path.join(dev, "device")),
                         "driver": drv, "boot_vga": _read1(os.path.join(dev, "boot_vga")) == "1",
                         "vram_gb": round(int(vram) / 2 ** 30, 1) if vram and vram.isdigit() else None, "name": None})
    if adapters and _which("lspci"):
        out = _run(h, ["lspci", "-mm", "-D"], 2000) or ""
        for line in out.splitlines():
            m = re.match(r'(\S+) "([^"]*)" "([^"]*)" "([^"]*)"', line)
            if m:
                for a in adapters:
                    if a["slot"] == m.group(1):
                        a["name"] = f"{m.group(3)} {m.group(4)}".strip()
    if not adapters:
        return None
    virtual = [a for a in adapters if a["vendor"] in ("Red Hat (virtio)", "QEMU", "VMware", "VirtualBox", "Microsoft")]
    real = [a for a in adapters if a not in virtual]
    vr = [a["vram_gb"] for a in real if a["vram_gb"]]
    return {"present": True, "max_vram_gb": max(vr) if vr else None, "adapters": real, "virtual_adapters": virtual,
            "other": len(virtual), "nvidia": any(a["vendor"] == "NVIDIA" for a in real)}


@lp("hw.battery", level="L1", family="hardware")
def hw_battery(h, facts):
    """Battery present in /sys/class/power_supply: laptop vs desktop, charge, wear."""
    for n in h.list_dir("/sys/class/power_supply", 50):
        b = os.path.join("/sys/class/power_supply", n)
        if _read1(os.path.join(b, "type")) != "Battery" or _read1(os.path.join(b, "scope")) == "Device":
            continue
        full = _read1(os.path.join(b, "energy_full")) or _read1(os.path.join(b, "charge_full"))
        design = _read1(os.path.join(b, "energy_full_design")) or _read1(os.path.join(b, "charge_full_design"))
        wear = round(100 * int(full) / int(design), 1) if full and design and full.isdigit() and design.isdigit() and int(design) else None
        ac = [_read1(os.path.join("/sys/class/power_supply", m, "online")) for m in h.list_dir("/sys/class/power_supply", 50)
              if _read1(os.path.join("/sys/class/power_supply", m, "type")) == "Mains"]
        cap = _read1(os.path.join(b, "capacity"))
        return {"present": True, "on_ac": "1" in ac if ac else None, "charge_pct": int(cap) if cap and cap.isdigit() else None,
                "status": _read1(os.path.join(b, "status")), "health_pct": wear,
                "cycle_count": _read1(os.path.join(b, "cycle_count"))}
    return None


@lp("periph.rgb_suites", level="L2", family="hardware", gate="hw.system")
def periph_rgb_suites(h, facts):
    """RGB / peripheral suites: OpenRGB, ckb-next, Piper/ratbagd, Solaar, polychromatic (by path)."""
    paths = {"OpenRGB": ["/usr/bin/openrgb", "~/.config/OpenRGB"], "ckb-next": ["/usr/bin/ckb-next"],
             "Piper": ["/usr/bin/piper"], "Solaar": ["/usr/bin/solaar", "~/.config/solaar"],
             "polychromatic": ["/usr/bin/polychromatic-controller"], "liquidctl": ["/usr/bin/liquidctl"]}
    found = sorted(k for k, ps in paths.items() if any(os.path.exists(os.path.expanduser(p)) for p in ps))
    return {"present": bool(found), "suites": found} if found else None


@lp("hw.power_plan", level="L2", family="hardware", gate="hw.system")
def hw_power_plan(h, facts):
    """Power profile (power-profiles-daemon / ACPI platform_profile / tuned / TLP) and CPU governor."""
    prof = None
    if _which("powerprofilesctl"):
        prof = (_run(h, ["powerprofilesctl", "get"], 2000) or "").strip() or None
    gov = _read1("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    plat = _read1("/sys/firmware/acpi/platform_profile")
    tuned = _read1("/etc/tuned/active_profile")
    if not any((prof, gov, plat, tuned)):
        return None
    return {"present": True, "active": prof or plat or tuned or gov, "power_profile": prof, "platform_profile": plat,
            "governor": gov, "tuned": tuned, "tlp": os.path.exists("/usr/sbin/tlp")}


@lp("hw.audio", level="L2", family="hardware", tier="T1", gate="hw.system", collect="extended")
def hw_audio(h, facts):
    """Sound cards from /proc/asound/cards; PipeWire vs PulseAudio user service."""
    cards = re.findall(r"^\s*\d+\s+\[[^\]]*\]:\s*(.+)$", _read("/proc/asound/cards") or "", re.M)
    return {"present": bool(cards), "cards": [c.strip() for c in cards][:10], "count": len(cards),
            "pipewire": os.path.exists("/usr/bin/pipewire"), "pulseaudio": os.path.exists("/usr/bin/pulseaudio")}


@lp("hw.bluetooth", level="L2", family="hardware", tier="T1", gate="hw.system", collect="extended")
def hw_bluetooth(h, facts):
    """Bluetooth adapters in /sys/class/bluetooth and bluetooth.service; paired devices need root (not read)."""
    ad = [n for n in h.list_dir("/sys/class/bluetooth", 20) if ":" not in n]
    return {"present": bool(ad), "adapters": len(ad), "service_enabled": "bluetooth.service" in _enabled_units(),
            "paired_devices": None}


@lp("hw.usb_devices", level="L2", family="hardware", tier="T2", gate="hw.system", collect="extended")
def hw_usb_devices(h, facts):
    """USB devices attached now (not history): count by class, product names at T2."""
    names, classes = [], collections.Counter()
    for d in glob.glob("/sys/bus/usb/devices/*")[:500]:
        if ":" in os.path.basename(d) or os.path.basename(d).startswith("usb"):
            continue
        cls = _read1(os.path.join(d, "bDeviceClass"))
        classes[cls or "?"] += 1
        p = _read1(os.path.join(d, "product"))
        if p:
            names.append(p)
    return {"present": bool(names or classes), "count": sum(classes.values()), "by_class": dict(classes),
            "names": sorted(set(names))[:30]}


@lp("hw.monitor_mode", level="L2", family="hardware", gate="hw.system", collect="extended")
def hw_monitor_mode(h, facts):
    """Connected display outputs from /sys/class/drm: connector types, preferred mode, virtual vs physical."""
    outs = []
    for c in glob.glob("/sys/class/drm/card*-*")[:64]:
        if _read1(os.path.join(c, "status")) != "connected":
            continue
        name = os.path.basename(c).split("-", 1)[1]
        modes = (_read(os.path.join(c, "modes"), 4096) or "").split()
        outs.append({"connector": re.sub(r"-\d+$", "", name), "preferred_mode": modes[0] if modes else None,
                     "edid": (h.meta(os.path.join(c, "edid")).get("bytes") or 0) > 0})
    if not outs:
        return None
    return {"present": True, "connected": len(outs), "outputs": outs,
            "virtual_only": all(o["connector"].lower().startswith("virtual") for o in outs)}


@lp("hw.nvidia_smi", level="L2", family="hardware", gate="hw.system", collect="extended")
def hw_nvidia_smi(h, facts):
    """nvidia-smi query: VRAM, power draw/limit, temperature, pstate, driver."""
    exe = _which("nvidia-smi")
    if not exe:
        return None
    fields = "name,memory.total,power.draw,power.limit,power.default_limit,power.max_limit,temperature.gpu,pstate,driver_version,fan.speed"
    out = h.run([exe, "--query-gpu=" + fields, "--format=csv,noheader,nounits"], timeout_ms=5000)
    if not out:
        return {"present": False, "exe": True}
    gpus = []
    for line in out.strip().splitlines()[:8]:
        vals = [x.strip() for x in line.split(",")]
        gpus.append({k: (None if v in ("[N/A]", "N/A", "[Not Supported]") else v) for k, v in zip(fields.split(","), vals)})
    try:
        vram = round(max(float(g["memory.total"]) for g in gpus if g.get("memory.total")) / 1024, 1)
    except ValueError:
        vram = None
    return {"present": bool(gpus), "vram_gb": vram, "gpus": gpus}


@lp("hw.fonts", level="L2", family="hardware", tier="T1", gate="hw.system", collect="extended")
def hw_fonts(h, facts):
    """Font inventory: fontconfig family count, user-installed font files, fonts-* packages, Nerd/coding fonts."""
    fams = set()
    if _which("fc-list"):
        for line in (_run(h, ["fc-list", ":", "family"], 3000) or "").splitlines()[:20000]:
            if line.strip():
                fams.add(line.split(",")[0].strip())
    user = 0
    for d in (os.path.join(_home(), ".local/share/fonts"), os.path.join(_home(), ".fonts")):
        for root, dirs, files in os.walk(d):
            user += sum(1 for f in files if f.lower().endswith((".ttf", ".otf", ".woff2", ".pcf.gz")))
            if user > 5000 or root.count(os.sep) - d.count(os.sep) > 3:
                break
    pk = [n for n in _dpkg_names(h) if n.startswith("fonts-")]
    coding = sorted({f for f in fams if re.search(r"nerd|jetbrains mono|fira code|cascadia|iosevka|hack\b|meslo", f, re.I)})
    return {"present": bool(fams or user), "families": len(fams), "user_font_files": user, "font_packages": len(pk),
            "coding_fonts": coding[:10]}
