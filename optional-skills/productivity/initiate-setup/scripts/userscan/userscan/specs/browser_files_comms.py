"""Comms and work probes (comms_work family, Windows).

Registration only at import time. Shared helpers and caches live in browser_files.
"""
from __future__ import annotations

import collections
import os
import re
import time

from .browser_files import (HKCU_EXPLORER, _cached, _env, _ft_iso, _hist_rows, _isdir, _iso,
    _load_json, _probe, _reg_values, _walk)

# ================================================================ family: comms_work

def _appx(h):
    out = {}
    for k in h.reg(r"HKCU\Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel"
                   r"\Repository\Packages") or []:
        parts = k.split("_")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def _prefetch(h):
    out = {}
    try:
        with os.scandir(r"C:\Windows\Prefetch") as it:
            for i, e in enumerate(it):
                if i > 5000:
                    break
                if e.name.endswith(".pf"):
                    exe = e.name.rsplit("-", 1)[0].upper()
                    try:
                        out[exe] = max(out.get(exe, 0), e.stat().st_mtime)
                    except OSError:
                        pass
    except OSError:
        return {}
    return out


def _run_keys(h):
    names = []
    for p in (r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run", r"HKLM\Software\Microsoft\Windows\CurrentVersion\Run"):
        names.extend((_reg_values(p) or {}).keys())
    return names


def _app_rows(h, table):
    L, R = _env("LOCALAPPDATA"), _env("APPDATA")
    appx = _cached(h, "appx", _appx)
    pf = _cached(h, "prefetch", _prefetch)
    run = _cached(h, "runkeys", _run_keys)
    out = {}
    for aid, (paths, pfs, pkgs, launched_paths) in table.items():
        hit = [p for p in paths if os.path.exists(p.format(L=L, R=R))]
        ax = {a: appx[a] for a in pkgs if a in appx}
        if not hit and not ax:
            continue
        last = max((pf[e] for e in pfs if e in pf), default=None)
        launched = None
        if launched_paths:
            launched = any(os.path.exists(p.format(L=L, R=R)) and bool(os.listdir(p.format(L=L, R=R)))
                           for p in launched_paths if _isdir(p.format(L=L, R=R)))
        row = {"installed": True, "appx_version": next(iter(ax.values()), None), "prefetch_last": _iso(last),
               "autostart": any(re.search(aid.split("_")[0], r, re.I) for r in run)}
        if launched is not None:
            row["launched"] = launched
        row["recent_use"] = bool(last and time.time() - last < 30 * 86400) or bool(launched)
        out[aid] = row
    return out


COMMS_APPS = {
    # id: (install paths, prefetch exes, appx package families, dirs that are non-empty only after first launch)
    "slack": ([r"{L}\slack", r"{R}\Slack"], ["SLACK.EXE"], ["91750D7E.Slack"], []),
    "teams_new": ([r"{L}\Packages\MSTeams_8wekyb3d8bbwe"], ["MS-TEAMS.EXE", "MSTEAMS.EXE"], ["MSTeams"],
                  [r"{L}\Packages\MSTeams_8wekyb3d8bbwe\LocalCache\Microsoft\MSTeams"]),
    "teams_classic": ([r"{L}\Microsoft\Teams", r"{R}\Microsoft\Teams"], ["TEAMS.EXE"], [], []),
    "zoom": ([r"{R}\Zoom", r"{L}\Zoom"], ["ZOOM.EXE"], ["ZoomVideoCommunications"], []),
    "telegram": ([r"{R}\Telegram Desktop"], ["TELEGRAM.EXE"], ["TelegramMessengerLLP.TelegramDesktop"], []),
    "whatsapp": ([r"{L}\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm", r"{L}\WhatsApp"], ["WHATSAPP.EXE"],
                 ["5319275A.WhatsAppDesktop"], []),
    "signal": ([r"{R}\Signal", r"{L}\Programs\signal-desktop"], ["SIGNAL.EXE"], [], []),
    "skype": ([r"{R}\Microsoft\Skype for Desktop"], ["SKYPE.EXE"], ["Microsoft.SkypeApp"], []),
    "webex": ([r"{L}\CiscoSpark", r"{L}\WebEx"], ["CISCOCOLLABHOST.EXE", "WEBEX.EXE"], [], []),
    "element": ([r"{R}\Element"], ["ELEMENT.EXE"], [], []),
    "thunderbird": ([r"{R}\Thunderbird"], ["THUNDERBIRD.EXE"], [], []),
    "outlook_classic": ([r"{L}\Microsoft\Outlook"], ["OUTLOOK.EXE"], [], []),
    "outlook_new": ([r"{L}\Packages\Microsoft.OutlookForWindows_8wekyb3d8bbwe"], ["OLK.EXE"],
                    ["Microsoft.OutlookForWindows"], [r"{L}\Microsoft\Olk"]),
    "mail_calendar_legacy": ([r"{L}\Packages\microsoft.windowscommunicationsapps_8wekyb3d8bbwe"],
                             ["HXOUTLOOK.EXE"], ["microsoft.windowscommunicationsapps"], []),
}

NOTES_APPS = {
    "notion": ([r"{R}\Notion", r"{L}\Programs\Notion"], ["NOTION.EXE"], [], []),
    "notion_calendar": ([r"{R}\Notion Calendar", r"{L}\Programs\cron-web"], ["NOTION CALENDAR.EXE"], [], []),
    "obsidian": ([r"{R}\obsidian\obsidian.json", r"{L}\Programs\obsidian"], ["OBSIDIAN.EXE"], [], []),
    "onenote": ([r"{L}\Microsoft\OneNote"], ["ONENOTE.EXE"], ["Microsoft.Office.OneNote"], []),
}


@_probe(id="comms.native_apps", level="L1", family="comms_work", tier="T0", collect="core")
def comms_native_apps(h, facts):
    """Native comms/mail apps installed (paths, appx) with prefetch last run and first-launch evidence."""
    rows = _app_rows(h, COMMS_APPS)
    if not rows:
        return {"present": False}
    return rows


@_probe(id="discord.present", level="L1", family="comms_work", tier="T0", collect="core")
def discord_present(h, facts):
    """Discord desktop installed; version from app-* dir; autostart."""
    roam, local = h.expand(r"%APPDATA%\discord"), h.expand(r"%LOCALAPPDATA%\Discord")
    if not (_isdir(roam) or _isdir(local)):
        return {"present": False}
    vers = sorted(d[4:] for d in h.list_dir(local, 100) if d.startswith("app-"))
    run = _cached(h, "runkeys", _run_keys)
    return {"present": True, "version": vers[-1] if vers else None, "autostart": any("discord" in r.lower() for r in run)}


@_probe(id="office.c2r", level="L1", family="comms_work", tier="T0", collect="core")
def office_c2r(h, facts):
    """Click-to-Run Office: product ids, version, platform, consumer vs business licence."""
    c = _reg_values(r"HKLM\SOFTWARE\Microsoft\Office\ClickToRun\Configuration") or {}
    pids = c.get("ProductReleaseIds")
    if not pids:
        return {"present": False}
    kind = ("consumer" if re.search(r"HomePrem|Personal|HomeStudent|Professional2", pids) else
            "business" if re.search(r"O365Business|ProPlus|Enterprise", pids) else "other")
    return {"present": True, "products": pids, "version": c.get("VersionToReport"), "platform": c.get("Platform"),
            "license_kind": kind}


@_probe(id="onedrive.accounts", level="L1", family="comms_work", tier="T0", collect="core")
def onedrive_accounts(h, facts):
    """OneDrive client and account slots; signed in only when UserEmail/cid/ConfiguredTenantId is set."""
    base = r"HKCU\SOFTWARE\Microsoft\OneDrive"
    exe = (h.exists(r"%LOCALAPPDATA%\Microsoft\OneDrive\OneDrive.exe") or
           h.exists(r"C:\Program Files\Microsoft OneDrive\OneDrive.exe"))
    keys = [k for k in (h.reg(base + r"\Accounts") or []) if k != "FileCoAuth"]
    if not exe and not keys:
        return {"present": False}
    signed = business = 0
    results = []
    for k in keys:
        v = _reg_values(base + r"\Accounts" + "\\" + k) or {}
        is_signed = bool(v.get("UserEmail") or v.get("cid") or v.get("ConfiguredTenantId"))
        signed += is_signed
        business += bool(is_signed and v.get("ConfiguredTenantId"))
        if isinstance(v.get("LastSignInResult"), int):
            results.append(hex(v["LastSignInResult"] & 0xFFFFFFFF))
    return {"present": True, "client_installed": bool(exe), "version": h.reg(base, "Version"),
            "account_slots": len(keys), "signed_in": signed, "business": business,
            "last_sign_in_results": sorted(set(results))}


@_probe(id="productivity.notes_apps", level="L1", family="comms_work", tier="T0", collect="core")
def notes_apps(h, facts):
    """Notes apps installed: Notion, Notion Calendar, Obsidian, OneNote."""
    rows = _app_rows(h, NOTES_APPS)
    return {"present": bool(rows), "apps": sorted(rows), "detail": rows}


@_probe(id="work.accounts", level="L1", family="comms_work", tier="T3", collect="core")
def work_accounts(h, facts):
    """Counts only: WAM token cache files, AAD broker accounts, MSA-linked logon marker. Files never opened."""
    tb = h.count_dir(r"%LOCALAPPDATA%\Microsoft\TokenBroker\Cache", 5000)
    aad = h.count_dir(r"%LOCALAPPDATA%\Packages\Microsoft.AAD.BrokerPlugin_cw5n1h2txyewy\AC\TokenBroker\Accounts", 5000)
    msa = h.reg(r"HKCU\SOFTWARE\Microsoft\IdentityCRL\UserExtendedProperties")
    aad_storage = h.reg(r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\AAD\Storage") or []
    return {"present": True, "tokenbroker_cache_files": max(tb, 0), "aad_broker_accounts": max(aad, 0),
            "aad_storage_keys": len(aad_storage), "windows_logon_msa_linked": bool(msa)}


@_probe(id="work.edition_org", level="L1", family="comms_work", tier="T0", collect="core")
def work_edition_org(h, facts):
    """Windows edition and whether RegisteredOrganization is set (the org name itself is not emitted)."""
    cv = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    ed = h.reg(cv, "EditionID")
    org = (h.reg(cv, "RegisteredOrganization") or "").strip()
    return {"present": bool(ed), "edition": ed, "registered_org_set": bool(org)}


@_probe(id="work.mdm", level="L1", family="comms_work", tier="T0", collect="core")
def work_mdm(h, facts):
    """Real MDM enrollments only (UPN, ProviderID MS DM Server, or DiscoveryServiceFullURL)."""
    base = r"HKLM\SOFTWARE\Microsoft\Enrollments"
    keys = h.reg(base) or []
    real = []
    for g in keys[:500]:
        v = _reg_values(base + "\\" + g) or {}
        if v.get("UPN") or v.get("ProviderID") in ("MS DM Server", "WMI_Bridge_Server") or v.get("DiscoveryServiceFullURL"):
            real.append({"type": v.get("EnrollmentType"), "provider": v.get("ProviderID")})
    omadm = h.reg(r"HKLM\SOFTWARE\Microsoft\Provisioning\OMADM\Accounts") or []
    return {"present": True, "real_enrollments": len(real), "enrollments": real[:5], "enrollment_keys_total": len(keys),
            "omadm_accounts": len(omadm),
            "intune_ime": h.exists(r"C:\Program Files (x86)\Microsoft Intune Management Extension")}


@_probe(id="work.mdm_policy_areas", level="L1", family="comms_work", tier="T0", collect="core")
def work_mdm_policy_areas(h, facts):
    """PolicyManager device areas and providers (corroborates MDM; built-in areas exist on every box)."""
    areas = h.reg(r"HKLM\SOFTWARE\Microsoft\PolicyManager\current\device") or []
    prov = h.reg(r"HKLM\SOFTWARE\Microsoft\PolicyManager\providers") or []
    return {"present": True, "device_areas": len(areas), "providers": len(prov), "areas": sorted(areas)[:40]}


@_probe(id="work.policies", level="L1", family="comms_work", tier="T0", collect="core")
def work_policies(h, facts):
    """Group-policy keys for browsers/Office/OneDrive/Teams/Slack/Zoom (HKLM+HKCU), value counts only."""
    out, apps = {}, {}
    for hive in ("HKLM", "HKCU"):
        out[hive + "_vendors"] = sorted(h.reg(hive + r"\SOFTWARE\Policies") or [])
        for b in (r"Google\Chrome", r"BraveSoftware\Brave", r"Microsoft\Edge", r"Mozilla\Firefox", r"Microsoft\Office",
                  r"Microsoft\OneDrive", r"Microsoft\Teams", "Slack", "Zoom"):
            p = hive + r"\SOFTWARE\Policies" + "\\" + b
            v, sk = _reg_values(p), h.reg(p)
            if v is not None or sk is not None:
                apps[f"{hive}:{b}"] = {"values": len(v or {}), "subkeys": len(sk or [])}
        for area in ("TenantRestrictions", "WorkplaceJoin"):
            v = _reg_values(hive + r"\SOFTWARE\Policies\Microsoft\Windows" + "\\" + area)
            if v:
                out.setdefault("windows_area_value_names", {})[f"{hive}:{area}"] = sorted(v)[:10]
    configured = {k: v for k, v in apps.items() if v["values"] or v["subkeys"]}
    return {"present": True, "app_policies": configured, "app_policy_count": len(configured), **out}


AGENT_RX = re.compile(r"^(IntuneManagementExtension|CSFalcon\w*|ZSATunnel|ZSAService|PanGPS|vpnagent|csc_\w*|Netskope\w*|"
                      r"stAgent\w*|SentinelAgent|Tanium\w*|CylanceSvc|CcmExec|jamf\w*|Kandji\w*|FortiClient\w*|"
                      r"NinjaRMM\w*|ManageEngine\w*|LTService|DattoRMM|CiscoAMP|OktaVerify\w*|DuoAuth\w*|Sense|"
                      r"GlobalProtect\w*|CarbonBlack|CbDefense|elastic-agent|WinCollect|Qualys\w*|Rapid7\w*|"
                      r"TaniumClient|TrendMicro\w*|SophosMCS\w*|Sophos\w*)$", re.I)


@_probe(id="work.security_agents", level="L1", family="comms_work", tier="T0", collect="core")
def work_security_agents(h, facts):
    """EDR/ZTNA/RMM agents from the Services registry (not sc query). Sense (Defender) is reported separately."""
    svcs = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services") or []
    hits = sorted(s for s in svcs if AGENT_RX.match(s))
    onboard = h.reg(r"HKLM\SOFTWARE\Microsoft\Windows Advanced Threat Protection\Status", "OnboardingState")
    return {"present": True, "agents": [s for s in hits if s.lower() != "sense"], "sense_present": "Sense" in hits,
            "defender_atp_onboarded": bool(onboard), "services_scanned": len(svcs)}


@_probe(id="work.join_state", level="L2", family="comms_work", tier="T0", collect="extended", gate="work.edition_org",
        timeout_ms=5000)
def work_join_state(h, facts):
    """Entra/AD/workplace join from dsregcmd /status, with the CloudDomainJoin registry as a cheap cross-check."""
    join_info = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\CloudDomainJoin\JoinInfo") or []
    domain = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters", "Domain") or ""
    txt = h.run(["dsregcmd", "/status"], timeout_ms=5000, text=False) or ""
    got = {}
    for k in ("AzureAdJoined", "EnterpriseJoined", "DomainJoined", "WorkplaceJoined", "AzureAdPrt"):
        m = re.search(rf"^\s*{k}\s*:\s*(\w+)", txt, re.M)
        if m:
            got[k] = m.group(1).upper() == "YES"
    joined = got.get("AzureAdJoined") or got.get("DomainJoined") or got.get("EnterpriseJoined")
    return {"present": True, "source": "dsregcmd" if got else "registry",
            "azure_ad_joined": got.get("AzureAdJoined", bool(join_info)), "domain_joined": got.get("DomainJoined"),
            "enterprise_joined": got.get("EnterpriseJoined"), "workplace_joined": got.get("WorkplaceJoined"),
            "azure_ad_prt": got.get("AzureAdPrt"), "cloud_join_info_keys": len(join_info),
            "tcpip_domain_set": bool(domain), "state": "joined" if joined else "workgroup"}


@_probe(id="comms.teams_launched", level="L2", family="comms_work", tier="T0", collect="core",
        gate="comms.native_apps")
def comms_teams_launched(h, facts):
    """New Teams actually started: LocalCache\\Microsoft\\MSTeams exists; tfw = work, tfl = personal profiles."""
    b = h.expand(r"%LOCALAPPDATA%\Packages\MSTeams_8wekyb3d8bbwe\LocalCache\Microsoft\MSTeams")
    if not _isdir(b):
        return {"present": False}
    prof = h.list_dir(os.path.join(b, "EBWebView"), 100)
    work = sum(1 for p in prof if "tfw" in p.lower())
    return {"present": True, "work_profile": work > 0, "work_profiles": work,
            "personal_profiles": sum(1 for p in prof if "tfl" in p.lower()),
            "log_files": max(h.count_dir(os.path.join(b, "Logs"), 5000), 0)}


def _userassist(h, pattern):
    rx = re.compile(pattern, re.I)
    base = HKCU_EXPLORER + r"\UserAssist"
    rot = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm")
    focus, last = 0, None
    for g in h.reg(base) or []:
        for n, v in (_reg_values(base + "\\" + g + r"\Count") or {}).items():
            if isinstance(v, bytes) and len(v) >= 68 and rx.search(n.translate(rot)):
                focus += int.from_bytes(v[12:16], "little") // 1000
                t = _ft_iso(int.from_bytes(v[60:68], "little"))
                last = max(filter(None, (last, t)), default=None)
    return focus, last


@_probe(id="discord.usage", level="L2", family="comms_work", tier="T1", collect="extended", gate="discord.present",
        timeout_ms=3000)
def discord_usage(h, facts):
    """Signed-in and voice days from renderer_js.log line tags; absent when the app never got past login."""
    log = h.expand(r"%APPDATA%\discord\logs\renderer_js.log")
    pf = _cached(h, "prefetch", _prefetch).get("DISCORD.EXE")
    focus_s, ua_last = _userassist(h, r"discord")
    days, gw, rtc = set(), set(), set()
    last_gw = last_auth = None
    size = 0
    if os.path.isfile(log):
        size = os.path.getsize(log)
        rx = re.compile(r"^\[(\d{4}-\d\d-\d\d) ([\d:.]+)\] \[\w+\]\s+\[?([A-Za-z_]+)")
        with open(log, encoding="utf-8", errors="replace") as fh:
            if size > 16_000_000:
                fh.seek(size - 16_000_000)
            for line in fh:
                m = rx.match(line)
                if not m:
                    continue
                d, tm, tag = m.groups()
                days.add(d)
                if tag == "GatewaySocket":
                    gw.add(d)
                    last_gw = d + "T" + tm[:8]
                elif tag.startswith("RTC"):
                    rtc.add(d)
                elif tag == "useAuthWebsocket":
                    last_auth = d + "T" + tm[:8]
    used = bool(gw or rtc)
    return {"present": used, "log_days": len(days), "first_day": min(days) if days else None,
            "last_day": max(days) if days else None, "gateway_days": len(gw), "rtc_days": len(rtc),
            "logged_in_last_session": bool(last_gw and (not last_auth or last_gw > last_auth)),
            "on_login_screen": bool(last_auth and (not last_gw or last_auth > last_gw)),
            "prefetch_last": _iso(pf), "userassist_focus_s": focus_s, "userassist_last": ua_last,
            "log_bytes": size}


@_probe(id="office.usage", level="L2", family="comms_work", tier="T1", collect="core", gate="office.c2r")
def office_usage(h, facts):
    """Office identity count, File/Place MRU item counts per app, Office exe prefetch dates (no names or paths)."""
    base = r"HKCU\SOFTWARE\Microsoft\Office"
    ids = h.reg(base + r"\16.0\Common\Identity\Identities") or []
    mru = {}
    for app in ("Word", "Excel", "PowerPoint", "OneNote", "Access", "Visio", "Publisher"):
        n = 0
        for sub in ("File MRU", "Place MRU"):
            n += sum(1 for k in (_reg_values(rf"{base}\16.0\{app}\{sub}") or {}) if k.lower().startswith("item"))
            for u in h.reg(rf"{base}\16.0\{app}\User MRU") or []:
                n += sum(1 for k in (_reg_values(rf"{base}\16.0\{app}\User MRU\{u}\{sub}") or {})
                         if k.lower().startswith("item"))
        if n:
            mru[app] = n
    pf = _cached(h, "prefetch", _prefetch)
    runs = {e: _iso(pf[e]) for e in ("WINWORD.EXE", "EXCEL.EXE", "POWERPNT.EXE", "OUTLOOK.EXE", "ONENOTE.EXE") if e in pf}
    return {"present": True, "identities": len(ids), "mru_items": sum(mru.values()), "mru_by_app": mru,
            "prefetch_last": runs, "used": bool(mru or runs)}


@_probe(id="obsidian.vaults", level="L2", family="comms_work", tier="T1", collect="extended",
        gate="productivity.notes_apps")
def obsidian_vaults(h, facts):
    """Obsidian vault count and note counts (vault names and paths are not emitted)."""
    j = _load_json(h.expand(r"%APPDATA%\obsidian\obsidian.json"))
    vs = (j or {}).get("vaults") or {} if isinstance(j, dict) else {}
    if not vs:
        return {"present": False}
    notes, files, newest = 0, 0, 0

    def on_file(e, s, depth):
        nonlocal notes, newest
        if e.name.lower().endswith(".md"):
            notes += 1
        newest = max(newest, s.st_mtime)
    for v in list(vs.values())[:10]:
        p = v.get("path") if isinstance(v, dict) else None
        if p and _isdir(p):
            st = _walk(p, max_depth=6, max_entries=30000, budget_s=1.5, on_file=on_file,
                       skip_dir=lambda n, _p: n in (".obsidian", ".git", ".trash", "node_modules"))
            files += st["files"]
    return {"present": True, "vaults": len(vs), "open": sum(1 for v in vs.values() if isinstance(v, dict) and v.get("open")),
            "notes_md": notes, "files": files, "newest_note": _iso(newest)}


def _history_ids(h):
    tot = collections.Counter()
    for r in _hist_rows(h):
        tot.update(r.get("_ids") or {})
    return tot


@_probe(id="browser.history.workspaces", level="L2", family="comms_work", tier="T2", collect="extended",
        gate="browser.catalog")
def history_workspaces(h, facts):
    """Distinct Discord guild / Slack workspace / Gmail slot / Linear workspace ids seen in history (counts only)."""
    rows = _hist_rows(h)
    if not rows:
        return {"present": False}
    ids = _history_ids(h)
    return {"discord": ids["discord_guilds"], "slack": ids["slack_workspaces"], "gmail": ids["gmail_slots"],
            "linear": ids["linear_workspaces"]}


@_probe(id="browser.history.work_hosts", level="L2", family="comms_work", tier="T2", collect="extended",
        gate="browser.catalog")
def history_work_hosts(h, facts):
    """Top hostnames for Slack, nousresearch.com, Discord, Notion and Linear (default profiles)."""
    hosts = {}
    for r in _hist_rows(h):
        if not r["is_default_profile"]:
            continue
        for k, c in (r.get("_work_hosts") or {}).items():
            hosts.setdefault(k, collections.Counter()).update(c)
    if not hosts:
        return {"present": False}
    return {"present": True, "hosts": {k: [x for x, _ in v.most_common(5)] for k, v in hosts.items()}}
