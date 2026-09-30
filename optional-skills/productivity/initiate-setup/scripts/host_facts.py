"""Machine and account facts for the ``/initiate-setup`` first turn.

The setup bot has no terminal or file tools, so every machine fact it branches on
is computed here and embedded in the one turn that starts setup. Everything is
deterministic: the bot never guesses a signal this script can compute.

Hardware facts come from ``hermes_platform.host`` (no environment input, no
subprocess). Account facts (full name, locale, home-folder age) are read from OS
user records, never from environment variables such as HOME or LANG.

Facts describe the machine that runs this Python process (the Hermes backend).
When the desktop app drives a remote backend, that is not the user's laptop.

Usage: ``python host_facts.py`` prints the JSON object. The ``/initiate-setup``
builder may also import ``collect()`` and embed its result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from hermes_platform.host import facts, products, runtime

SCHEMA_VERSION = 2

# 21 days leaves time to finish setup without counting a daily-use machine as new.
NEW_MACHINE_DAYS = 21

# Generic account names that are not a person's name.
_NON_NAMES = frozenset({
    "admin", "administrator", "default", "guest", "me", "owner", "root", "test", "user",
})

_SPARK_MODEL = re.compile(r"\b(dgx|spark|gb10)\b", re.I)

_FORK_QUESTION = "Know what you'd like it to make?"
_FALLBACK_QUESTION = "What sounds better?"

SCAN_DIR = Path(__file__).resolve().parent / "userscan"
SCAN_TIER = "T1"
SCAN_CACHE_MAX_AGE_S = 24 * 3600
SCAN_DEADLINE_S = 11
SETTLING_DAYS = 120

_NOT_VISIBLE_T1 = [
    "work hosts and repos", "named domains", "GPU and monitor history", "second browser profile",
    "ChatGPT conversation count", "disk encryption",
]

_SCAN_APPS = {
    "Blender": ("blender",),
    "OBS Studio": ("obs studio", "obs64", "obsproject", "=obs"),
    "DaVinci Resolve": ("davinci", "resolve.exe"),
    "Photoshop": ("photoshop",),
    "Premiere Pro": ("premiere",),
    "Lightroom": ("lightroom",),
    "GIMP": ("gimp",),
    "Krita": ("krita",),
    "Inkscape": ("inkscape",),
    "Audacity": ("audacity",),
    "Ableton Live": ("ableton",),
    "FL Studio": ("fl studio", "fl64"),
    "Unity": ("unity hub", "unity.exe", "=unity"),
    "Unreal Engine": ("unreal",),
    "Figma": ("figma",),
    "Clipchamp": ("clipchamp",),
    "VS Code": ("visual studio code", "vscode", "=code", "code.exe"),
    "Docker": ("docker desktop", "=docker", "orbstack"),
    "Obsidian": ("obsidian",),
    "Notion": ("notion",),
    "Slack": ("slack",),
    "Discord": ("discord",),
    "Teams": ("teams",),
    "Outlook": ("outlook",),
    "Zoom": ("zoom",),
    "Spotify": ("spotify",),
    "Steam": ("steam",),
}

_AGENT_NAMES = {"claude_code": "Claude Code", "codex": "Codex", "hermes": "Hermes"}

_BROWSERS = frozenset({
    "arc", "aside", "brave", "chrome", "chromium", "dia", "edge", "firefox", "librewolf", "opera", "orion",
    "safari", "vivaldi", "waterfox", "zen",
})

_BLENDER_TASK = {"id": "plugin:blender", "label": "Help me make something in Blender", "plugins": ["blender"]}
_NVIDIA_TASK = {
    "id": "plugin:nvidia",
    "label": "Set up my games and streaming",
    "plugins": ["nvidia-app", "nvidia-broadcast"],
}


# --- account facts -----------------------------------------------------------------


def _posix_account() -> tuple[str, str, str]:
    """Return (login, full name, home) from the user database."""
    import pwd

    entry = pwd.getpwuid(os.getuid())
    return entry.pw_name, entry.pw_gecos.split(",", 1)[0].strip(), entry.pw_dir


def _windows_account() -> tuple[str, str, str]:
    """Return (login, display name, profile folder) from Win32 account APIs."""
    import ctypes
    from ctypes import wintypes

    login_buf = ctypes.create_unicode_buffer(257)
    login_len = wintypes.DWORD(len(login_buf))
    login = login_buf.value if ctypes.windll.advapi32.GetUserNameW(login_buf, ctypes.byref(login_len)) else ""

    # EXTENDED_NAME_FORMAT NameDisplay = 3. Local accounts without a display name fail here.
    name_buf = ctypes.create_unicode_buffer(257)
    name_len = wintypes.ULONG(len(name_buf))
    full = name_buf.value if ctypes.windll.secur32.GetUserNameExW(3, name_buf, ctypes.byref(name_len)) else ""

    # CSIDL_PROFILE = 0x28: the user's profile folder, without reading USERPROFILE.
    home_buf = ctypes.create_unicode_buffer(260)
    home = home_buf.value if ctypes.windll.shell32.SHGetFolderPathW(None, 0x28, None, 0, home_buf) == 0 else ""
    return login, full, home


def _account() -> tuple[str, str, str]:
    try:
        return _windows_account() if sys.platform == "win32" else _posix_account()
    except (AttributeError, ImportError, KeyError, OSError):
        return "", "", ""


def _suggested_name(login: str, full: str) -> str | None:
    """A real full name only; a login handle is never offered as the user's name."""
    name = " ".join(full.split())
    if not (2 <= len(name) <= 40) or name.lower() in _NON_NAMES or name.lower() == login.lower():
        return None
    return name


def _home_age_days(home: str) -> int | None:
    """Home-folder birth time approximates account age. Linux exposes no birth time here."""
    if not home:
        return None
    try:
        born = getattr(os.stat(home), "st_birthtime", 0)
    except OSError:
        return None
    if born <= 0:
        return None
    return max(0, int((time.time() - born) // 86_400))


def _darwin_locale() -> str:
    """First preferred UI language, e.g. ``de-DE``, via CoreFoundation."""
    import ctypes
    import ctypes.util

    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    cf.CFLocaleCopyPreferredLanguages.restype = ctypes.c_void_p
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [ctypes.c_void_p]

    languages = cf.CFLocaleCopyPreferredLanguages()
    if not languages:
        return ""
    try:
        if cf.CFArrayGetCount(languages) < 1:
            return ""
        buffer = ctypes.create_string_buffer(64)
        # kCFStringEncodingUTF8
        if not cf.CFStringGetCString(cf.CFArrayGetValueAtIndex(languages, 0), buffer, 64, 0x08000100):
            return ""
        return buffer.value.decode()
    finally:
        cf.CFRelease(languages)


def _windows_locale() -> str:
    import ctypes

    buffer = ctypes.create_unicode_buffer(85)
    return buffer.value if ctypes.windll.kernel32.GetUserDefaultLocaleName(buffer, 85) else ""


def _linux_locale() -> str:
    """System locale from its config file; the shell's LANG is deliberately not read."""
    for path in ("/etc/locale.conf", "/etc/default/locale"):
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    key, _, value = line.strip().partition("=")
                    if key == "LANG" and value:
                        return value.strip("\"'").split(".", 1)[0].replace("_", "-")
        except OSError:
            continue
    return ""


def _locale() -> str:
    try:
        if sys.platform == "darwin":
            return _darwin_locale()
        if sys.platform == "win32":
            return _windows_locale()
        return _linux_locale()
    except (AttributeError, OSError, TypeError, ValueError):
        return ""


# --- derived signals ---------------------------------------------------------------


def _is_spark(os_family: str, arch: str, gpu: str, cpu: str) -> bool:
    """RTX Sparks by platform, architecture and GPU; DGX Sparks by model string."""
    rtx = os_family == "win32" and arch == "arm64" and gpu == "nvidia"
    return rtx or products.is_nvidia_arm_soc() or bool(_SPARK_MODEL.search(cpu.replace("_", " ")))


def _machine_kind(os_family: str, spark: bool) -> str:
    if spark:
        return "Spark"
    return {"darwin": "Mac", "win32": "PC"}.get(os_family, "computer")


def _days_ago(days: int) -> str:
    return "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"


def _description(*, looks_new: bool, age: int | None, spark: bool, gpu: str, cpu: str,
                 os_family: str, release: str, arch: str) -> str:
    """Age leads because a new machine needs setup work a daily-use one may have done."""
    parts = [
        f"set up {_days_ago(age)}" if looks_new and age is not None else "",
        "an NVIDIA Spark" if spark else "has an NVIDIA GPU" if gpu == "nvidia" else "",
        cpu,
        f"{os_family} {release}".strip(),
        arch,
    ]
    return ", ".join(part for part in parts if part)


def _fork(kind: str, leads: bool, plugin_tasks: list[dict]) -> dict:
    """Fork options in the order the flow pins. Ids are stable; labels may be translated."""
    mind = {"id": "mind", "label": "I have something in mind"}
    automate = {"id": "automate", "label": "Automate something I already do"}
    machine = {"id": "machine", "label": f"Help me set up this {kind}"}
    figure = {"id": "figure", "label": "Let's figure it out together"}
    skip = {"id": "skip", "label": "Skip this for now"}
    tasks = [{"id": task["id"], "label": task["label"]} for task in plugin_tasks]
    if leads:
        return {
            "question": _FORK_QUESTION,
            "options": [machine, {"id": "something_else", "label": "Something else"}],
            "fallback_question": _FALLBACK_QUESTION,
            "fallback_options": [mind, automate, *tasks, figure, skip],
        }
    return {
        "question": _FORK_QUESTION,
        "options": [mind, automate, machine, *tasks, figure, skip],
        "fallback_question": None,
        "fallback_options": [],
    }


def _hermes_home() -> Path | None:
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        home = os.environ.get("HERMES_HOME")
        return Path(home) if home else None
    return get_hermes_home()


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, default=str)
    os.replace(tmp, path)


def _started(profile: dict) -> datetime | None:
    try:
        started = datetime.fromisoformat(str(profile.get("run", {}).get("started")))
    except ValueError:
        return None
    return started if started.tzinfo else None


def _l1_fired(profile: dict) -> list[str]:
    return sorted(k for k, v in profile.get("facts", {}).items()
                  if isinstance(v, dict) and v.get("level") == "L1" and v.get("status") == "ok")


def _cache_is_fresh(profile: dict, version: str) -> bool:
    started = _started(profile)
    run = profile.get("run", {})
    return (started is not None and run.get("max_tier") == SCAN_TIER and run.get("collector_version") == version
            and 0 <= (datetime.now(timezone.utc) - started).total_seconds() < SCAN_CACHE_MAX_AGE_S)


def _scan_now() -> tuple[dict, str]:
    if str(SCAN_DIR) not in sys.path:
        sys.path.insert(0, str(SCAN_DIR))
    import userscan.specs  # noqa: F401
    from userscan import __version__
    from userscan.host import detect_os
    from userscan.registry import REGISTRY
    from userscan.runner import run

    home = _hermes_home()
    path = home / "insights" / "profile.json" if home else None
    cached = _read_json(path)
    if cached and _cache_is_fresh(cached, __version__):
        here = detect_os()
        l1 = run(max_tier=SCAN_TIER, only=sorted({p.id for p in REGISTRY.values()
                                                  if p.level == "L1" and p.os in ("any", here)}))
        if _l1_fired(l1) == _l1_fired(cached):
            return cached, "cache"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    profile = run(max_tier=SCAN_TIER)
    profile["run"]["started"] = started
    if path is not None:
        try:
            _write_private(path, profile)
        except OSError:
            pass
    return profile, "fresh"


def _scan() -> tuple[dict | None, str]:
    box: dict = {}

    def work() -> None:
        try:
            box["result"] = _scan_now()
        except Exception as exc:
            box["error"] = type(exc).__name__

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(SCAN_DEADLINE_S)
    if "result" in box:
        return box["result"]
    return None, box.get("error", "timeout")


def _value(profile: dict, fid: str) -> dict | None:
    record = profile.get("facts", {}).get(fid)
    if isinstance(record, dict) and record.get("status") == "ok" and isinstance(record.get("value"), dict):
        return record["value"]
    return None


def _answered(profile: dict, fid: str) -> bool:
    record = profile.get("facts", {}).get(fid)
    return isinstance(record, dict) and record.get("status") in ("ok", "absent", "skipped_flag")


def _insight(profile: dict, iid: str) -> dict | None:
    for item in profile.get("insights", []):
        if isinstance(item, dict) and item.get("id") == iid:
            return item.get("value") if isinstance(item.get("value"), dict) else {}
    return None


def _num(value) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _row_names(rows) -> list[str]:
    if not isinstance(rows, list):
        return []
    return [str(row[0]) for row in rows if isinstance(row, (list, tuple)) and row]


def _app_labels(names: list[str]) -> set[str]:
    found = set()
    for name in (n.lower().strip() for n in names):
        for label, tokens in _SCAN_APPS.items():
            if any(name == t[1:] if t.startswith("=") else t in name for t in tokens):
                found.add(label)
    return found


def _scan_apps(profile: dict) -> tuple[list[str], list[str], bool]:
    installed_names: list[str] = []
    used_names: list[str] = []
    usage_seen = False
    taxonomy = _value(profile, "apps.taxonomy")
    if taxonomy and isinstance(taxonomy.get("names"), dict):
        for names in taxonomy["names"].values():
            installed_names += [str(n) for n in names] if isinstance(names, list) else []
    focus = _value(profile, "userassist.focus")
    if focus:
        usage_seen = True
        used_names += _row_names(focus.get("top_focus_h")) + _row_names(focus.get("top_runs"))
        used_names += _row_names(focus.get("top"))
    last_run = _value(profile, "bam.last_run")
    if last_run and isinstance(last_run.get("top"), list):
        usage_seen = True
        used_names += _row_names(last_run["top"])
    for key, app in (_value(profile, "comms.native_apps") or {}).items():
        if isinstance(app, dict):
            installed_names += [key] if app.get("installed") else []
            used_names += [key] if app.get("launched") else []
    obs = _value(profile, "media.obs")
    if obs:
        installed_names.append("obs studio")
        used_names += ["obs studio"] if obs.get("config_dir") is True or obs.get("obs_configs") else []
    if _value(profile, "discord.present"):
        installed_names.append("discord")
    discord = _value(profile, "discord.usage")
    if discord and (_num(discord.get("gateway_days")) or 0) > 0:
        used_names.append("discord")
    if _value(profile, "steam.present"):
        installed_names.append("steam")
    sessions = _value(profile, "steam.local_sessions")
    if sessions and (_num(sessions.get("sessions")) or 0) > 0:
        used_names.append("steam")
    used = _app_labels(used_names)
    unused = _app_labels(installed_names) - used
    return sorted(used)[:10], sorted(unused)[:10], usage_seen


def _agent_evidence(profile: dict) -> list[str]:
    evidence = []
    power = _insight(profile, "persona.ai_power_user") or {}
    sessions = power.get("sessions") if isinstance(power.get("sessions"), dict) else {}
    for key, name in _AGENT_NAMES.items():
        count = _num(sessions.get(key))
        if count is not None and count >= 3:
            evidence.append(f"{name} {int(count)} sessions")
    hermes = _value(profile, "hermes.present") or {}
    side_homes = _num(hermes.get("side_homes_operator")) or 0
    if side_homes:
        evidence.append(f"{int(side_homes)} Hermes test homes")
    accounts = _value(profile, "acct.local_users") or {}
    kinds = accounts.get("enabled_kinds") if isinstance(accounts.get("enabled_kinds"), dict) else {}
    sandboxes = _num(kinds.get("agent_or_tool")) or 0
    if sandboxes:
        evidence.append(f"{int(sandboxes)} agent sandbox accounts")
    if _value(profile, "cua_driver.present"):
        evidence.append("computer-use driver")
    automation = _value(profile, "browser.automation") or {}
    if any(key in automation for key in ("playwright", "puppeteer", "camoufox")):
        evidence.append("browser automation")
    mcp = _value(profile, "l3.mcp_inventory") or {}
    servers = _num(mcp.get("user_configured_total")) or 0
    if servers:
        evidence.append(f"{int(servers)} MCP servers set up")
    return evidence


def interpret(profile: dict, source: str) -> dict:
    unknown = []
    state = _insight(profile, "install.install_state") or {}
    tenure = _num(state.get("tenure_days"))
    lived_in = _num(state.get("lived_in"))
    sophistication = (_insight(profile, "persona.sophistication") or {}).get("tier")
    level = {"novice": "beginner", "power-user": "power-user", "expert": "expert"}.get(sophistication, "unknown")
    developer = (_insight(profile, "persona.developer") or {}).get("developer")
    evidence = _agent_evidence(profile)
    agent_probes = ("hermes.present", "codex.present", "claude_code.present")
    runs_agents = bool(evidence) or ("unknown" if not any(_answered(profile, p) for p in agent_probes) else False)
    if level in ("power-user", "expert") or developer is True or runs_agents is True:
        beginner = False
    elif level == "beginner":
        beginner = True
    else:
        beginner = "unknown"

    timeline = _value(profile, "srum.app_timeline") or {}
    ratio = _num(timeline.get("input_focus_ratio"))
    if ratio is None:
        hands_on = "unknown"
        unknown.append("hands-on vs remote")
    else:
        hands_on = "remote-driven" if ratio < 0.05 else "hands-on" if ratio > 0.3 else "mixed"
    identity = (_value(profile, "dev.git_global_config") or {}).get("identity_set") is True
    if not (identity and _value(profile, "dev.repos.my_commits")):
        unknown.append("own commit count")

    used, unused, usage_seen = _scan_apps(profile)
    if not usage_seen:
        unknown.append("which apps get used")

    history = (_insight(profile, "install.user_history_elsewhere") or {}).get("user_history_elsewhere")
    split = _value(profile, "l3.power_event_split")
    crash = _num(split.get("crash_30d")) if split else None
    theme = _value(profile, "theme.dark")
    dark = theme.get("apps_dark") if theme else None
    browser = str((_value(profile, "l3.primary_browser") or {}).get("primary") or "").lower()
    started = _started(profile)
    block = {
        "source": source,
        "age_h": round((datetime.now(timezone.utc) - started).total_seconds() / 3600, 1) if started else "unknown",
        "tier": profile.get("run", {}).get("max_tier", "unknown"),
        "machine_state": state.get("install_state", "unknown"),
        "owned_days": int(tenure) if tenure is not None else "unknown",
        "lived_in_of_10": lived_in if lived_in is not None else "unknown",
        "history_before_this_install": history if isinstance(history, bool) else "unknown",
        "user_level": level,
        "developer": developer if isinstance(developer, bool) else "unknown",
        "beginner_framing": beginner,
        "runs_agents": runs_agents,
        "agent_evidence": evidence,
        "hands_on": hands_on,
        "apps_used": used,
        "apps_installed_no_use_seen": unused,
        "crash_30d": int(crash) if crash is not None else "unknown",
        "ui_theme": "unknown" if not isinstance(dark, bool) else "dark" if dark else "light",
        "browser": browser if browser in _BROWSERS else "unknown" if not browser else "other",
        "unknown": unknown,
        "not_visible_at_tier": _NOT_VISIBLE_T1 if profile.get("run", {}).get("max_tier") == SCAN_TIER else [],
    }
    sessions = _value(profile, "steam.local_sessions")
    if sessions:
        hours, recent = _num(sessions.get("total_hours")), _num(sessions.get("hours_30d"))
        block["games_here_h"] = round(hours) if hours is not None else "unknown"
        block["games_here_h_30d"] = round(recent) if recent is not None else "unknown"
    elif _value(profile, "steam.present"):
        block["games_here_h"] = "unknown"
    return block


def _machine_state(scan: dict | None, age: int | None) -> tuple[str, int | None]:
    if scan and scan.get("machine_state") in ("fresh", "settling", "established"):
        owned = scan.get("owned_days")
        return scan["machine_state"], owned if isinstance(owned, int) else age
    if age is None:
        return "unknown", None
    return ("fresh" if age <= NEW_MACHINE_DAYS else "settling" if age < SETTLING_DAYS else "established"), age


def collect(scan_json: Path | None = None) -> dict:
    """Return the fact block the ``/initiate-setup`` first turn embeds."""
    if scan_json is not None:
        profile, source = _read_json(scan_json), "file"
    else:
        profile, source = _scan()
    scan = interpret(profile, source) if profile else {"source": "unavailable", "reason": source}

    os_family = facts.os_family()
    arch = facts.native_arch()
    gpu = facts.gpu_class()
    cpu = facts.cpu_model()
    ram = facts.ram_total_bytes()
    release = platform.release()

    login, full, home = _account()
    age = _home_age_days(home)
    locale = _locale()

    state, setup_age = _machine_state(scan if profile else None, age)
    looks_new = state == "fresh"
    spark = _is_spark(os_family, arch, gpu, cpu)
    leads = spark or looks_new
    kind = _machine_kind(os_family, spark)
    plugin_tasks = [_NVIDIA_TASK, _BLENDER_TASK] if os_family == "win32" and gpu == "nvidia" else [_BLENDER_TASK]

    return {
        "schema_version": SCHEMA_VERSION,
        "machine": {
            "os_family": os_family,
            "os_release": release,
            "native_arch": arch,
            "cpu_model": cpu,
            "ram_gb": round(ram / 2**30) if ram else None,
            "gpu_class": gpu,
            "wsl": runtime.is_wsl(),
            "container": runtime.is_container(),
        },
        "account": {
            "suggested_name": _suggested_name(login, full),
            "locale": locale,
            "locale_is_english": not locale or locale.lower().startswith("en"),
            "home_age_days": age,
        },
        "signals": {
            "machine_kind": kind,
            "machine_state": state,
            "looks_new": looks_new,
            "is_spark": spark,
            "has_nvidia_gpu": gpu == "nvidia",
            "machine_setup_leads": leads,
            "description": _description(
                looks_new=looks_new, age=setup_age, spark=spark, gpu=gpu, cpu=cpu,
                os_family=os_family, release=release, arch=arch,
            ),
        },
        "plugin_tasks": plugin_tasks,
        "fork": _fork(kind, leads, plugin_tasks),
        "scan": scan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", type=Path, help="interpret this saved scan instead of scanning")
    args = parser.parse_args()
    print(json.dumps(collect(args.from_json), ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
