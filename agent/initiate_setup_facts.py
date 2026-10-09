"""Machine and account facts for the ``/initiate-setup`` first turn and the setup cards.

Hardware facts come from ``hermes_platform.host``. Account facts (full name, locale) come from OS
user records, never from environment variables such as HOME or LANG. They describe the machine
that runs the Hermes backend; when the desktop app drives a remote backend, that is not the
user's laptop.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import time
from pathlib import Path

from hermes_constants import get_optional_skills_dir
from hermes_platform.host import facts as host
from hermes_platform.host import products, runtime

# Generic account names that are not a person's name.
_NON_NAMES = frozenset({
    "admin", "administrator", "default", "guest", "me", "owner", "root", "test", "user",
})
_HANDLE_CHARS = re.compile(r"[\d_@/\\]")
_SPARK_MODEL = re.compile(r"\b(dgx|spark|gb10)\b", re.IGNORECASE)

FORK_QUESTION = "Know what you'd like it to make?"
# Reading the Blender plugin's app declaration takes the network.
_APP_STATE_DEADLINE_S = 5

_BLENDER_TASK = {"id": "plugin:blender", "label": "Help me make something in Blender", "plugins": ["blender"]}
_NVIDIA_TASK = {
    "id": "plugin:nvidia",
    "label": "Set up my games and streaming",
    "plugins": ["nvidia-app", "nvidia-broadcast"],
}


def skill_dir() -> Path:
    return get_optional_skills_dir(Path(__file__).resolve().parent.parent / "optional-skills") / "productivity" / "initiate-setup"


# --- account facts -----------------------------------------------------------------


def _posix_account() -> tuple[str, str]:
    """Return (login, full name) from the user database."""
    import pwd

    entry = pwd.getpwuid(os.getuid())  # windows-footgun: ok — only called off Windows (_account)
    return entry.pw_name, entry.pw_gecos.split(",", 1)[0].strip()


def _windows_account() -> tuple[str, str]:
    """Return (login, display name) from Win32 account APIs."""
    import ctypes
    from ctypes import wintypes

    login_buf = ctypes.create_unicode_buffer(257)
    login_len = wintypes.DWORD(len(login_buf))
    login = login_buf.value if ctypes.windll.advapi32.GetUserNameW(login_buf, ctypes.byref(login_len)) else ""

    # EXTENDED_NAME_FORMAT NameDisplay = 3. Local accounts without a display name fail here.
    name_buf = ctypes.create_unicode_buffer(257)
    name_len = wintypes.ULONG(len(name_buf))
    full = name_buf.value if ctypes.windll.secur32.GetUserNameExW(3, name_buf, ctypes.byref(name_len)) else ""
    return login, full


def _account() -> tuple[str, str]:
    try:
        return _windows_account() if sys.platform == "win32" else _posix_account()
    except (AttributeError, ImportError, KeyError, OSError):
        return "", ""


def _suggested_name(login: str, full: str) -> str | None:
    """A real full name only; a login handle is never offered as the user's name."""
    name = " ".join(full.split())
    if not (2 <= len(name) <= 40) or name.lower() in _NON_NAMES or name.lower() == login.lower():
        return None
    # Digits or underscores ("p14", "CD_01.05") or an all-lowercase cased name mark a handle.
    if _HANDLE_CHARS.search(name) or name == name.lower() != name.upper():
        return None
    return name


def suggested_name() -> str | None:
    """The row ``setup_choose`` adds to the name card. It stays on this computer unless the user picks it."""
    return _suggested_name(*_account())


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
            with open(path, encoding="utf-8-sig") as handle:
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


# --- the facts block ------------------------------------------------------------------


def _is_spark(os_family: str, arch: str, gpu: str, cpu: str) -> bool:
    """RTX Sparks by platform, architecture and GPU; DGX Sparks by model string."""
    rtx = os_family == "win32" and arch == "arm64" and gpu == "nvidia"
    return rtx or products.is_nvidia_arm_soc() or bool(_SPARK_MODEL.search(cpu.replace("_", " ")))


def _machine_kind(os_family: str, spark: bool) -> str:
    if spark:
        return "Spark"
    return {"darwin": "Mac", "win32": "PC"}.get(os_family, "computer")


def facts() -> dict:
    """The facts the ``/initiate-setup`` turn sends, read as they are. A key that was not measured is left out.
    The account's full name is not one of them: ``setup_choose`` adds it to the name card itself, so it reaches the
    model only when the user picks it."""
    os_family, arch, gpu, cpu = host.os_family(), host.native_arch(), host.gpu_class(), host.cpu_model()
    ram = host.ram_total_bytes()
    spark = _is_spark(os_family, arch, gpu, cpu)
    machine = {
        "os_family": os_family,
        "os_release": platform.release(),
        "native_arch": arch,
        "cpu_model": cpu,
        "ram_gb": round(ram / 2**30) if ram else None,
        "gpu_class": None if gpu == "unknown" else gpu,
        "vendor": host.cpu_vendor(),
        "wsl": runtime.is_wsl(),
        "container": runtime.is_container(),
    }
    block = {
        "machine": {key: value for key, value in machine.items() if value not in (None, "")},
        "machine_kind": _machine_kind(os_family, spark),
        "has_nvidia_gpu": gpu == "nvidia",
        "is_spark": spark,
        "locale": _locale(),
    }
    return {key: value for key, value in block.items() if value not in (None, "")}


# --- setup cards ----------------------------------------------------------------------


def _fork(kind: str) -> dict:
    """The fork card's fixed rows; ``fork_card`` puts first tasks from the setup picks in front. Ids are stable;
    labels may be translated."""
    return {"question": FORK_QUESTION, "options": [
        {"id": "mind", "label": "I have something in mind"},
        {"id": "machine", "label": f"Help me set up this {kind}"},
        {"id": "figure", "label": "Let's figure it out together"},
    ]}


# Connector picks a daily brief has nothing to read from.
_NOT_BRIEF = frozenset({"discord", "spotify", "steam", "telegram", "whatsapp", "youtube", "twitch", "reddit"})


def _from_picks(cards: dict) -> list[dict]:
    picks, ids = cards.get("picks") or {}, cards.get("pick_ids") or {}
    apps = [str(label) for label in picks.get("connectors") or () if str(label).lower() not in _NOT_BRIEF]
    plugins = set(ids.get("plugins") or ())
    rows = [{"id": "brief", "label": f"A daily brief from {' and '.join(apps[:2])}"}] if apps else []
    rows += [{"id": task_id, "label": task["label"]} for task_id, task in (cards.get("plugin_tasks") or {}).items()
             if plugins & set(task["plugins"])]
    return rows + ([{"id": "week", "label": "A summary of my week"}] if apps else [])


def _from_machine(suggest: dict) -> list[dict]:
    # No local-model row: the app's own offer sets that up on the managed runtime; a task chat would improvise one.
    kind = suggest.get("kind") or "computer"
    return [{"id": "apps", "label": f"Install a few apps for this {kind}"}] if suggest.get("spark") else []


def _from_nothing(suggest: dict) -> list[dict]:
    page = f"A one-page guide to this {suggest.get('kind') or 'computer'}"
    return [{"id": "page", "label": page}, {"id": "script", "label": "A quick script that tidies my Downloads"}]


def fork_card(cards: dict) -> dict:
    """The fork card when it is shown: two first tasks, each finishable in minutes, from the apps and plugins
    picked earlier in this setup and the machine, in front of the fixed rows."""
    suggest = cards.get("suggest") or {}
    picks, machine = _from_picks(cards), _from_machine(suggest)
    ordered = picks[:1] + machine[:1] + picks[1:] + machine[1:] + _from_nothing(suggest)
    fork = cards.get("fork") or _fork(suggest.get("kind") or "computer")
    return {**fork, "options": ordered[:2] + fork["options"]}


def _app_states(names: list[str]) -> dict[str, str]:
    """Each catalog plugin's app state as the plugins card and the installer judge it: the resolver over the
    plugin's pinned ``app:`` declaration. Read in parallel; ``unknown`` for any that takes longer than the deadline."""
    from agent.memory_provider import spawn_context_thread
    from hermes_cli.plugin_catalog import get_live_catalog_entry
    from hermes_cli.plugin_catalog_presence import presence

    states: dict[str, str] = {}

    def read(name: str) -> None:
        entry = get_live_catalog_entry(name)
        states[name] = presence(entry).state if entry else "unknown"

    workers = [spawn_context_thread(lambda n=name: read(n), name=f"initiate-setup-{name}") for name in names]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + _APP_STATE_DEADLINE_S
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    return {name: states.get(name, "unknown") for name in names}


def _plugin_tasks(states: dict[str, str]) -> list[dict]:
    """The fork's plugin rows, each offered only when the app its plugins need is there (*states* from
    ``_app_states``): the installer refuses a plugin whose app is missing, so the card never offers one."""
    tasks = []
    nvidia_plugins = [name for name in _NVIDIA_TASK["plugins"] if states.get(name, "missing_app") != "missing_app"]
    if nvidia_plugins:
        tasks.append({**_NVIDIA_TASK, "plugins": nvidia_plugins})
    if states["blender"] != "missing_app":
        tasks.append(_BLENDER_TASK)
    return tasks


def _description(block: dict) -> str:
    machine = block.get("machine") or {}
    parts = [
        "an NVIDIA Spark" if block.get("is_spark") else "has an NVIDIA GPU" if block.get("has_nvidia_gpu") else "",
        machine.get("cpu_model", ""),
        f"{machine.get('os_family', '')} {machine.get('os_release', '')}".strip(),
        machine.get("native_arch", ""),
    ]
    return ", ".join(part for part in parts if part)


def _handoff(kind: str, description: str) -> dict:
    """The handoff message's parts and its two plans from ``templates/handoff.md``, the machine plan naming this
    computer, so the model reads them only when it reaches the handoff."""
    text = (skill_dir() / "templates" / "handoff.md").read_text(encoding="utf-8-sig")
    sections = dict(re.findall(r"^## (\S+)\n\n(.*?)\n*(?=^## |\Z)", text, re.MULTILINE | re.DOTALL))
    machine = sections["machine"].replace("<machine_kind>", kind).replace("<description>", description)
    return {"message": sections["message"], "build": sections["build"], "machine": machine}


def setup_cards(block: dict) -> dict:
    """What the setup cards take from the facts: the fork rows and what its first tasks are built from, the
    Blender preselect, the handoff text the fork result carries, and the machine line ``start_chat`` appends
    to the task chat's first message."""
    kind, description = block["machine_kind"], _description(block)
    nvidia = block.get("has_nvidia_gpu") and block["machine"].get("os_family") == "win32"
    states = _app_states(["blender", *(_NVIDIA_TASK["plugins"] if nvidia else [])])
    plugin_tasks, blender = _plugin_tasks(states), states["blender"]
    ram = f", {block['machine']['ram_gb']} GB RAM" if block["machine"].get("ram_gb") else ""
    return {
        "learned": [f"This {kind}: {description}{ram}."],
        "fork": _fork(kind),
        # What ``fork_card`` builds first tasks from, beside the picks the cards record.
        "suggest": {"kind": kind, "spark": bool(block.get("is_spark"))},
        "plugin_tasks": {task["id"]: {"label": task["label"], "plugins": task["plugins"]} for task in plugin_tasks},
        "preselected": {"connectors": [], "plugins": ["blender"] if blender in ("present", "app_not_running") else []},
        "handoff": _handoff(kind, description),
    }
