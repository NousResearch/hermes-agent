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

import json
import os
import platform
import re
import sys
import time

from hermes_platform.host import facts, products, runtime

SCHEMA_VERSION = 1

# 21 days leaves time to finish setup without counting a daily-use machine as new.
NEW_MACHINE_DAYS = 21

# Generic account names that are not a person's name.
_NON_NAMES = frozenset({
    "admin", "administrator", "default", "guest", "me", "owner", "root", "test", "user",
})

_SPARK_MODEL = re.compile(r"\b(dgx|spark|gb10)\b", re.I)

_FORK_QUESTION = "Know what you'd like it to make?"
_FALLBACK_QUESTION = "What sounds better?"

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


def collect() -> dict:
    """Return the fact block the ``/initiate-setup`` first turn embeds."""
    os_family = facts.os_family()
    arch = facts.native_arch()
    gpu = facts.gpu_class()
    cpu = facts.cpu_model()
    ram = facts.ram_total_bytes()
    release = platform.release()

    login, full, home = _account()
    age = _home_age_days(home)
    locale = _locale()

    looks_new = age is not None and age <= NEW_MACHINE_DAYS
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
            "looks_new": looks_new,
            "is_spark": spark,
            "has_nvidia_gpu": gpu == "nvidia",
            "machine_setup_leads": leads,
            "description": _description(
                looks_new=looks_new, age=age, spark=spark, gpu=gpu, cpu=cpu,
                os_family=os_family, release=release, arch=arch,
            ),
        },
        "plugin_tasks": plugin_tasks,
        "fork": _fork(kind, leads, plugin_tasks),
    }


def main() -> None:
    print(json.dumps(collect(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
