"""Forward the user's package-index and transport configuration into uv.

PM strips ambient ``UV_*`` so a caller's uv settings cannot steer which
project, interpreter or cache an operation uses (pm.environment). Index and
transport knobs are different: on mirrored or air-gapped networks they are the
only way any dependency resolves at all (#88453, #94613, #95608). Only those
cross the boundary; the lockfile stays authoritative for what gets installed.

uv never reads pip's configuration, so a pip-only mirror (``PIP_INDEX_URL`` or
``index-url`` in pip.conf) is bridged to ``UV_INDEX_URL`` unless uv already has
an index of its own. Stdlib only: the bootstrap runner imports this before any
dependency exists.
"""
from __future__ import annotations

from collections.abc import Mapping
import configparser
import os
from pathlib import Path
import sys

# Explicit uv index / transport settings that survive into uv. UV_INDEX_<NAME>_
# {USERNAME,PASSWORD} credentials match by prefix in is_forwarded().
FORWARDED_UV_SETTINGS = frozenset({
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX", "UV_NO_INDEX",
    "UV_FIND_LINKS", "UV_INDEX_STRATEGY", "UV_KEYRING_PROVIDER",
    "UV_NATIVE_TLS", "UV_INSECURE_HOST", "UV_HTTP_TIMEOUT",
})

_UV_INDEX_KNOBS = ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX")

# Only these replace the default index; UV_INDEX adds one alongside it.
_DEFAULT_INDEX_KNOBS = ("UV_INDEX_URL", "UV_DEFAULT_INDEX")
_PYPI_DEFAULT_INDEX = "https://pypi.org/simple"

# pip knob → uv knob, applied only when uv has no value of its own.
_PIP_TO_UV = (
    ("PIP_EXTRA_INDEX_URL", "UV_EXTRA_INDEX_URL"),
    ("PIP_TRUSTED_HOST", "UV_INSECURE_HOST"),
)

TIMEOUT_HINT = ("uv timed out. If your network needs a package mirror, set index-url in "
                "pip.conf (bridged to uv automatically) or UV_INDEX_URL; raise UV_HTTP_TIMEOUT "
                "for slow links.")


def is_forwarded(key: str) -> bool:
    return key in FORWARDED_UV_SETTINGS or key.startswith("UV_INDEX_")


def default_index_override(settings: Mapping[str, str]) -> str | None:
    """The non-PyPI default index *settings* carry, if any.

    A mirrored default index (bridged from pip or set directly) re-points every
    resolution away from PyPI, so a lockfile generated against PyPI — as the
    committed one is — no longer matches what uv would resolve. ``--locked``
    then rejects it (#122112). Callers staging that lock into a workspace they
    own re-resolve instead; extra indexes (UV_INDEX) don't move the default,
    so they can't desync the lock and don't count.
    """
    for key in _DEFAULT_INDEX_KNOBS:
        value = (settings.get(key) or "").strip().rstrip("/")
        if value and value != _PYPI_DEFAULT_INDEX:
            return value
    return None


def pip_config_candidates(env: Mapping[str, str]) -> list[Path]:
    """pip's config files, lowest precedence first, as ``pip._internal.configuration`` ranks them.

    Global, then user (skipped entirely when ``PIP_CONFIG_FILE`` names an existing file), then
    the interpreter's ``sys.prefix`` site file, then ``PIP_CONFIG_FILE`` itself on top.
    ``RawConfigParser.read`` applies them in order, so the last file wins.
    ``PIP_CONFIG_FILE=os.devnull`` disables all of them.
    """
    explicit = env.get("PIP_CONFIG_FILE", "")
    if explicit == os.devnull:
        return []
    home = Path.home()
    if sys.platform == "win32":
        name = "pip.ini"
        global_files = [Path(env.get("ProgramData") or r"C:\ProgramData") / "pip" / name]
        user_files = [home / "pip" / name,
                      Path(env.get("APPDATA") or home / "AppData" / "Roaming") / "pip" / name]
    elif sys.platform == "darwin":
        name = "pip.conf"
        global_files = [Path("/Library/Application Support/pip") / name]
        app_support = home / "Library" / "Application Support" / "pip"
        user_files = [home / ".pip" / name,
                      (app_support if app_support.is_dir() else home / ".config" / "pip") / name]
    else:
        name = "pip.conf"
        xdg_dirs = (env.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(os.pathsep)
        global_files = [Path(d) / "pip" / name for d in xdg_dirs if d] + [Path("/etc") / name]
        user_files = [home / ".pip" / name, Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "pip" / name]
    explicit_files = [Path(explicit)] if explicit else []
    if explicit_files and explicit_files[0].is_file():
        user_files = []
    return global_files + user_files + [Path(sys.prefix) / name] + explicit_files


def pip_conf_index_url(env: Mapping[str, str]) -> str | None:
    # Raw: pip does not interpolate, and mirror URLs carry percent-encoded credentials.
    parser = configparser.RawConfigParser()
    try:
        parser.read(str(path) for path in pip_config_candidates(env))
        if not parser.has_section("global"):
            return None
        return parser.get("global", "index-url", fallback="").strip() or None
    except configparser.Error:
        return None


def bridged_index_settings(ambient: Mapping[str, str]) -> dict[str, str]:
    """The uv index/transport settings *ambient* asks for, pip knobs translated.

    ``PIP_INDEX_URL`` beats pip.conf, as in pip; any explicit uv index knob beats both.
    """
    settings = {key: value for key, value in ambient.items() if is_forwarded(key)}
    if not any(settings.get(key) for key in _UV_INDEX_KNOBS):
        index_url = (ambient.get("PIP_INDEX_URL") or "").strip() or pip_conf_index_url(ambient)
        if index_url:
            settings["UV_INDEX_URL"] = index_url
    for pip_key, uv_key in _PIP_TO_UV:
        value = (ambient.get(pip_key) or "").strip()
        if value and not settings.get(uv_key):
            settings[uv_key] = value
    return settings
