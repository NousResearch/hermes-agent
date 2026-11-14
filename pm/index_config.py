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
import json
import os
from pathlib import Path
import re
import sys

# Explicit uv index / transport settings that survive into uv. UV_INDEX_<NAME>_
# {USERNAME,PASSWORD} credentials match by prefix in is_forwarded().
FORWARDED_UV_SETTINGS = frozenset({
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX", "UV_NO_INDEX",
    "UV_FIND_LINKS", "UV_INDEX_STRATEGY", "UV_KEYRING_PROVIDER",
    "UV_NATIVE_TLS", "UV_INSECURE_HOST", "UV_HTTP_TIMEOUT",
})

_UV_INDEX_KNOBS = ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX")

# pip knob → uv knob, applied only when uv has no value of its own.
_PIP_TO_UV = (
    ("PIP_EXTRA_INDEX_URL", "UV_EXTRA_INDEX_URL"),
    ("PIP_TRUSTED_HOST", "UV_INSECURE_HOST"),
)

TIMEOUT_HINT = ("uv timed out. If your network needs a package mirror, set index-url in "
                "pip.conf (bridged to uv automatically) or UV_INDEX_URL; raise UV_HTTP_TIMEOUT "
                "for slow links.")
ISOLATED_TIMEOUT_HINT = (
    "uv timed out. This isolated operation ignores package-index overrides; changing a mirror "
    "setting will not redirect its downloads. Check connectivity to the pinned artifact URLs "
    "and your proxy/CA configuration; UV_HTTP_TIMEOUT is still honored for slow HTTP requests."
)


def is_forwarded(key: str) -> bool:
    return key in FORWARDED_UV_SETTINGS or key.startswith("UV_INDEX_")


# Settings that redirect WHERE packages resolve from (index URLs, find-links,
# resolution strategy, index credentials). These are what no_config isolation
# must strip from the child: re-resolving the official-index runtime lockfile
# against a mirror is exactly what trips `uv sync --locked` (#124418/#125071).
# Transport knobs (UV_NATIVE_TLS, UV_INSECURE_HOST, UV_HTTP_TIMEOUT) are NOT
# index redirects — corporate networks need them to reach the pinned URLs at
# all, so they survive no_config isolation.
INDEX_REDIRECT_SETTINGS = frozenset({
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX", "UV_NO_INDEX",
    "UV_FIND_LINKS", "UV_INDEX_STRATEGY", "UV_KEYRING_PROVIDER",
})


def is_index_redirect(key: str) -> bool:
    return key in INDEX_REDIRECT_SETTINGS or key.startswith("UV_INDEX_")


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


NPM_PUBLIC_REGISTRY = "https://registry.npmjs.org/"


def hermes_npmrc() -> Path | None:
    """The profile's ``$HERMES_HOME/npmrc`` when it exists."""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "npmrc"
    return path if path.is_file() else None


def user_npmrc(env: Mapping[str, str]) -> Path:
    """The user npmrc Hermes's npm and pnpm children read: a non-empty ``npm_config_userconfig``
    (that exact name first, as pnpm and the desktop read it, then any case), else ``hermes_npmrc()``
    (``source_build_env`` points ``NPM_CONFIG_USERCONFIG`` at it), else ``~/.npmrc``. As npm's
    userconfig, the Hermes file replaces ``~/.npmrc``; the two never merge."""
    names = sorted((key for key in env if key.lower() == "npm_config_userconfig"), key=lambda key: key != "npm_config_userconfig")
    explicit = next((env[key] for key in names if env[key]), "")
    if explicit:
        return Path(explicit).expanduser()
    return hermes_npmrc() or Path.home() / ".npmrc"


# npm's line grammar (the ``ini`` package @npmcli/config parses with) and its ${VAR} / ${VAR?}
# expansion (@npmcli/config/lib/env-replace.js), backslash escapes included.
_INI_LINE = re.compile(r"^\[([^\]]*)\]\s*$|^([^=]+)(=(.*))?$")
_INI_COMMENT = re.compile(r"^\s*[;#]")
_NPMRC_ENV_REF = re.compile(r"(?<!\\)(\\*)\$\{([^${}?]+)(\?)?\}")


def _ini_unsafe(raw: str) -> str:
    """One ini key or value as ``ini.unsafe`` decodes it: quoted text is unquoted (JSON
    escapes apply), anything else ends at the first unescaped ``;`` or ``#``."""
    val = raw.strip()
    if val and val[0] == val[-1] and val[0] in "\"'":
        if val[0] == "'":
            val = val[1:-1]
        try:
            parsed = json.loads(val)
        except ValueError:
            return val
        return parsed if isinstance(parsed, str) else val
    out, escaped = [], False
    for char in val:
        if escaped:
            out.append(char if char in "\\;#" else "\\" + char)
            escaped = False
        elif char in ";#":
            break
        elif char == "\\":
            escaped = True
        else:
            out.append(char)
    return ("".join(out) + ("\\" if escaped else "")).strip()


def _npmrc_env_replace(text: str, env: Mapping[str, str]) -> str | None:
    """*text* with ``${VAR}`` expanded; ``${VAR?}`` of an unset variable is empty. ``None`` when
    a plain ``${VAR}`` is unset: npm would keep the literal, which no consumer can use."""
    unset = False

    def expand(ref: re.Match) -> str:
        nonlocal unset
        escapes, name, optional = ref.groups()
        if len(escapes) % 2:
            return ref.group(0)[(len(escapes) + 1) // 2:]
        value = env.get(name)
        if value is None:
            unset = unset or not optional
            value = ""
        return escapes[len(escapes) // 2:] + value

    expanded = _NPMRC_ENV_REF.sub(expand, text)
    return None if unset else expanded


def read_npmrc(path: Path, env: Mapping[str, str]) -> dict[str, str]:
    """The top-level settings of an npmrc, valued as npm hands them to scripts.

    npm reads the file as UTF-8 (lossily, so a UTF-16 file from Windows PowerShell's ``>``
    yields only unusable keys rather than an error), ignores everything under an ``[section]``,
    turns ``true``/``false``/``null`` and a bare key into ``"true"``/``""``/``""``/``"true"``,
    and expands ``${VAR}`` in keys and values. Array lines (``key[]=``), entries naming an
    unset variable and entries no child environment can carry (``_spawnable``) are left out; a
    missing or unreadable file reads as empty.
    """
    try:
        text = path.read_bytes().decode("utf-8", errors="replace").removeprefix("\ufeff")
    except OSError:
        return {}
    settings: dict[str, str] = {}
    for line in re.split(r"[\r\n]+", text):
        match = _INI_LINE.match(line) if line.strip() and not _INI_COMMENT.match(line) else None
        if match is None:
            continue
        if match.group(1) is not None:
            break  # every later line belongs to a section, which npm never reads as a setting
        key = _ini_unsafe(match.group(2))
        if len(key) > 2 and key.endswith("[]"):
            continue
        value = _ini_unsafe(match.group(4)) if match.group(3) else "true"
        key, value = _npmrc_env_replace(key, env), _npmrc_env_replace("" if value in ("false", "null") else value, env)
        if key is not None and value is not None and _spawnable(key) and _spawnable(value):
            settings[key] = value
    return settings


def _spawnable(text: str) -> bool:
    """Whether *text* can travel in a child's environment: no NUL (a raw byte or a ``\\u0000``
    escape) and no lone surrogate (a ``\\ud800`` escape), either of which fails the spawn."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return "\x00" not in text


def npm_registry(env: Mapping[str, str]) -> str:
    """The npm registry the user configured, as npm resolves it: ``npm_config_registry`` (any
    case) beats ``registry=`` in the user npmrc (``user_npmrc``)."""
    lowered = {key.lower(): value for key, value in env.items()}
    value = (lowered.get("npm_config_registry") or "").strip()
    if not value:
        value = read_npmrc(user_npmrc(env), env).get("registry", "")
    return value.rstrip("/") + "/" if value.startswith(("https://", "http://")) else NPM_PUBLIC_REGISTRY


def npm_registry_url(url: str, env: Mapping[str, str]) -> str:
    """*url* on the configured npm registry; non-npm URLs are returned unchanged."""
    if not url.startswith(NPM_PUBLIC_REGISTRY):
        return url
    return npm_registry(env) + url[len(NPM_PUBLIC_REGISTRY):]


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
