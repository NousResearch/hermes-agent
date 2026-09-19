"""Offline Honcho snapshot preparation, before the host publishes a profile.

This intentionally does not use the runtime config/secret resolver: a scoped miss
in a single-profile process still falls back to the launching account's env.
Keep root and selected-host layers separate: aliases and empty values have
field-specific precedence, so flattening them changes runtime behavior.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from agent.secret_scope import _decode_env_bytes, _parse_env_text
from hermes_cli.profiles import _get_default_hermes_home

from .client import HOST, _HostLookup, _behavior_fields, _first_set, _host_block, profile_host_key


def _read_optional(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        if path.is_symlink():
            raise ValueError("Dangling configuration link") from None
        return None


def _source_config(home: Path) -> tuple[dict, bool]:
    for path in dict.fromkeys((home / "honcho.json", _get_default_hermes_home() / "honcho.json",
                               Path.home() / ".honcho" / "config.json")):
        content = _read_optional(path)
        if content is not None:
            raw = json.loads(content.decode("utf-8-sig"))
            if not isinstance(raw, dict) or not isinstance(raw.get("hosts", {}), dict):
                raise ValueError("Malformed Honcho configuration")
            return raw, True
    return {}, False


# Preserve runtime scalar coercion (including false/null/zero and blank timeout
# aliases), but never carry arbitrary objects through a recognized setting name.
_SCALAR = (str, bool, int, float)
_STRING = (str,)
_SETTING_SHAPES = {
    **dict.fromkeys(("workspace", "peerName", "environment", "baseUrl", "base_url",
                     "runtimePeerPrefix", "dialecticReasoningLevel", "reasoningLevelCap",
                     "recallMode", "injectionFrequency", "observationMode", "sessionStrategy"), _STRING),
    **dict.fromkeys(("enabled", "timeout", "requestTimeout", "pinUserPeer", "pinPeerName",
                     "saveMessages", "writeFrequency", "contextTokens", "dialecticDynamic",
                     "dialecticMaxChars", "dialecticDepth", "reasoningHeuristic", "messageMaxChars",
                     "dialecticMaxInputChars", "recallSync", "initOnSessionStart", "contextCadence",
                     "dialecticCadence", "queryRewrite", "firstTurnBaseWait", "firstTurnDialecticWait",
                     "sessionPeerPrefix", "a2aSessions", "sessionAiPeerPrefix", "logging"), _SCALAR),
    "dialecticDepthLevels": [_STRING],
    "userPeerAliases": {_STRING: _STRING},
    "observation": {"user": {"observeMe": _SCALAR, "observeOthers": _SCALAR},
                    "ai": {"observeMe": _SCALAR, "observeOthers": _SCALAR}},
    "injection": {"sessionStart": [_STRING]},
}


def _safe_setting(value: Any, shape: Any) -> Any:
    if value is None:
        return None
    if isinstance(shape, tuple) and isinstance(value, shape):
        return value
    if isinstance(shape, list) and isinstance(value, list):
        return [_safe_setting(item, shape[0]) for item in value]
    if isinstance(shape, dict) and isinstance(value, dict):
        if _STRING in shape:  # User-defined alias names, not an open nested object.
            return {key: _safe_setting(item, shape[_STRING]) for key, item in value.items()}
        return {key: _safe_setting(item, shape[key]) for key, item in value.items() if key in shape}
    raise ValueError("Malformed Honcho setting")


def _settings(layer: dict) -> dict:
    return _safe_setting(layer, _SETTING_SHAPES)


def _oauth(value) -> bool:
    return isinstance(value, str) and value.strip().startswith(("hch-at-", "hch-rt-"))


def _validate_auth(layer: dict) -> None:
    # Validate even falsy and shadowed keys before precedence can hide malformed
    # credentials. OAuth objects are never copied, only used as grant provenance.
    _safe_setting(layer.get("apiKey"), _STRING)
    if layer.get("oauth") is not None and not isinstance(layer["oauth"], dict):
        raise ValueError("Malformed Honcho authentication")


def _atomic_private(path: Path, text: str) -> None:
    # replace() replaces a leaf symlink rather than opening/chmodding its target.
    fd, temporary = tempfile.mkstemp(prefix=".honcho-clone-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _check_staging_entry(path: Path, *, replace_leaf: bool = False) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    # An ordinary JSON file symlink is replaced atomically, never read. Reject
    # junctions/other reparse points even on Python versions without is_junction.
    if stat.S_ISLNK(info.st_mode) and replace_leaf:
        return
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError("Staging link or reparse point")


def _prepare_env(path: Path, values: dict[str, str]) -> str:
    from hermes_cli.config import _env_line_defines_key, _quote_env_value

    text = _decode_env_bytes(_read_optional(path) or b"")
    lines = [line for line in text.splitlines(keepends=True)
             if not any(_env_line_defines_key(line, key) for key in values)]
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    # Tombstone absent connection fields as a unit: preserving a static key but
    # borrowing the caller's endpoint would send that key to another server.
    lines.extend(f"{key}={_quote_env_value(value)}\n" for key, value in values.items())
    return "".join(lines)


def prepare_clone(*, source_home: Path, source_name: str, staging_home: Path,
                  destination_home: Path, destination_name: str, clone_all: bool) -> dict | None:
    """Materialize safe native settings; never provision peers or refresh grants.

    Both clone modes use the same credential policy: root/env static keys may
    travel, host-only keys and all OAuth grants may not. Removing effective auth
    disables the clone instead of silently selecting a lower-priority account.
    The caller owns rollback of the private staging directory on any exception.
    """
    try:
        source_home, staging_home = Path(source_home), Path(staging_home)
        if not staging_home.name or ".." in staging_home.parts:
            raise ValueError("Clone requires a private staging directory")
        # The caller owns the parent. Resolve its aliases (including macOS /tmp
        # and /var), then check the unpublished root and every Honcho-owned leaf.
        # Never canonicalize the staging root itself before checking for links.
        staging_home = staging_home.parent.resolve(strict=True) / staging_home.name
        _check_staging_entry(staging_home)
        if staging_home.resolve() in {source_home.resolve(), Path(destination_home).resolve()}:
            raise ValueError("Clone requires a private staging directory")
        raw, has_file = _source_config(source_home)
        env = _parse_env_text(_decode_env_bytes(_read_optional(source_home / ".env") or b""))
        relevant_env = ("HONCHO_API_KEY", "HONCHO_BASE_URL", "HONCHO_URL", "HONCHO_ENVIRONMENT",
                        "HERMES_HONCHO_HOST")
        if not has_file and not any(env.get(key) for key in relevant_env):
            return None  # Installed but never configured: do not alter the host's clone baseline.
        _check_staging_entry(staging_home / ".env")
        _check_staging_entry(staging_home / "honcho.json", replace_leaf=True)
        host = (env.get("HERMES_HONCHO_HOST") or "").strip() or profile_host_key(source_name)
        if not (env.get("HERMES_HONCHO_HOST") or "").strip() and host == HOST:
            host = str(raw.get("defaultHost", "")).strip() or host
        block = _host_block(raw, host)
        if not isinstance(block, dict):
            raise ValueError("Malformed Honcho host")
        for layer in (raw, block):
            _validate_auth(layer)
            _safe_setting(layer.get("endpoint"), {"baseUrl": _STRING})
        look = _HostLookup(block, raw)
        key = look.pick("apiKey") or env.get("HONCHO_API_KEY")
        # A grant without an access token is still this account's login. Do not
        # turn stripping its metadata into a fallback to another account's key.
        needs_auth = bool(block.get("apiKey")) or any(
            layer.get("oauth") is not None for layer in (block, raw)
        ) or _oauth(key)
        root, selected = _settings(raw), _settings(block)
        if isinstance(raw.get("endpoint"), dict) and "baseUrl" in raw["endpoint"]:
            root["endpoint"] = {"baseUrl": raw["endpoint"]["baseUrl"]}
        if raw.get("apiKey") and not needs_auth and not _oauth(raw["apiKey"]):
            root["apiKey"] = raw["apiKey"]
        # Native JSON defaults workspace to HOST *of the selected source*. Env-only
        # resolution defaults to 'hermes'. Materialize it to keep the shared workspace
        # rather than silently moving the clone to its new host's empty workspace.
        selected["workspace"] = (look.pick("workspace") or host) if has_file else HOST
        selected["aiPeer"] = destination_name
        selected["sessionAiPeerPrefix"] = True
        behavior = _behavior_fields(look, bool(block) or raw.get("enabled") is True)
        selected["observationMode"] = behavior["observation_mode"]
        # Filtering a truthy host object to {} changes fallback to the root.
        # Persist the runtime-resolved fields, including preset-derived values.
        selected["observation"] = {
            peer: {field: behavior[f"{peer}_observe_{who}"]
                   for field, who in (("observeMe", "me"), ("observeOthers", "others"))}
            for peer in ("user", "ai")
        }
        if not has_file:
            selected["environment"] = env.get("HONCHO_ENVIRONMENT") or "production"
        endpoint = root.get("endpoint") or {}
        base_url = (block.get("baseUrl") or block.get("base_url") or endpoint.get("baseUrl")
                    or raw.get("baseUrl") or raw.get("base_url")
                    or (env.get("HONCHO_BASE_URL") or "").strip() or (env.get("HONCHO_URL") or "").strip())
        if base_url and not (block.get("baseUrl") or block.get("base_url") or endpoint.get("baseUrl")
                             or raw.get("baseUrl") or raw.get("base_url")):
            root["baseUrl"] = base_url
        # Match the runtime URL rejection without logging a potentially secret URL.
        usable_url = base_url and all(0x20 <= ord(c) < 0x7F for c in base_url)
        enabled = _first_set(*look.vals("enabled"), default=bool(key or usable_url))
        selected["enabled"] = False if needs_auth else enabled
        destination_host = profile_host_key(destination_name)
        root["hosts"] = {destination_host: selected}
        env_key = env.get("HONCHO_API_KEY") or ""
        connection_env = {name: env.get(name) or "" for name in relevant_env}
        connection_env["HONCHO_API_KEY"] = "" if needs_auth or _oauth(env_key) else env_key
        connection_env["HERMES_HONCHO_HOST"] = destination_host
        env_text = _prepare_env(staging_home / ".env", connection_env)
        _atomic_private(staging_home / "honcho.json", json.dumps(root, indent=2) + "\n")
        _atomic_private(staging_home / ".env", env_text)
        return {"connection": "needs-auth" if needs_auth else "preserved", "needs_auth": needs_auth,
                "enabled": bool(selected["enabled"])}
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
        raise ValueError("Honcho clone could not prepare safe local settings; repair the source configuration "
                         "and use a private staging directory, then retry. No profile was published.") from None
