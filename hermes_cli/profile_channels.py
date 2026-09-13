"""Messaging-channel settings a profile clone must NOT inherit.

A ``--clone``d profile that keeps the source's bot tokens, allowlists and platform state makes two
gateways fight over one bot (standalone) or blocks ``hermes gateway migrate --multiplex`` with a
duplicate-credential finding per platform. The key set is DERIVED from the platform adapters — the
``Platform`` enum + plugin registry (``required_env``, allowlist/allow-all/home-channel env names),
the gateway env-override table (``gateway.config_env._ENV_STEPS`` / ``_ENV_ENABLE_CREDENTIALS``) and
the ``<PLATFORM>_`` env prefix every adapter's keys share — so a new adapter is covered without a
hand-written list. Model/provider keys, tool keys, memory and general config are never touched.
"""

from __future__ import annotations

import contextlib
import logging
import re
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Platforms whose env names do not share the ``<PLATFORM>_`` prefix of their config id. The
# dashboard Channels page uses the same table to decide which Keys-page fields a card owns.
_PLATFORM_ENV_PREFIX_ALIASES: dict[str, tuple[str, ...]] = {
    "email": ("EMAIL_",),
    "homeassistant": ("HASS_",),
    "qqbot": ("QQ_", "QQBOT_"),
    "sms": ("TWILIO_",),
    "wecom": ("WECOM_BOT_", "WECOM_SECRET"),
    "wecom_callback": ("WECOM_CALLBACK_",),
}

# Multiplexer-owner settings: a clone of the default that inherits them and is then started
# standalone tries to be a second multiplexer for every profile on the host.
_GATEWAY_OWNER_KEYS = ("multiplex_profiles", "profile_routes")

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def platform_env_prefixes(platform_id: str) -> tuple[str, ...]:
    """Env-var prefixes owned by one messaging platform."""
    return _PLATFORM_ENV_PREFIX_ALIASES.get(platform_id, (platform_id.upper().replace("-", "_") + "_",))


def platform_ids() -> List[str]:
    """Every messaging platform id: built-in ``Platform`` members plus registered plugin adapters."""
    from gateway.config import Platform
    ids = {m.value for m in Platform.__members__.values() if m.value != "local"}
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        ids.update(entry.name for entry in platform_registry.all_entries())
    return sorted(ids)


def _cred_row_envs(row) -> Set[str]:
    """Every env name a ``gateway.config_env._Cred`` row reads."""
    names: Set[str] = set()

    def _flatten(spec) -> None:
        if isinstance(spec, str):
            names.add(spec)
        elif isinstance(spec, (tuple, list)):
            for item in spec:
                _flatten(item)

    _flatten(row.creds)
    if row.token:
        names.add(row.token)
    for key_env in (*row.fixed, *row.optional, *row.optional_stripped):
        _flatten(key_env[1])
    if row.warn_missing:
        names.add(row.warn_missing[0])
    if row.home:
        names.update({row.home, f"{row.home}_NAME", f"{row.home}_THREAD_ID"})
    return names


def declared_channel_env_keys() -> Dict[str, str]:
    """``{ENV_KEY: platform_id}`` for every env name an adapter declares outright (registry entry
    fields, the gateway env-override table). Prefix matching covers the rest."""
    keys: Dict[str, str] = {}
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.all_entries():
            for name in (*entry.required_env, entry.allowed_users_env, entry.allow_all_env, entry.cron_deliver_env_var):
                if name:
                    keys[name] = entry.name
    with contextlib.suppress(Exception):
        from gateway import config_env
        for platform, names in config_env._ENV_ENABLE_CREDENTIALS.items():
            keys.update(dict.fromkeys(names, platform.value))
        for step in config_env._ENV_STEPS:
            if isinstance(step, config_env._Cred):
                keys.update(dict.fromkeys(_cred_row_envs(step), step.platform.value))
            elif isinstance(step, partial):
                platform = step.keywords.get("platform")
                for kw in ("env", "env_base"):
                    if step.keywords.get(kw) and platform is not None:
                        keys[step.keywords[kw]] = platform.value
    return keys


_CREDENTIAL_SUFFIXES = (
    "_TOKEN", "_SECRET", "_KEY", "_PASSWORD", "_APP_ID", "_CLIENT_ID", "_BOT_ID", "_ACCOUNT_SID",
    "_SERVICE_ACCOUNT_JSON", "_PROJECT_ID",
)


def credential_env_keys() -> Dict[str, str]:
    """``{ENV_KEY: platform_id}`` for the keys that make an adapter CONNECT AS a bot (token / app id /
    client id / secret — the shape ``GatewayRunner._adapter_credential_fingerprint`` hashes). Enable
    flags, URLs and hosts are excluded: two profiles pointing at one Mattermost server collide only
    when they also share the token."""
    keys: Dict[str, str] = {}
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.all_entries():
            keys.update(dict.fromkeys(entry.required_env, entry.name))
    with contextlib.suppress(Exception):
        from gateway import config_env
        for platform, names in config_env._ENV_ENABLE_CREDENTIALS.items():
            keys.update(dict.fromkeys(names, platform.value))
        for step in config_env._ENV_STEPS:
            if isinstance(step, config_env._Cred):
                creds: Set[str] = set()
                for group in step.creds:
                    creds.update((group,) if isinstance(group, str) else group)
                if step.token:
                    creds.add(step.token)
                keys.update(dict.fromkeys(creds, step.platform.value))
    return {key: pid for key, pid in keys.items() if key.endswith(_CREDENTIAL_SUFFIXES)}


class ChannelKeyIndex:
    """Resolves an env key to the messaging platform that owns it (``None`` = not a channel key)."""

    def __init__(self) -> None:
        self.platforms = platform_ids()
        self.declared = declared_channel_env_keys()
        self._prefixes: List[Tuple[str, str]] = sorted(
            ((prefix, pid) for pid in self.platforms for prefix in platform_env_prefixes(pid)),
            key=lambda item: -len(item[0]),  # longest prefix wins: WECOM_CALLBACK_ before WECOM_
        )

    def platform_for(self, key: str) -> Optional[str]:
        if key in self.declared:
            return self.declared[key]
        return next((pid for prefix, pid in self._prefixes if key.startswith(prefix)), None)


def _env_key_of_line(line: str) -> Optional[str]:
    match = _ENV_LINE_RE.match(line)
    return match.group(1) if match else None


def strip_channel_env_file(env_path: Path, index: Optional[ChannelKeyIndex] = None) -> Dict[str, List[str]]:
    """Drop every messaging-channel assignment from ``env_path`` in place; comments, blank lines and
    every other key survive verbatim. Returns ``{platform: [keys removed]}``."""
    if not env_path.is_file():
        return {}
    index = index or ChannelKeyIndex()
    removed: Dict[str, List[str]] = {}
    kept: List[str] = []
    text = env_path.read_text(encoding="utf-8-sig", errors="replace")
    for line in text.splitlines():
        key = _env_key_of_line(line)
        platform = index.platform_for(key) if key else None
        if key is None or platform is None:
            kept.append(line)
        else:
            removed.setdefault(platform, []).append(key)
    if removed:
        env_path.write_text("\n".join(kept) + ("\n" if text.endswith("\n") or kept else ""), encoding="utf-8")
    return removed


def _channel_config_paths(raw: dict, platforms: Iterable[str]) -> List[Tuple[str, ...]]:
    """Dotted paths in a raw config.yaml mapping that hold platform identity: ``platforms``, every
    top-level ``<platform>:`` block, ``gateway.platforms`` / ``gateway.<platform>``, and the
    multiplexer-owner keys (both spellings the gateway loader accepts)."""
    paths: List[Tuple[str, ...]] = []
    gateway: dict = raw["gateway"] if isinstance(raw.get("gateway"), dict) else {}
    if "platforms" in raw:
        paths.append(("platforms",))
    if "platforms" in gateway:
        paths.append(("gateway", "platforms"))
    for key in _GATEWAY_OWNER_KEYS:
        if key in raw:
            paths.append((key,))
        if key in gateway:
            paths.append(("gateway", key))
    for pid in platforms:
        if pid in raw:
            paths.append((pid,))
        if pid in gateway:
            paths.append(("gateway", pid))
    return paths


def strip_channel_config(config_path: Path, index: Optional[ChannelKeyIndex] = None) -> List[str]:
    """Remove platform sections from a raw ``config.yaml`` in place. Returns the dotted paths removed."""
    if not config_path.is_file():
        return []
    from hermes_cli.config import read_user_config_raw
    from utils import atomic_yaml_write
    index = index or ChannelKeyIndex()
    raw = read_user_config_raw(config_path)
    paths = _channel_config_paths(raw, index.platforms)
    if not paths:
        return []
    for path in paths:
        node = raw
        for seg in path[:-1]:
            node = node[seg]
        node.pop(path[-1], None)
    if isinstance(raw.get("gateway"), dict) and not raw["gateway"]:
        raw.pop("gateway")
    atomic_yaml_write(config_path, raw, sort_keys=False)
    return [".".join(path) for path in paths]


def channel_state_entries(root: Path, index: Optional[ChannelKeyIndex] = None) -> List[Path]:
    """Root entries of a profile that hold per-bot runtime identity: pairing approvals and the
    WhatsApp device session (``platforms/`` + legacy dirs), the gateway's per-platform ledgers,
    channel directories and every ``<platform>_*`` state file an adapter writes beside config.yaml."""
    if not root.is_dir():
        return []
    index = index or ChannelKeyIndex()
    fixed = {"platforms", "pairing", "whatsapp", "gateway", "channel_directory.json", "channel_aliases.json"}
    prefixes = tuple(f"{pid}_" for pid in index.platforms)
    return sorted(
        entry for entry in root.iterdir()
        if entry.name in fixed or (entry.is_file() and entry.name.startswith(prefixes))
    )


def strip_channel_settings(profile_dir: Path, *, include_state: bool) -> Dict[str, List[str]]:
    """Strip channel credentials/identity from a freshly cloned profile. ``include_state`` also
    drops the runtime state ``--clone-all`` copied. Returns ``{platform|"config"|"state": [what]}``."""
    import shutil
    index = ChannelKeyIndex()
    stripped: Dict[str, List[str]] = dict(strip_channel_env_file(profile_dir / ".env", index))
    config_paths = strip_channel_config(profile_dir / "config.yaml", index)
    if config_paths:
        stripped["config"] = config_paths
    if include_state:
        dropped = []
        for entry in channel_state_entries(profile_dir, index):
            shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
            dropped.append(entry.name)
        if dropped:
            stripped["state"] = dropped
    return stripped


def channel_platforms_configured(profile_dir: Path) -> List[str]:
    """Platform ids with any channel setting in ``profile_dir`` (.env keys or config.yaml sections) —
    what a channel-less clone of it leaves behind. Pure read."""
    index = ChannelKeyIndex()
    found: Set[str] = set()
    env_path = profile_dir / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            key = _env_key_of_line(line)
            platform = index.platform_for(key) if key else None
            if platform:
                found.add(platform)
    config_path = profile_dir / "config.yaml"
    if config_path.is_file():
        from hermes_cli.config import read_user_config_raw
        raw = read_user_config_raw(config_path)
        for path in _channel_config_paths(raw, index.platforms):
            node = raw
            for seg in path:
                node = node[seg]
            if path[-1] == "platforms" and isinstance(node, dict):
                found.update(str(k) for k in node)
            elif path[-1] in index.platforms:
                found.add(path[-1])
    return sorted(found)


def _env_values(env_path: Path, wanted: Dict[str, str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.is_file():
        return values
    from dotenv import dotenv_values
    with contextlib.suppress(Exception):
        for key, value in (dotenv_values(env_path, encoding="utf-8-sig") or {}).items():
            if key in wanted and value and value.strip():
                values[key] = value.strip()
    return values


def _config_platform_tokens(config_path: Path) -> Dict[str, str]:
    """``{platform: token}`` from ``platforms.<p>.token|api_key`` (both nesting spellings)."""
    tokens: Dict[str, str] = {}
    if not config_path.is_file():
        return tokens
    from hermes_cli.config import read_user_config_raw
    raw = read_user_config_raw(config_path)
    gateway: dict = raw["gateway"] if isinstance(raw.get("gateway"), dict) else {}
    for section in (raw.get("platforms"), gateway.get("platforms")):
        if not isinstance(section, dict):
            continue
        for pid, block in section.items():
            if isinstance(block, dict):
                token = block.get("token") or block.get("api_key")
                if isinstance(token, str) and token.strip():
                    tokens[str(pid)] = token.strip()
    return tokens


def shared_channel_credentials(profile_dir: Path, source_dir: Path) -> List[str]:
    """Platforms whose CONNECTING credential (bot token / app id / account) in ``profile_dir`` is
    byte-identical to ``source_dir``'s — the bots that will collide. Pure file reads: no secret
    manager, no gateway config load, so ``hermes profile list`` can afford it per profile."""
    wanted = credential_env_keys()
    mine = _env_values(profile_dir / ".env", wanted)
    theirs = _env_values(source_dir / ".env", wanted)
    shared = {wanted[key] for key in mine if theirs.get(key) == mine[key]}
    mine_cfg = _config_platform_tokens(profile_dir / "config.yaml")
    theirs_cfg = _config_platform_tokens(source_dir / "config.yaml")
    shared.update(pid for pid, token in mine_cfg.items() if theirs_cfg.get(pid) == token)
    return sorted(shared)


def shared_credential_warning(profile: str, platforms: List[str], source: str = "default") -> str:
    return (
        f"⚠ Profile '{profile}' shares its {', '.join(platforms)} credential with {source}: the bot can "
        f"only belong to one profile. Give '{profile}' its own bot (hermes -p {profile} setup, or the "
        f"dashboard Messaging page) or remove the token from '{profile}'; a multiplexed gateway parks "
        f"the duplicate and `hermes gateway migrate --multiplex` refuses until it is gone."
    )


def format_stripped_notice(profile: str, platforms: List[str], clone_flag: str = "--clone") -> List[str]:
    """Lines printed after a channel-less clone so the user knows what was left behind and how to
    configure the new profile's own bots."""
    if not platforms:
        return []
    return [
        f"Messaging channels were NOT cloned ({', '.join(platforms)}): a copied bot token or allowlist "
        "would make two gateways fight over one bot.",
        f"  Configure this profile's own bots:  hermes -p {profile} setup   (or the dashboard Messaging page)",
        f"  To copy the source's channels anyway:  hermes profile create {profile} {clone_flag} --clone-channels",
    ]
