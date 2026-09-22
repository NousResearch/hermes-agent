"""Messaging-channel settings a profile clone must NOT inherit.

A ``--clone``d profile that keeps the source's bot tokens, allowlists and platform state makes two
gateways fight over one bot (standalone) or blocks ``hermes gateway migrate --multiplex`` with a
duplicate-credential finding per platform.

The inventory is OWNERSHIP-based and evaluated in the SOURCE profile's plugin scope: every adapter
(built-in ``Platform`` member or plugin registered under the source's ``HERMES_HOME``) owns the env
keys it declares outright (``required_env``, allowlist / allow-all / home-channel names, the gateway
env-override table ``gateway.config_env._ENV_STEPS`` / ``_ENV_ENABLE_CREDENTIALS``) plus every key
under its canonical ``<PLATFORM>_`` prefix and its historical alias prefixes. Gateway-wide channel
policy (``GATEWAY_ALLOW_ALL_USERS`` / ``GATEWAY_ALLOWED_USERS``) and relay enrollment identity
(``GATEWAY_RELAY_*``) are channel settings too. A prefix an adapter SHARES with a non-channel
capability (``HASS_*`` is also the Home Assistant tool, ``TWILIO_*`` the telephony skill, ``EMAIL_*``
mail-sending scripts) is stripped only when the source actually runs that adapter — the credential is
then the bot's identity; otherwise it is a tool key and survives. Model/provider keys, tool keys,
memory and general config are never touched.
"""

from __future__ import annotations

import contextlib
import logging
import re
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Historical env prefixes that do not match a platform's config id. They SUPPLEMENT the canonical
# ``<PLATFORM>_`` prefix (``WECOM_DM_POLICY`` and ``SMS_WEBHOOK_PORT`` are wecom / sms keys too).
_PLATFORM_ENV_PREFIX_ALIASES: dict[str, tuple[str, ...]] = {
    "email": ("EMAIL_",),
    "homeassistant": ("HASS_",),
    "qqbot": ("QQ_",),
    "relay": ("GATEWAY_RELAY_",),
    "sms": ("TWILIO_",),
}

# Pseudo-platform for gateway-wide channel policy: not an adapter, but a clone that inherits the
# source's allow-all / allowlist boundary hands a freshly configured bot the source's authorization.
GATEWAY_POLICY_ID = "gateway"
_GATEWAY_POLICY_KEYS = ("GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS")

# Prefixes a messaging adapter shares with a NON-channel capability. Their credentials belong to the
# channel only while the source runs that adapter; the adapter's policy keys are channel-only always.
_SHARED_WITH_TOOLS: dict[str, tuple[str, ...]] = {
    "homeassistant": ("HASS_",),   # tools/homeassistant_tool.py reads HASS_TOKEN / HASS_URL
    "sms": ("TWILIO_",),           # telephony skill reads TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN
    "email": ("EMAIL_",),          # mail-sending skills read EMAIL_ADDRESS / EMAIL_PASSWORD / EMAIL_SMTP_*
}
_POLICY_MARKERS = ("_ALLOWED_USERS", "_ALLOW_ALL_USERS", "_ALLOWED_CHATS", "_HOME_CHANNEL", "_HOME_ADDRESS")

# Multiplexer-owner settings: a clone of the default that inherits them and is then started
# standalone tries to be a second multiplexer for every profile on the host.
_GATEWAY_OWNER_KEYS = ("multiplex_profiles", "profile_routes")

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


@contextlib.contextmanager
def _plugin_scope(source_dir: Optional[Path]):
    """Evaluate adapter discovery in ``source_dir``'s plugin scope (its private ``plugins/``), not the
    ambient profile's. ``None`` keeps the ambient scope."""
    if source_dir is None:
        yield
        return
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(source_dir))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def platform_env_prefixes(platform_id: str) -> tuple[str, ...]:
    """Env-var prefixes owned by one messaging platform: canonical ``<PLATFORM>_`` plus aliases."""
    canonical = platform_id.upper().replace("-", "_") + "_"
    return (canonical, *_PLATFORM_ENV_PREFIX_ALIASES.get(platform_id, ()))


def _registry_entries() -> list:
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent per profile scope
        from gateway.platform_registry import platform_registry
        return list(platform_registry.all_entries())
    return []


def platform_ids(source_dir: Optional[Path] = None) -> List[str]:
    """Every messaging platform id: built-in ``Platform`` members plus the plugin adapters registered
    in ``source_dir``'s scope (ambient scope when ``None``)."""
    from gateway.config import Platform
    ids = {m.value for m in Platform.__members__.values() if m.value != "local"}
    with _plugin_scope(source_dir):
        ids.update(entry.name for entry in _registry_entries())
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


def declared_channel_env_keys(source_dir: Optional[Path] = None) -> Dict[str, str]:
    """``{ENV_KEY: platform_id}`` for every env name an adapter declares outright (registry entry
    fields, the gateway env-override table) plus gateway-wide channel policy. Prefix matching covers
    the rest."""
    keys: Dict[str, str] = dict.fromkeys(_GATEWAY_POLICY_KEYS, GATEWAY_POLICY_ID)
    with _plugin_scope(source_dir):
        for entry in _registry_entries():
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


def _policy_env_keys(source_dir: Optional[Path] = None) -> Set[str]:
    """Allowlist / allow-all / home-channel names adapters declare — channel-only by nature."""
    names: Set[str] = set(_GATEWAY_POLICY_KEYS)
    with _plugin_scope(source_dir):
        for entry in _registry_entries():
            names.update(n for n in (entry.allowed_users_env, entry.allow_all_env, entry.cron_deliver_env_var) if n)
    with contextlib.suppress(Exception):
        from gateway import config_env
        for step in config_env._ENV_STEPS:
            if isinstance(step, config_env._Cred) and step.home:
                names.update({step.home, f"{step.home}_NAME", f"{step.home}_THREAD_ID"})
            elif isinstance(step, partial) and step.keywords.get("env_base"):
                base = step.keywords["env_base"]
                names.update({base, f"{base}_NAME", f"{base}_THREAD_ID"})
    return names


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
    for entry in _registry_entries():
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


def _env_values(env_path: Path, wanted: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Non-blank assignments in ``env_path`` (restricted to ``wanted`` keys when given)."""
    values: Dict[str, str] = {}
    if not env_path.is_file():
        return values
    from dotenv import dotenv_values
    with contextlib.suppress(Exception):
        for key, value in (dotenv_values(env_path, encoding="utf-8-sig") or {}).items():
            if (wanted is None or key in wanted) and value and value.strip():
                values[key] = value.strip()
    return values


def _explicit_enabled(raw: dict, pid: str) -> Optional[bool]:
    """``platforms.<pid>.enabled`` (either nesting) when the source config states it, else None."""
    gateway = raw["gateway"] if isinstance(raw.get("gateway"), dict) else {}
    for section in (raw.get("platforms"), gateway.get("platforms")):
        block = section.get(pid) if isinstance(section, dict) else None
        if isinstance(block, dict) and "enabled" in block:
            return bool(block["enabled"])
    return None


def _shared_adapters_active(source_dir: Optional[Path]) -> Set[str]:
    """Shared-prefix platforms the SOURCE runs as a channel: explicitly enabled in its config.yaml, or
    auto-enabled by a complete credential set in its ``.env`` and not explicitly disabled — the same
    gate ``gateway.config_env._Cred`` applies at gateway start."""
    if source_dir is None:
        return set(_SHARED_WITH_TOOLS)  # no source to consult: the historical (strip) behaviour
    raw: dict = {}
    if (source_dir / "config.yaml").is_file():
        from hermes_cli.config import read_user_config_raw
        with contextlib.suppress(Exception):
            raw = read_user_config_raw(source_dir / "config.yaml") or {}
    env = _env_values(source_dir / ".env")
    creds_by_platform: Dict[str, Set[str]] = {}
    with contextlib.suppress(Exception):
        from gateway import config_env
        for platform, names in config_env._ENV_ENABLE_CREDENTIALS.items():
            creds_by_platform[platform.value] = set(names)
    with _plugin_scope(source_dir):
        for entry in _registry_entries():
            creds_by_platform.setdefault(entry.name, set(entry.required_env))
    active: Set[str] = set()
    for pid in _SHARED_WITH_TOOLS:
        explicit = _explicit_enabled(raw, pid)
        if explicit is not None:
            if explicit:
                active.add(pid)
            continue
        creds = creds_by_platform.get(pid) or set()
        if creds and all(env.get(name) for name in creds):
            active.add(pid)
    return active


class ChannelKeyIndex:
    """Resolves an env key to the messaging platform that owns it (``None`` = not a channel key),
    evaluated in ``source_dir``'s plugin scope with the ownership rule for shared prefixes."""

    def __init__(self, source_dir: Optional[Path] = None) -> None:
        self.source_dir = source_dir
        self.platforms = platform_ids(source_dir)
        self.declared = declared_channel_env_keys(source_dir)
        self.policy = _policy_env_keys(source_dir)
        self.shared_active = _shared_adapters_active(source_dir)
        self._prefixes: List[Tuple[str, str]] = sorted(
            ((prefix, pid) for pid in self.platforms for prefix in platform_env_prefixes(pid)),
            key=lambda item: -len(item[0]),  # longest prefix wins: WECOM_CALLBACK_ before WECOM_
        )

    def platform_for(self, key: str) -> Optional[str]:
        pid = self.declared.get(key) or next((pid for prefix, pid in self._prefixes if key.startswith(prefix)), None)
        if pid is None:
            return None
        shared = _SHARED_WITH_TOOLS.get(pid)
        if shared and key.startswith(shared) and pid not in self.shared_active and not self._is_policy(key):
            return None  # a tool credential the source never used as a bot: keep it
        return pid

    def _is_policy(self, key: str) -> bool:
        return key in self.policy or any(marker in key for marker in _POLICY_MARKERS)


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
    channel directories and every ``<platform>_*`` file OR directory an adapter writes beside
    config.yaml (Google Chat keeps user tokens in ``google_chat_user_tokens/``)."""
    if not root.is_dir():
        return []
    index = index or ChannelKeyIndex()
    fixed = {"platforms", "pairing", "whatsapp", "gateway", "channel_directory.json", "channel_aliases.json"}
    prefixes = tuple(f"{pid}_" for pid in index.platforms)
    return sorted(entry for entry in root.iterdir() if entry.name in fixed or entry.name.startswith(prefixes))


def _remove_entry(entry: Path) -> None:
    """Remove a state entry without following symlinks (``rmtree`` refuses a symlinked dir and would
    otherwise leave the copied link in place; unlinking the link never touches the source)."""
    import shutil
    if entry.is_symlink() or not entry.is_dir():
        entry.unlink(missing_ok=True)
    else:
        shutil.rmtree(entry, ignore_errors=True)


def strip_channel_settings(profile_dir: Path, *, include_state: bool, source_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """Strip channel credentials/identity from a freshly cloned profile, judged in ``source_dir``'s
    plugin scope. ``include_state`` also drops the runtime state ``--clone-all`` copied. Returns
    ``{platform|"config"|"state": [what]}``."""
    index = ChannelKeyIndex(source_dir)
    stripped: Dict[str, List[str]] = dict(strip_channel_env_file(profile_dir / ".env", index))
    config_paths = strip_channel_config(profile_dir / "config.yaml", index)
    if config_paths:
        stripped["config"] = config_paths
    if include_state:
        dropped = []
        for entry in channel_state_entries(profile_dir, index):
            _remove_entry(entry)
            dropped.append(entry.name)
        if dropped:
            stripped["state"] = dropped
    return stripped


def channel_platforms_configured(profile_dir: Path) -> List[str]:
    """Platform ids with any channel setting in ``profile_dir`` (.env keys or config.yaml sections) —
    what a channel-less clone of it leaves behind. Pure read, in ``profile_dir``'s plugin scope."""
    index = ChannelKeyIndex(profile_dir)
    found: Set[str] = set()
    env_path = profile_dir / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            key = _env_key_of_line(line)
            platform = index.platform_for(key) if key else None
            if platform and platform != GATEWAY_POLICY_ID:
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


def clone_channels_refusal(source_dir: Path, source_label: str) -> Optional[str]:
    """Why ``--clone-channels`` must be refused for ``source_dir``: a live multiplexer already serves
    the source, so the copied bot would be parked as a duplicate credential at once (the same finding
    the migrate preflight reports). ``None`` when the copy is allowed. Shared by CLI, REST and TUI
    through :func:`hermes_cli.profiles.create_profile`."""
    from hermes_cli.gateway_multiplex_served import recorded_served_profiles
    from hermes_cli.profiles import normalize_profile_name
    served = recorded_served_profiles()
    if not served or len(served) < 2 or normalize_profile_name(source_label) not in {
        normalize_profile_name(p) for p in served
    }:
        return None
    platforms = channel_platforms_configured(source_dir)
    if not platforms:
        return None
    return (
        f"--clone-channels would copy {', '.join(platforms)} from '{source_label}', which the running "
        "multiplexed gateway already serves: the bot can only belong to one profile, so the copy would be "
        "parked as a duplicate credential. Clone without --clone-channels and give the new profile its own bot "
        "(hermes -p <name> setup), or route its chats with gateway.profile_routes instead."
    )


def _config_platform_tokens(config_path: Path, env_dir: Optional[Path] = None) -> Dict[str, str]:
    """``{platform: token}`` from ``platforms.<p>.token|api_key`` (both nesting spellings).

    Values that are ``${VAR}`` placeholders are expanded against ``env_dir``'s ``.env``
    (each profile's own), so two profiles that reference the same env-var name but resolve
    it to different tokens are correctly distinguished. Pass ``env_dir`` from the profile
    dir that owns ``config_path``.
    """
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
                    token = token.strip()
                    # Expand ${VAR} placeholders against the owning profile's own .env
                    # so two profiles sharing a placeholder name but with different tokens
                    # are correctly distinguished (fixes #119099).
                    if token.startswith("${") and token.endswith("}") and env_dir is not None:
                        env = _env_values(env_dir / ".env", credential_env_keys())
                        var_name = token[2:-1]
                        if var_name in env:
                            token = env[var_name]
                    tokens[str(pid)] = token
    return tokens


def shared_channel_credentials(profile_dir: Path, source_dir: Path) -> List[str]:
    """Platforms whose CONNECTING credential (bot token / app id / account) in ``profile_dir`` is
    byte-identical to ``source_dir``'s — the bots that will collide. Pure file reads: no secret
    manager, no gateway config load, so ``hermes profile list`` can afford it per profile."""
    wanted = credential_env_keys()
    mine = _env_values(profile_dir / ".env", wanted)
    theirs = _env_values(source_dir / ".env", wanted)
    shared = {wanted[key] for key in mine if theirs.get(key) == mine[key]}
    mine_cfg = _config_platform_tokens(profile_dir / "config.yaml", env_dir=profile_dir)
    theirs_cfg = _config_platform_tokens(source_dir / "config.yaml", env_dir=source_dir)
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
