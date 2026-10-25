"""config.yaml readers for the Discord adapter: DM role-auth guild opt-in and button-view timeout.

Re-exported from ``plugins.platforms.discord.adapter`` (callers and tests import them there).
"""
from typing import Any, Optional


def _read_dm_role_auth_guild() -> Optional[int]:
    """Return the guild ID opted-in for DM role-based auth, or None (secure default). Read from
    config.yaml ``discord.dm_role_auth_guild`` only (behavioral, not a secret); int or numeric string."""
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config() or {}
        discord_cfg = cfg.get("discord", {}) or {}
        raw = discord_cfg.get("dm_role_auth_guild")
    except Exception:
        return None
    if raw is None or raw == "":
        return None
    try:
        guild_id = int(raw)
    except (TypeError, ValueError):
        return None
    return guild_id if guild_id > 0 else None


# Default timeout for Discord button views when ``approvals.discord_prompt_timeout`` is unset;
# Discord interaction tokens expire at ~15 minutes, so 900s is the practical ceiling.
_DISCORD_PROMPT_TIMEOUT_DEFAULT = 300
_DISCORD_PROMPT_TIMEOUT_MIN = 30
_DISCORD_PROMPT_TIMEOUT_MAX = 900

# Opt-in owner-mention switches for the interactive prompt families (approval card, clarify,
# slash confirm, update prompt); default off. Seeded into ``extra`` plus a DISCORD_* env default.
_PROMPT_MENTION_YAML_KEYS = (
    ("approval_mentions", "DISCORD_APPROVAL_MENTIONS"),
    ("clarify_mentions", "DISCORD_CLARIFY_MENTIONS"),
    ("slash_confirm_mentions", "DISCORD_SLASH_CONFIRM_MENTIONS"),
    ("update_prompt_mentions", "DISCORD_UPDATE_PROMPT_MENTIONS"),
)


def _seed_prompt_mention_keys(
    discord_cfg: dict, platform_extra_cfg: dict, seeded_extra: dict, env_setter,
) -> None:
    """Seed the prompt-mention opt-ins from ``discord:`` YAML (or the platform ``extra`` block)
    into ``seeded_extra`` plus a ``DISCORD_*`` env default via ``env_setter(key, value)``."""
    for key, env_key in _PROMPT_MENTION_YAML_KEYS:
        value = discord_cfg[key] if key in discord_cfg else platform_extra_cfg.get(key)
        if value is not None:
            seeded_extra[key] = value
            env_setter(env_key, str(value).lower())


def _read_discord_prompt_timeout() -> int:
    """Timeout (seconds) for Discord button views from ``approvals.discord_prompt_timeout``
    (default 300), clamped to [MIN, MAX] so a typo can't make prompts vanish or outlive tokens."""
    raw: Any = None
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config() or {}
        approvals_cfg = cfg.get("approvals", {}) or {}
        raw = approvals_cfg.get("discord_prompt_timeout")
    except Exception:
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    if raw is None or raw == "":
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        return _DISCORD_PROMPT_TIMEOUT_DEFAULT
    if seconds < _DISCORD_PROMPT_TIMEOUT_MIN:
        return _DISCORD_PROMPT_TIMEOUT_MIN
    if seconds > _DISCORD_PROMPT_TIMEOUT_MAX:
        return _DISCORD_PROMPT_TIMEOUT_MAX
    return seconds
