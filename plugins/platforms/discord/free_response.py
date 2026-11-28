"""Free-response resolution for Discord: which channels and threads need no @mention.

``free_response_channels`` lists channels (and threads) answered without a mention. A thread has
always matched through its parent too, so listing a channel freed every thread under it.
``free_response_thread_mode`` (``inherit`` | ``never`` | ``always``) chooses how threads decide, and
``free_response_overrides`` (``{ID: bool}``) overrides any single channel or thread.
"""
from __future__ import annotations

import enum
import json
import logging
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def keys_for(channel_id: Any, name: Any) -> tuple[str, ...]:
    """Match keys for one channel, most specific first: ID, bare name, ``#name``."""
    keys: list[str] = []
    if channel_id is not None and str(channel_id).strip():
        keys.append(str(channel_id).strip())
    name = str(name or "").strip()
    if name:
        keys.extend((name, f"#{name}"))
    return tuple(keys)


def own_channel_keys(channel: Any) -> tuple[str, ...]:
    """ID / name keys of the channel itself (a thread's own keys, never its parent's)."""
    return keys_for(getattr(channel, "id", None), getattr(channel, "name", ""))


def parent_channel_keys(channel: Any, parent_channel_id: Optional[str] = None) -> tuple[str, ...]:
    """ID / name keys of a thread's parent channel; empty for a non-thread channel."""
    parent = getattr(channel, "parent", None)
    return keys_for(
        parent_channel_id or getattr(channel, "parent_id", None),
        getattr(parent, "name", "") if parent else "",
    )


def parse_threads_mode(raw: Any, name: str = "Discord") -> str:
    """``inherit`` (default) | ``never`` | ``always``; anything else warns and means ``inherit``."""
    text = str(raw).strip().lower() if raw is not None else ""
    if text in {"inherit", "never", "always"}:
        return text
    if text:
        logger.warning(
            "[%s] Ignoring invalid free_response_thread_mode=%r (allowed: inherit, never, always); using inherit",
            name, raw)
    return "inherit"


def _flag(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return True if text in _TRUE else False if text in _FALSE else None


def parse_overrides(raw: Any, name: str = "Discord") -> dict[str, bool]:
    """``{channel-or-thread ID (or name): bool}`` from a YAML map or a JSON object string (what the
    YAML bridge writes to the env var). Anything else, and any entry whose value is not a boolean,
    is skipped with a warning."""
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except ValueError:
            logger.warning("[%s] Ignoring free_response_overrides: not valid JSON (expected an object like"
                           ' {"<id>": true})', name)
            return {}
    if raw is None or raw == "" or raw == {}:
        return {}
    if not isinstance(raw, dict):
        logger.warning("[%s] Ignoring free_response_overrides: expected a map of ID to true/false, got %s",
                       name, type(raw).__name__)
        return {}
    overrides: dict[str, bool] = {}
    for key, value in raw.items():
        key, flag = str(key).strip(), _flag(value)
        if key and flag is not None:
            overrides[key] = flag
        else:
            logger.warning("[%s] Ignoring free_response_overrides entry %r=%r (allowed values: true, false)",
                           name, key, value)
    return overrides


class FreeResponseSettings(NamedTuple):
    """``free_response_thread_mode`` and ``free_response_overrides``, parsed once per config load."""

    thread_mode: str = "inherit"
    overrides: dict = {}


def derive_settings(extra: Optional[dict], env_get: Any, name: str = "Discord") -> FreeResponseSettings:
    """Parse both settings (config first, then env, like ``free_response_channels``). Call at connect
    and on a live config reload, not per message: an invalid value warns once per load."""
    extra = extra if isinstance(extra, dict) else {}
    mode = extra.get("free_response_thread_mode")
    overrides = extra.get("free_response_overrides")
    return FreeResponseSettings(
        parse_threads_mode(mode if mode is not None else env_get("DISCORD_FREE_RESPONSE_THREAD_MODE"), name),
        parse_overrides(overrides if overrides is not None else env_get("DISCORD_FREE_RESPONSE_OVERRIDES"), name),
    )


class FreeResponse(enum.Enum):
    """How a server message is gated by the free-response settings."""

    FREE = "free"  # no @mention needed
    GATED = "gated"  # @mention needed, except in a thread the bot already joined
    # @mention needed even in a joined thread: the operator opted this thread out (its own ``false``
    # override, or ``never`` withholding a free status it would otherwise inherit). Without this a
    # thread the bot ever answered in stays mention-free forever and the opt-out never takes effect.
    OPTED_OUT = "opted_out"


def resolve_free_response(
    channel: Any,
    parent_channel_id: Optional[str],
    *,
    free_channels: set,
    threads_mode: str,
    overrides: dict[str, bool],
    thread_cls: Any,
    self_id: Optional[str] = None,
) -> FreeResponse:
    """Plain channel: its override, else listed in ``free_channels`` (or ``"*"``). Thread: its own
    override, else its own ID/name listed, else by *threads_mode*: ``always`` free, ``inherit`` the
    parent's status (parent override, then parent listed or ``"*"``; the historical behaviour),
    ``never`` opted out where it would have inherited free. A thread the bot created (*self_id* owns
    it: auto-thread, ``/thread``) is only gated under ``never``, so its own conversation continues."""
    def override_for(keys: tuple[str, ...]) -> Optional[bool]:
        return next((overrides[k] for k in keys if k in overrides), None)

    own_keys = own_channel_keys(channel)
    own_override = override_for(own_keys)
    if own_override is not None:
        return FreeResponse.FREE if own_override else FreeResponse.OPTED_OUT
    parent_keys = parent_channel_keys(channel, parent_channel_id)
    if not parent_keys and not isinstance(channel, thread_cls):
        listed = "*" in free_channels or bool(set(own_keys) & free_channels)
        return FreeResponse.FREE if listed else FreeResponse.GATED
    if set(own_keys) & free_channels or threads_mode == "always":
        return FreeResponse.FREE
    parent_override = override_for(parent_keys)
    inherited = parent_override if parent_override is not None else (
        "*" in free_channels or bool(set(parent_keys) & free_channels))
    if not inherited:
        return FreeResponse.GATED
    if threads_mode != "never":
        return FreeResponse.FREE
    bot_owned = self_id is not None and str(getattr(channel, "owner_id", "")) == self_id
    return FreeResponse.GATED if bot_owned else FreeResponse.OPTED_OUT


def seed_overrides_from_yaml(discord_cfg: dict, seeded_extra: dict, env_default: Any) -> None:
    """Carry ``discord.free_response_overrides`` into ``extra`` (as written) and the env bridge (as JSON)."""
    value = discord_cfg.get("free_response_overrides")
    if value is None:
        return
    seeded_extra["free_response_overrides"] = value
    env_default(
        "DISCORD_FREE_RESPONSE_OVERRIDES",
        json.dumps({str(k): v for k, v in value.items()}) if isinstance(value, dict) else str(value),
    )
