"""``$<profile>`` mentions: address a multiplexed profile by name from a shared Discord channel.

Each profile opts ITSELF in, in its own ``config.yaml``::

    discord:
      mentionable: true      # a YAML boolean; anything else leaves the profile unreachable
      mention_handle: a1     # optional; defaults to the profile name

A message that names an opted-in profile (``$agent-one what's the status?``, anywhere in the text)
runs ONE normal turn in that profile's own session; ``gateway.multiplex_profiles`` must be on.
Several handles in one message fan the same prompt out to each profile as independent turns (no
ordering between them, no shared room). The ``$`` sigil is deliberate: an ``@mention`` would ping
real users and roles.

The handles a bot answers to come from the profiles this gateway SERVES, each read from its own
config: the bot's own profile from its live-refreshed ``config.extra`` (the allowlist refresh in
``adapter_live_gates.py``), every other served profile from the config the hot-serve reconcile
(``gateway/run_profile_reconcile.py``) last loaded for it. That reconcile reloads a served profile
whose ``config.yaml`` changed and drops a deleted one, so flipping the flag takes effect without a
restart; nothing here polls. A profile that is not served, did not opt in, or claims a handle
another profile also claims is unreachable.

Each copy is the event's own source with ``profile`` stamped, exactly what ``build_source`` does
for a ``profile_routes`` match, so the gateway's single identity seam (``resolve_identity``) keys,
authorizes (the receiving bot's allowlist) and delivers it. ``profile_mention`` marks the source
so the reply is labelled with the profile's name (``gateway/run_turn_persistence.py``).

A message with no mentionable handle passes through untouched. Unknown ``$handles`` are logged and
left in the text; a handle with no prompt after it is a no-op, like a bare bot @mention.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import weakref
from functools import lru_cache
from typing import Any, Optional

from agent.secret_scope import is_multiplex_active
from hermes_constants import PROFILE_ID_RE

logger = logging.getLogger(__name__)

#: The opt-in keys under a profile's OWN ``discord:`` block.
MENTIONABLE_KEY = "mentionable"
HANDLE_KEY = "mention_handle"

_HANDLE = re.compile(r"(?<![\w$])\$([A-Za-z0-9][A-Za-z0-9_-]{0,63})(?![\w-])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


@dataclasses.dataclass(frozen=True)
class MentionPlan:
    """Profiles a message addresses (first-mention order, deduplicated) and its prompt text."""

    profiles: tuple[str, ...]
    prompt: str
    unknown: tuple[str, ...] = ()


def _owner_profile(adapter: Any, runner: Any) -> Optional[str]:
    """The served profile whose bot *adapter* is: the launch profile's, or a secondary's."""
    platform = adapter.platform
    if (getattr(runner, "adapters", None) or {}).get(platform) is adapter:
        return getattr(runner, "_primary_profile_name", None) or "default"
    for name, adapters in (getattr(runner, "_profile_adapters", None) or {}).items():
        if (adapters or {}).get(platform) is adapter:
            return name
    return None


def _profile_extra(adapter: Any, runner: Any, name: str, owner: Optional[str]) -> dict:
    """*name*'s own ``discord:`` settings, as this gateway last loaded them."""
    if name == owner:
        refresh = getattr(adapter, "_refresh_live_gates", None)
        if callable(refresh):
            refresh()  # a stat unless the bot's own config.yaml changed
        return getattr(adapter.config, "extra", None) or {}
    config = (getattr(runner, "_profile_configs", None) or {}).get(name)
    if config is None and name == (getattr(runner, "_primary_profile_name", None) or "default"):
        config = getattr(runner, "config", None)  # the launch profile: never rebuilt by the reconcile
    platform_config = (getattr(config, "platforms", None) or {}).get(adapter.platform)
    return getattr(platform_config, "extra", None) or {}


def _opt_ins(adapter: Any) -> tuple[tuple[str, str], ...]:
    """``(profile, handle)`` for every served profile whose own config sets ``mentionable: true``."""
    runner = getattr(adapter, "gateway_runner", None)
    served = getattr(runner, "served_profile_names", None)
    if not callable(served):
        return ()
    owner = _owner_profile(adapter, runner)
    items = []
    for name in served():
        extra = _profile_extra(adapter, runner, name, owner)
        if extra.get(MENTIONABLE_KEY) is True:
            items.append((str(name), str(extra.get(HANDLE_KEY) or name)))
    return tuple(items)


@lru_cache(maxsize=32)
def _handles(items: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """``{handle: profile}``. An invalid handle, or one several profiles claim, reaches nobody;
    each is warned about once per distinct set of opt-ins."""
    claims: dict[str, list[str]] = {}
    for profile, handle in items:
        handle = handle.strip().lstrip("$").lower()
        if not PROFILE_ID_RE.fullmatch(handle):
            logger.warning("Discord profile mention: profile %r has an invalid discord.%s; not mentionable",
                           profile, HANDLE_KEY)
            continue
        claims.setdefault(handle, []).append(profile)
    handles: dict[str, str] = {}
    for handle, profiles in claims.items():
        if len(profiles) > 1:
            logger.warning("Discord profile mention: profiles %s all claim $%s; none of them gets it",
                           sorted(profiles), handle)
            continue
        handles[handle] = profiles[0]
    return handles


def mentionable_handles(adapter: Any) -> dict[str, str]:
    """``{handle: profile}`` this bot dispatches (empty = no served profile opted in)."""
    return _handles(_opt_ins(adapter))


def plan_mentions(text: str, handles: dict[str, str]) -> Optional[MentionPlan]:
    """The dispatch plan for *text*, or None when it names no mentionable profile."""
    if not handles or not text or "$" not in text:
        return None
    profiles: list[str] = []
    unknown: list[str] = []
    for match in _HANDLE.finditer(text):
        profile = handles.get(match.group(1).lower())
        bucket, name = (profiles, profile) if profile else (unknown, match.group(1).lower())
        if name not in bucket:
            bucket.append(name)
    if not profiles:
        if unknown:
            logger.info("Discord profile mention: %d unknown $handle(s), none mentionable; not dispatched", len(unknown))
            logger.debug("Discord profile mention: unknown handles %s", unknown)
        return None
    prompt = _HANDLE.sub(lambda m: "" if m.group(1).lower() in handles else m.group(0), text)
    prompt = _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(prompt.split())).lstrip(",;: ").strip()
    return MentionPlan(tuple(profiles), prompt, tuple(unknown))


def _active_handles(adapter: Any, text: str) -> dict[str, str]:
    """The handles to plan *text* against: none without a ``$``, or without multiplexing (there is
    no per-profile session to hand to)."""
    if "$" not in text or not is_multiplex_active():
        return {}
    return mentionable_handles(adapter)


def addresses_profile(adapter: Any, text: str) -> bool:
    """True when *text* names a mentionable profile, so the require_mention gate lets it through."""
    return plan_mentions(text, _active_handles(adapter, text or "")) is not None


def _profile_event(adapter: Any, event: Any, profile: str, prompt: str) -> Any:
    """*event* re-homed to *profile*: a fresh, unresolved source (as ``build_source`` makes it) so
    the gateway's identity seam routes it, and per-copy media lists so batching one copy never
    edits another."""
    source = dataclasses.replace(event.source, profile=profile, profile_mention=True)
    source._transport_adapter_ref = weakref.ref(adapter)
    return dataclasses.replace(
        event, text=prompt, source=source, media_urls=list(event.media_urls),
        media_types=list(event.media_types), media_text_inlined=list(event.media_text_inlined),
        metadata=dict(event.metadata or {}))


def mention_events(adapter: Any, event: Any) -> list:
    """The events to dispatch for *event*: itself when it names no mentionable profile, one copy
    per named profile otherwise, or none when nothing but the handles was sent."""
    text = event.text or ""
    plan = plan_mentions(text, _active_handles(adapter, text))
    if plan is None:
        return [event]
    if not plan.prompt and not event.media_urls and not event.channel_context:
        logger.info("[%s] Ignoring profile-mention-only message (no prompt) for %d profile(s)",
                    adapter.name, len(plan.profiles))
        return []
    logger.info("[%s] Dispatching message %s to %d mentioned profile(s)", adapter.name,
                event.message_id, len(plan.profiles))
    return [_profile_event(adapter, event, profile, plan.prompt) for profile in plan.profiles]


async def dispatch_event(adapter: Any, event: Any, *, batch: bool) -> None:
    """Hand *event* (or its per-profile copies) to the gateway, batched or straight through."""
    for item in mention_events(adapter, event):
        if batch:
            adapter._enqueue_text_event(item)
        else:
            await adapter.handle_message(item)
