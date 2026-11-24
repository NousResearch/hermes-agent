"""``$<profile>`` mentions: address a multiplexed profile by name from a shared Discord channel.

``discord.profile_mentions`` lists the profiles one shared bot may hand a message to. A message
that names one (``$agent-one what's the status?``, anywhere in the text) runs ONE normal turn in
that profile's own session; ``gateway.multiplex_profiles`` must be on. Several handles in one
message fan the same prompt out to each profile as independent turns (no ordering between them,
no shared room). The ``$`` sigil is deliberate: an ``@mention`` would ping real users and roles.

Each copy is the event's own source with ``profile`` stamped, exactly what ``build_source`` does
for a ``profile_routes`` match, so the gateway's single identity seam (``resolve_identity``) keys,
authorizes (the receiving bot's allowlist) and delivers it; a profile the gateway does not serve is
dropped there with a WARNING. ``profile_mention`` marks the source so the reply is labelled with
the profile's name (``gateway/run_turn_persistence.py``).

A message with no configured handle passes through untouched. Unknown ``$handles`` are logged and
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

logger = logging.getLogger(__name__)

_HANDLE = re.compile(r"(?<![\w$])\$([A-Za-z0-9][A-Za-z0-9_-]{0,63})(?![\w-])")
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


@dataclasses.dataclass(frozen=True)
class MentionPlan:
    """Profiles a message addresses (first-mention order, deduplicated) and its prompt text."""

    profiles: tuple[str, ...]
    prompt: str
    unknown: tuple[str, ...] = ()


def _config_items(raw: Any) -> tuple[tuple[str, str], ...]:
    """``(profile, handle)`` pairs from a list/CSV of profiles or a ``{profile: handle}`` map."""
    if isinstance(raw, dict):
        return tuple((str(k), str(v or k)) for k, v in raw.items())
    if isinstance(raw, str):
        raw = raw.split(",")
    if isinstance(raw, (list, tuple)):
        return tuple((str(p), str(p)) for p in raw if p is not None)
    return ()


@lru_cache(maxsize=32)
def _handles(items: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """``{handle: profile}``; invalid entries are skipped with one warning per distinct config."""
    handles: dict[str, str] = {}
    for profile, handle in items:
        profile, handle = profile.strip().lower(), handle.strip().lstrip("$").lower()
        if not _PROFILE_NAME.fullmatch(profile) or not _PROFILE_NAME.fullmatch(handle):
            logger.warning("discord.profile_mentions: skipping invalid entry %r -> %r", profile, handle)
            continue
        handles[handle] = profile
    return handles


def configured_handles(extra: Optional[dict]) -> dict[str, str]:
    """``{handle: profile}`` from ``discord.profile_mentions`` (empty = feature off)."""
    return _handles(_config_items((extra or {}).get("profile_mentions")))


def plan_mentions(text: str, handles: dict[str, str]) -> Optional[MentionPlan]:
    """The dispatch plan for *text*, or None when it names no configured profile."""
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
            logger.info("Discord profile mention: %d unknown $handle(s), none configured; not dispatched", len(unknown))
            logger.debug("Discord profile mention: unknown handles %s", unknown)
        return None
    prompt = _HANDLE.sub(lambda m: "" if m.group(1).lower() in handles else m.group(0), text)
    prompt = _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(prompt.split())).lstrip(",;: ").strip()
    return MentionPlan(tuple(profiles), prompt, tuple(unknown))


def addresses_profile(adapter: Any, text: str) -> bool:
    """True when *text* names a configured profile, so the require_mention gate lets it through."""
    return _active(adapter) and plan_mentions(text, configured_handles(adapter.config.extra)) is not None


def _active(adapter: Any) -> bool:
    if not configured_handles(getattr(adapter.config, "extra", None)):
        return False
    if is_multiplex_active():
        return True
    logger.debug("[%s] discord.profile_mentions is set but gateway.multiplex_profiles is off", adapter.name)
    return False


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
    """The events to dispatch for *event*: itself when it names no configured profile, one copy per
    named profile otherwise, or none when nothing but the handles was sent."""
    plan = plan_mentions(event.text or "", configured_handles(adapter.config.extra)) if _active(adapter) else None
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
