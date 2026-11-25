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
restart; nothing here polls. Known gap (documented, pinned by a test): when the bot belongs to a
secondary profile, the LAUNCH profile's flag comes from ``runner.config``, which the reconcile
never reloads, so switching it off keeps that profile reachable until a restart. A profile that
is not served, did not opt in, or claims a handle another profile also claims is unreachable.

Each copy is the event's own source with ``profile`` stamped, exactly what ``build_source`` does
for a ``profile_routes`` match, so the gateway's single identity seam (``resolve_identity``) keys,
authorizes (the receiving bot's allowlist) and delivers it. ``profile_mention`` marks the source
so the reply is labelled with the profile's name (``gateway/run_turn_persistence.py``). The same
in-process marker is what the session authority's binding accepts the re-homed profile from
(``session_ingress_context._mention_profile``): it re-asks this bot, on admission and when the turn
is restored, whether the profile is still one it may hand to (``profile_mention_targets``).

A message with no mentionable handle passes through untouched. Unknown ``$handles`` are logged and
left in the text. A handle with no prompt after it behaves like a bare bot @mention: with
backfilled channel context (``history_backfill``, on by default) it is an empty-text "catch me up"
turn for that profile; with no context and no media it is a no-op.

Only free-text prompts are dispatched. A slash command (``/reset $agent-one``, typed or native) is
never re-homed: it runs exactly as it would without this feature, ``$handle`` and all, so a
``$<profile>`` in its arguments can neither address the bot nor act on another profile's session.
Neither is a message that becomes a command once its handles are stripped (``$agent-one /new``):
the gateway tells a command by its text (``MessageEvent.is_command``), not its ``MessageType``, so
the copy would run ``/new`` in that profile's session. It stays plain text in the channel's own.

Only what the user TYPED is scanned, the same text the ``require_mention`` gate reads. Text the
adapter inlines from an attached document (``[Content of notes.md]: ...``) never routes a message,
and its ``$handles`` are never stripped: the profile gets the document byte for byte, followed by
the typed text without its handles.

A gateway can run several Discord bots (one per profile token), and every bot in the channel
receives the same message and computes the same handles. Exactly one of them dispatches it to a
given profile (``_claim``), and always the same one for that profile's session lane: the gateway
binds a lane to the bot that carried its first turn and pauses it if a later turn comes through
another bot. Until a lane has a recorded bot, the first bot to reach dispatch claims it.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import threading
import weakref
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Optional

from agent.secret_scope import is_multiplex_active
from gateway.platforms.event import MessageEvent, MessageType
from hermes_constants import PROFILE_ID_RE

logger = logging.getLogger(__name__)

#: The opt-in keys under a profile's OWN ``discord:`` block.
MENTIONABLE_KEY = "mentionable"
HANDLE_KEY = "mention_handle"

_HANDLE = re.compile(r"(?<![\w$])\$([A-Za-z0-9][A-Za-z0-9_-]{0,63})(?![\w-])")
_HSPACE = " \t"
_BLANK_LEAD = re.compile(r"\A(?:[ \t]*\n)+")  # blank lines a removed leading handle line leaves

#: Per gateway runner: ``(platform, message id, profile)`` -> a weak ref to the bot that dispatched
#: it, oldest first, capped at ``_CLAIMS_MAX`` (Discord message ids never repeat).
_CLAIMS: weakref.WeakKeyDictionary[Any, OrderedDict] = weakref.WeakKeyDictionary()
_CLAIMS_LOCK = threading.Lock()
_CLAIMS_MAX = 4096


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
    spans = [m.span() for m in _HANDLE.finditer(text) if m.group(1).lower() in handles]
    return MentionPlan(tuple(profiles), _strip_handles(text, spans), tuple(unknown))


def _handle_runs(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """*spans* with the handles that only spaces separate (``$a $b``) merged into one run."""
    runs: list[tuple[int, int]] = []
    for start, end in spans:
        if runs and not text[runs[-1][1]:start].strip(_HSPACE):
            start = runs.pop()[0]
        runs.append((start, end))
    return runs


def _strip_handles(text: str, spans: list[tuple[int, int]]) -> str:
    """*text* without the handle tokens at *spans*; everything else (newlines, indentation, code
    fences, inner spacing) is kept byte for byte. Only what a removed run leaves behind goes with
    it: its flanking spaces, a ``:``/``,``/``;`` right after a leading handle, a line that held
    nothing but handles, and the blank edges that leaves (no global whitespace normalization)."""
    out, pos = "", 0
    for start, end in _handle_runs(text, spans):
        gap = text[pos:start]
        out += gap.rstrip(_HSPACE)
        spaced_left = gap != gap.rstrip(_HSPACE)
        rest = text[end:].lstrip(_HSPACE)
        spaced_right = len(rest) < len(text) - end
        pos = len(text) - len(rest)
        before, after = out[-1:] or "\n", rest[:1] or "\n"
        if before == "\n" and after == "\n":  # the line held only handles: drop the whole line
            if rest:
                pos += 1
            else:
                out = out[:-1]
        elif before == "\n" and after in ",;:":  # "$a1: do X" -> "do X"
            pos = len(text) - len(rest[1:].lstrip(_HSPACE))
        elif before != "\n" and after not in "\n,.;:!?" and spaced_left and spaced_right:
            out += " "  # "ask $a1 now" -> "ask now" (but "hey $a1, hi" -> "hey, hi")
    return _BLANK_LEAD.sub("", out + text[pos:]).rstrip()


def _active_handles(adapter: Any, text: str) -> dict[str, str]:
    """The handles to plan *text* against: none without a ``$``, or without multiplexing (there is
    no per-profile session to hand to)."""
    if "$" not in text or not is_multiplex_active():
        return {}
    return mentionable_handles(adapter)


def _is_command_text(text: str) -> bool:
    """Text the gateway runs as a slash command. The gateway classifies by text, not by
    ``MessageType`` (``MessageEvent.is_command``: a leading ``/``, after any ``@file:`` refs)."""
    return MessageEvent(text=text or "").is_command()


def _prompt_plan(adapter: Any, typed: str) -> Optional[MentionPlan]:
    """The plan to dispatch *typed* by, or None. A slash command is never dispatched, nor is a
    message that becomes one once its handles are stripped (``$agent-one /new``): its copy would
    run ``/new`` in that profile's session."""
    if _is_command_text(typed):
        return None
    plan = plan_mentions(typed, _active_handles(adapter, typed))
    return None if plan is None or _is_command_text(plan.prompt) else plan


def addresses_profile(adapter: Any, text: str) -> bool:
    """True when *text* names a mentionable profile, so the require_mention gate lets it through.
    A slash command never does, ``$handle`` before or after it: it faces the gate exactly as it
    would without the handle."""
    return _prompt_plan(adapter, text or "") is not None


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


def _bot_of(runner: Any, platform: Any, profile: str) -> Any:
    """The Discord bot *profile* owns in this gateway, or None (``_owner_profile``'s inverse)."""
    if profile == (getattr(runner, "_primary_profile_name", None) or "default"):
        return (getattr(runner, "adapters", None) or {}).get(platform)
    return ((getattr(runner, "_profile_adapters", None) or {}).get(profile) or {}).get(platform)


def _lane(runner: Any, copy: Any) -> Optional[str]:
    """The session lane (session key) a dispatched *copy* runs in, when the runner can tell."""
    key_for = getattr(runner, "_session_key_for_source", None)
    return key_for(copy.source) if callable(key_for) else None


def _lane_carrier(adapter: Any, runner: Any, lane: Optional[str]) -> Optional[str]:
    """The profile whose bot the gateway recorded as carrying *lane* (``SessionEntry.transport_profile``),
    while that bot is still part of this gateway."""
    store = getattr(runner, "session_store", None)
    entry = store.lookup_by_session_key(lane) if lane and store is not None else None
    name = getattr(entry, "transport_profile", None)
    return name if name and _bot_of(runner, adapter.platform, name) is not None else None


def _claim(adapter: Any, event: Any, profile: str, lane: Optional[str] = None) -> bool:
    """True when *adapter* is the bot that dispatches *event* to *profile*.

    A profile's session lane answers through ONE bot: the gateway records the bot that carried its
    first turn (``SessionEntry.transport_profile``) and pauses the lane when a later turn arrives
    through another (``check_native_route``). So once *lane* has a recorded bot, only that bot
    dispatches to it. Before that, the first bot to reach dispatch claims the message and the lane
    for the whole gateway, and the others hand that profile nothing. The same bot may plan its own
    message again (its own dedup governs that); a message claim whose bot is gone still holds, so
    a later bot never re-sends the message."""
    runner = getattr(adapter, "gateway_runner", None)
    if runner is None or not event.message_id:
        return True
    carrier = _lane_carrier(adapter, runner, lane)
    if carrier is not None:
        return carrier == _owner_profile(adapter, runner)
    platform = str(getattr(adapter.platform, "value", adapter.platform))
    key, lane_key = (platform, str(event.message_id), profile), (platform, "lane", lane)
    with _CLAIMS_LOCK:
        claims = _CLAIMS.setdefault(runner, OrderedDict())
        owner = claims.get(key)
        if owner is not None:
            return owner() is adapter
        holder = claims[lane_key]() if lane and lane_key in claims else None
        if holder is not None and holder is not adapter:
            return False
        claims[key] = weakref.ref(adapter)
        if lane:
            claims[lane_key] = weakref.ref(adapter)
            claims.move_to_end(lane_key)
        while len(claims) > _CLAIMS_MAX:
            claims.popitem(last=False)
        return True


def _rebuild_prompt(text: str, typed: str, stripped: str) -> str:
    """The event *text* with its *typed* part (the adapter appends it after any inlined document
    text) replaced by *stripped*. Text that is not where the typed part should be is returned
    unchanged: only typed text is ever edited."""
    if not typed or not text.endswith(typed):
        return text
    head = text[:len(text) - len(typed)]
    return head + stripped if stripped else head.removesuffix("\n\n")


def mention_events(adapter: Any, event: Any, typed: str) -> list:
    """The events to dispatch for *event*, judged on *typed* (what the user typed; never inlined
    attachment text): itself when it names no mentionable profile, one copy per named profile
    otherwise (unless another bot already dispatched it), or none when nothing but the handles was
    sent and there is no backfilled context or media (a bare @mention's rule). A slash command is
    always itself, also with a ``$handle`` in front of it: it runs where it would without this
    feature, its text (``$handle`` included) untouched."""
    if event.message_type == MessageType.COMMAND:
        return [event]
    plan = _prompt_plan(adapter, typed or "")
    if plan is None:
        return [event]
    prompt = _rebuild_prompt(event.text or "", typed, plan.prompt)
    if not prompt and not event.media_urls and not event.channel_context:
        logger.info("[%s] Ignoring profile-mention-only message (no prompt) for %d profile(s)",
                    adapter.name, len(plan.profiles))
        return []
    runner = getattr(adapter, "gateway_runner", None)
    copies = [_profile_event(adapter, event, profile, prompt) for profile in plan.profiles]
    copies = [copy for copy in copies if _claim(adapter, event, copy.source.profile, _lane(runner, copy))]
    if len(copies) < len(plan.profiles):
        logger.info("[%s] Message %s: %d mentioned profile(s) dispatched by another bot", adapter.name,
                    event.message_id, len(plan.profiles) - len(copies))
    if copies:
        logger.info("[%s] Dispatching message %s to %d mentioned profile(s)", adapter.name,
                    event.message_id, len(copies))
    return copies


class DiscordProfileMentionsMixin:
    """What the gateway's session binding asks the receiving bot about a ``$<profile>`` copy."""

    def profile_mention_targets(self) -> frozenset[str]:
        """The served, opted-in profiles this bot may hand a message to right now. The session
        binding (``session_ingress_context._mention_profile``) admits a dispatched copy, and later
        its restored turn, only while its profile is in here."""
        return frozenset(mentionable_handles(self).values()) if is_multiplex_active() else frozenset()


async def dispatch_event(adapter: Any, event: Any, typed: str, *, batch: bool) -> None:
    """Hand *event* (or its per-profile copies, chosen from the *typed* text) to the gateway,
    batched or straight through."""
    for item in mention_events(adapter, event, typed):
        if batch:
            adapter._enqueue_text_event(item)
        else:
            await adapter.handle_message(item)
