"""Stale-target delivery fallback for cron jobs.

A cron job created from a live conversation stores that conversation as its
``origin`` and, by default, delivers its output back there. But the original
thread/topic/channel can disappear between scheduling and firing — a Discord
thread is deleted, a Telegram forum topic is closed, a Slack channel is
archived, the bot is removed. When that happens the delivery fails and the
notification is silently lost: the user simply never hears back from a reminder
they explicitly asked for.

This module decides, platform-neutrally, whether a delivery failure is
*definitive* (the target genuinely cannot receive the message) and, if so,
produces an ordered list of saner targets to retry instead:

    original thread/topic  ->  parent channel  ->  home channel

Crucially, only definitive failures are redirected. Uncertain failures —
timeouts, rate limits, 5xx, anything unrecognized — are NOT, because the
original send may already have landed and a redirect would deliver the message
twice. Detection reuses the canonical
:func:`gateway.platforms.base.classify_send_error`, so every platform whose
adapter surfaces a recognizable reason is covered, and adding a new signature
there benefits this fallback automatically.
"""

from typing import List, Optional

from gateway.platforms.base import classify_send_error

# ``classify_send_error`` kinds that mean the target definitely cannot receive
# the message, so redirecting elsewhere is safe (it will not duplicate a send
# that might otherwise have succeeded). Everything else — ``rate_limited``,
# ``transient``, ``too_long``, ``bad_format``, ``unknown`` — is treated as
# uncertain and left for the caller's normal error handling.
_DEFINITIVE_ERROR_KINDS = frozenset({"not_found", "forbidden"})


def is_definitive_delivery_failure(error: object) -> bool:
    """Return True when *error* means the target cannot receive the message.

    *error* may be an exception or any value carrying an error string (e.g. the
    ``error`` field of a send result). It is classified via
    :func:`gateway.platforms.base.classify_send_error`; only ``not_found`` and
    ``forbidden`` are considered definitive. Transient/uncertain failures
    (timeouts, rate limits, 5xx, unrecognized) return False so the caller does
    not risk a duplicate delivery.
    """
    if error is None:
        return False
    if isinstance(error, BaseException):
        kind = classify_send_error(error)
    else:
        text = str(error).strip()
        if not text:
            return False
        kind = classify_send_error(None, error_text=text)
    return kind in _DEFINITIVE_ERROR_KINDS


def _parent_fallback_target(
    platform: str, cur_chat: str, cur_thread: str, parent_chat_id: Optional[str]
) -> Optional[dict]:
    """Resolve the parent-channel retry target for a stale thread, if any.

    Two thread models exist across platforms:

    * **Separate parent id** (Discord, Matrix): the target ``chat_id`` *is* the
      thread, and the parent channel is a distinct id carried in
      ``parent_chat_id``. Redirect to that parent channel.
    * **Thread inside a channel** (Telegram forum topics, Slack threads): the
      target ``chat_id`` is already the parent channel and ``thread_id`` names
      the topic/thread within it. Redirect to the channel by dropping the
      thread id.

    Returns ``None`` when neither applies (e.g. a flat DM, or the parent is the
    same place that just failed).
    """
    if parent_chat_id and str(parent_chat_id) != cur_chat:
        return {
            "platform": platform,
            "chat_id": str(parent_chat_id),
            "thread_id": None,
            "fallback_kind": "parent",
        }
    if cur_thread and cur_thread != cur_chat:
        return {
            "platform": platform,
            "chat_id": cur_chat,
            "thread_id": None,
            "fallback_kind": "parent",
        }
    return None


def build_fallback_targets(
    target: dict,
    *,
    parent_chat_id: Optional[str] = None,
    home_chat_id: str = "",
    home_thread_id: Optional[str] = None,
    is_direct_message: bool = False,
) -> List[dict]:
    """Ordered redirect targets for a definitively-undeliverable *target*.

    Order is parent channel (when resolvable) then home channel. Each entry is
    a delivery-target dict ``{platform, chat_id, thread_id, fallback_kind}``
    where ``fallback_kind`` is ``"parent"`` or ``"home"``. Candidates equal to
    the failed target, or to an earlier candidate, are skipped so a message is
    never resent to the same place. Same platform throughout — a stale target is
    retried on saner channels of the *same* platform, never cross-posted.

    ``parent_chat_id`` should be supplied only when *target* is the job's own
    origin thread (so the stored parent belongs to it); pass ``None`` for
    fan-out/broadcast targets. ``home_chat_id`` / ``home_thread_id`` come from
    the platform's configured cron home channel.

    ``is_direct_message`` suppresses the *home-channel* fallback when the failed
    target is a private 1:1 chat. A DM that becomes undeliverable (the user
    blocked/deleted the bot) has no broader-but-still-private place to escalate
    to: the configured home channel is typically a shared group, so redirecting
    private DM content there would leak it. The parent fallback is unaffected —
    dropping a deleted DM *topic* back to the DM root stays in the same private
    conversation.
    """
    platform = str(target.get("platform", ""))
    cur_chat = str(target.get("chat_id", ""))
    cur_thread_raw = target.get("thread_id")
    cur_thread = str(cur_thread_raw) if cur_thread_raw is not None else ""

    def _key(t: dict) -> tuple:
        tid = t.get("thread_id")
        return (str(t.get("platform", "")).lower(), str(t.get("chat_id", "")),
                str(tid) if tid is not None else "")

    seen = {(platform.lower(), cur_chat, cur_thread)}
    out: List[dict] = []

    candidates = [_parent_fallback_target(platform, cur_chat, cur_thread, parent_chat_id)]
    if home_chat_id and not is_direct_message:
        candidates.append({
            "platform": platform,
            "chat_id": str(home_chat_id),
            "thread_id": home_thread_id,
            "fallback_kind": "home",
        })

    for cand in candidates:
        if not cand:
            continue
        key = _key(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def format_fallback_notice(content: str, fallback_kind: Optional[str]) -> str:
    """Prepend a plain-text notice explaining why delivery was redirected.

    Plain English with a leading ``⚠️`` glyph and no markdown, matching the
    cron/gateway notice convention so it renders uniformly across every
    platform. Returns *content* unchanged for an unknown ``fallback_kind``.
    """
    if fallback_kind == "parent":
        prefix = (
            "⚠️ The original thread was no longer available, so this update was "
            "delivered to the parent channel instead."
        )
    elif fallback_kind == "home":
        prefix = (
            "⚠️ The original conversation was no longer available, so this update "
            "was delivered to the home channel instead."
        )
    else:
        return content
    if not content:
        return prefix
    return f"{prefix}\n\n{content}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Platforms whose ``chat_type="thread"`` origin is ALWAYS a shared thread (never a 1:1 DM), so a
# stale thread may escalate to the home channel without leaking private content. Discord threads
# always live under a guild channel; a Slack thread can sit inside a DM, so Slack is deliberately
# excluded (fail closed). Matrix emits no "thread" origin; Telegram uses "forum".
_THREAD_ALWAYS_SHARED_PLATFORMS = frozenset({"discord"})


def home_escalation_allowed(origin: dict, platform_name: str) -> bool:
    """Whether origin content may be republished into the SHARED home channel.

    Fail closed: only a positively-shared origin escalates. A missing or unknown ``chat_type``
    — every job created before it was captured — plus ``webhook`` and the explicit DM kinds are
    treated as possibly-private and stay put. Deliberately NOT ``_TargetDelivery.is_dm_target``,
    which is fail-OPEN (unknown chat_type reads as non-DM): that is the right default for seeding
    a session key and the wrong one for deciding what may be made visible to a group.
    """
    chat_type = str((origin or {}).get("chat_type", "")).strip().lower()
    return chat_type in ("group", "channel", "forum") or (
        chat_type == "thread" and platform_name.lower() in _THREAD_ALWAYS_SHARED_PLATFORMS)


def _target_key(platform: str, chat_id: object, thread_id: object) -> tuple:
    return (str(platform).lower(), str(chat_id), str(thread_id) if thread_id is not None else "")


def attempt_delivery_fallback(t, content: str, media_files, error: object, sent_keys: set) -> tuple:
    """Redirect a definitively-undeliverable cron target to parent/home.

    Returns ``(handled, error_message)``:

    * ``(False, None)`` — no fallback applies (failure not definitive, or no saner target).
      The caller falls through to its normal error handling.
    * ``(True, None)`` — a fallback delivered the message; the caller records no error.
    * ``(True, <str>)`` — fallbacks were attempted and all failed; the caller records ``<str>``.

    Only :func:`is_definitive_delivery_failure` failures redirect, so an uncertain failure
    (timeout, rate limit, 5xx) is never duplicated. A fallback that itself fails definitively
    advances to the next hop; one that fails for an uncertain reason stops the chain rather than
    retrying that send somewhere else too.

    ``sent_keys`` is shared across every target of one delivery run, so several failing targets
    that redirect to the same parent/home channel deliver the identical content there once.
    """
    from dataclasses import replace

    from cron.scheduler_delivery import (
        _get_home_target_chat_id, _get_home_target_thread_id, _standalone_send, logger)

    if not is_definitive_delivery_failure(error):
        return (False, None)

    job, platform_name, chat_id, thread_id = t.job, t.platform_name, t.chat_id, t.thread_id
    # The stored parent channel and chat_type describe the ORIGIN conversation, so they are only
    # trusted when this target IS the origin. An explicit fan-out target (a user-typed
    # ``deliver=platform:chan``) keeps the home fallback: the user chose that destination, so a
    # dead one redirecting home is the intended behaviour, and no private origin is involved.
    parent_chat_id = t.origin.get("parent_chat_id") if t.origin_target else None
    suppress_home = t.origin_target and not home_escalation_allowed(t.origin, platform_name)

    fallback_targets = build_fallback_targets(
        {"platform": platform_name, "chat_id": chat_id, "thread_id": thread_id},
        parent_chat_id=parent_chat_id,
        home_chat_id=_get_home_target_chat_id(platform_name),
        home_thread_id=_get_home_target_thread_id(platform_name),
        is_direct_message=suppress_home)
    if not fallback_targets:
        return (False, None)

    fail_prefix = f"delivery to {t.where} failed: {error}; "
    last_error = error
    for fb in fallback_targets:
        kind = fb.get("fallback_kind", "fallback")
        fb_key = _target_key(fb["platform"], fb["chat_id"], fb.get("thread_id"))
        if fb_key in sent_keys:
            # An earlier target this run already delivered the identical content there.
            logger.info("Job '%s': %s fallback to %s skipped — content already delivered there "
                        "this run", job["id"], kind, fb["chat_id"])
            return (True, None)
        logger.warning("Job '%s': %s target %s:%s is undeliverable (%s); retrying %s channel %s",
                       job["id"], platform_name, chat_id, thread_id, last_error, kind,
                       fb["chat_id"])
        # Reuse the standalone sender (its loop/shutdown/timeout/profile-context handling is the
        # contract the primary send is held to) against the redirected location.
        fb_result, fb_err = _standalone_send(
            replace(t, chat_id=fb["chat_id"], thread_id=fb.get("thread_id")),
            format_fallback_notice(content, fb.get("fallback_kind")), media_files)
        if fb_err is None and fb_result and fb_result.get("error"):
            fb_err = fb_result["error"]
        if fb_err is not None:
            if is_definitive_delivery_failure(fb_err):
                last_error = fb_err
                continue
            msg = f"{fail_prefix}{kind} fallback to {fb['chat_id']} failed: {fb_err}"
            logger.error("Job '%s': %s", job["id"], msg)
            return (True, msg)

        sent_keys.add(fb_key)
        logger.info("Job '%s': delivered to %s:%s after %s fallback from %s:%s",
                    job["id"], fb["platform"], fb["chat_id"], kind, platform_name, chat_id)
        return (True, None)

    msg = f"{fail_prefix}all fallbacks failed; last error: {last_error}"
    logger.error("Job '%s': %s", job["id"], msg)
    return (True, msg)
