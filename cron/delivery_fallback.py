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
