"""Slack button-click handling for the PR-review digest.

When the operator clicks Approve / Request changes / Comment, Slack dispatches
to :func:`handle_action` (wired in ``__init__.register``). The handler:

1. acks immediately (Slack's 3-second budget),
2. authorizes the clicking user against ``SLACK_ALLOWED_USERS``,
3. atomically pops the staged review from the store (double-click guard),
4. stale-guards: re-fetches the PR head SHA and refuses to post if it moved
   since the review was staged (the operator would be approving unseen code),
5. posts the *verbatim* staged body to GitHub via the ``gh`` wrapper,
6. rewrites the Slack message to strip the buttons and record the decision.

The GitHub and Slack effects are injected (``github`` module, ``client``) so the
whole flow is unit-tested without live calls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from . import blocks, github, store

logger = logging.getLogger(__name__)

_ACTION_EVENT = {
    "prreview_approve": "approve",
    "prreview_request_changes": "request-changes",
    "prreview_comment": "comment",
}


def event_for(action_id: str) -> Optional[str]:
    """Map a Slack action_id to a review event, or ``None`` if unknown."""
    return _ACTION_EVENT.get(action_id)


def is_authorized(user_id: str) -> bool:
    """Authorize a clicker against ``SLACK_ALLOWED_USERS``.

    Clicking a button posts a review to GitHub — a write action — and this
    plugin's handler is the ONLY gate (the adapter's plugin wrapper does not
    run the gateway's own interactive-auth check for us). So this fails CLOSED:
    empty/unset → deny. ``*`` → everyone. Otherwise the id must be in the
    comma-separated allowlist.
    """
    raw = os.environ.get("SLACK_ALLOWED_USERS")
    if raw is None or not raw.strip():
        logger.warning(
            "[pr_review_buttons] SLACK_ALLOWED_USERS not set — denying click by %s "
            "(posting is a write action; set the allowlist to enable buttons)",
            user_id,
        )
        return False
    allowed = {u.strip() for u in raw.split(",") if u.strip()}
    return "*" in allowed or user_id in allowed


def is_stale(*, stored_sha: Optional[str], current_sha: Optional[str]) -> bool:
    """True when the review can't be proven current against the live PR head.

    Fails CLOSED: a ``None`` current SHA (head fetch failed) counts as stale.
    The whole point is a what-you-approved gate — if we can't verify the head
    hasn't moved, we must not post a review of possibly-changed code. The
    staged entry is kept (not popped) on stale, so the operator just clicks
    again once ``gh`` recovers.
    """
    if current_sha is None:
        return True
    return stored_sha != current_sha


async def handle_action(ack, body, action, *, client, store_path: Optional[Path] = None) -> None:
    """Handle one button click. Always acks; posts at most once per staged PR."""
    await ack()

    action_id = action.get("action_id", "")
    key = action.get("value", "")
    event = event_for(action_id)

    user = body.get("user", {}) or {}
    user_id = user.get("id", "")
    user_name = user.get("name", user_id or "unknown")
    channel_id = (body.get("channel", {}) or {}).get("id", "")
    msg_ts = (body.get("message", {}) or {}).get("ts", "")

    if event is None:
        logger.warning("[pr_review_buttons] unknown action_id %r", action_id)
        return

    # Reject clicks with no identifiable user (malformed payload) before the
    # allowlist check — never post on behalf of an anonymous actor.
    if not user_id:
        logger.warning("[pr_review_buttons] click with no user id on %s — ignoring", key)
        return

    if not is_authorized(user_id):
        logger.warning(
            "[pr_review_buttons] unauthorized click by %s (%s) on %s — ignoring",
            user_name, user_id, key,
        )
        return

    # Peek before popping so a stale review stays in the store for re-review.
    entry = store.get(key, path=store_path)
    if entry is None:
        # Already actioned (double-click) or unknown key. Nothing to do.
        logger.info("[pr_review_buttons] no staged review for %s (already posted?)", key)
        return

    repo = entry.get("repo", "")
    number = entry.get("number", "")
    current = github.current_head_sha(repo, number)
    if is_stale(stored_sha=entry.get("head_sha"), current_sha=current):
        reviewed = str(entry.get("head_sha") or "?")[:8]
        if current is None:
            context = (
                f"⚠️ Not posted by {user_name}: could not verify the PR head via "
                f"`gh` (transient error?). Review kept — click again to retry."
            )
        else:
            context = (
                f"⚠️ Not posted by {user_name}: the PR head moved "
                f"(reviewed `{reviewed}`, now `{str(current)[:8]}`). "
                f"Review kept — re-review to refresh."
            )
        await _rewrite(
            client, channel_id, msg_ts, entry,
            text=f"⚠️ {repo}#{number} not posted (stale/unverified)",
            context=context,
        )
        logger.info("[pr_review_buttons] stale/unverified review for %s — skipped post", key)
        return

    # Atomic pop = commit point. First caller wins; a racing double-click gets
    # None and returns without posting again.
    entry = store.pop(key, path=store_path)
    if entry is None:
        return

    result = github.submit_review(repo, number, event, entry.get("body", ""))

    if result.ok:
        posted_blocks = blocks.build_posted_blocks(
            entry, event=event, user=user_name, detail=result.detail,
        )
        await _update(
            client, channel_id, msg_ts,
            text=f"{repo}#{number}: {event} posted by {user_name}",
            blocks_payload=posted_blocks,
        )
        logger.info(
            "[pr_review_buttons] posted %s review to %s#%s by %s (%s)",
            event, repo, number, user_name, result.detail,
        )
    elif result.own_pr:
        # Deliberate refusal: it's the operator's own PR. Retrying can never
        # succeed, so drop it (already popped) and show a clean skip — no
        # buttons, no re-stage.
        await _rewrite(
            client, channel_id, msg_ts, entry,
            text=f"🚫 {repo}#{number} not posted (your own PR)",
            context=(
                f"🚫 Skipped by {user_name}: Hermes does not review your own PRs. "
                f"Nothing was posted to GitHub."
            ),
        )
        logger.info(
            "[pr_review_buttons] refused own-PR post for %s#%s (clicked by %s)",
            repo, number, user_name,
        )
    else:
        # Post failed — re-stage so the operator can retry, and surface why.
        store.stage(entry, path=store_path)
        await _rewrite(
            client, channel_id, msg_ts, dict(entry, _key=key),
            text=f"❌ Failed to post {repo}#{number}",
            context=f"❌ Post failed ({user_name}): {result.detail[:300]}. Re-staged — try again.",
            keep_buttons=True,
        )
        logger.error(
            "[pr_review_buttons] failed to post %s to %s#%s: %s",
            event, repo, number, result.detail,
        )


async def _update(client, channel, ts, *, text, blocks_payload) -> None:
    try:
        await client.chat_update(channel=channel, ts=ts, text=text, blocks=blocks_payload)
    except Exception as exc:  # pragma: no cover - defensive
        # The GitHub post already succeeded (or failed) by the time we get here;
        # a failed message rewrite only means the buttons/decision line are
        # stale in Slack. Log loudly at error so the operator can reconcile —
        # never silently. The click itself is not retried (already acked).
        logger.error(
            "[pr_review_buttons] chat_update failed for %s@%s — Slack message not "
            "updated (GitHub state already applied): %s",
            channel, ts, exc,
        )


async def _rewrite(client, channel, ts, entry, *, text, context, keep_buttons: bool = False) -> None:
    """Rewrite the message with a context line; optionally keep the buttons."""
    if keep_buttons:
        payload = blocks.build_digest_blocks([dict(entry)])
        payload.append({"type": "context", "elements": [{"type": "mrkdwn", "text": context}]})
    else:
        payload = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": context}]},
        ]
    await _update(client, channel, ts, text=text, blocks_payload=payload)
