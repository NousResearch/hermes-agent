"""Cross-profile slash relay for the Slack adapter: intake, consumer and replies.

Slack slash names are workspace-global; with several profile apps in one workspace, Slack
delivers every colliding /command to a single app. The receiving adapter resolves the invoking
channel's owning profile (``slash_relay.resolve_channel_owner``) and, when it is foreign,
enqueues the payload on a shared SQLite queue. Every adapter polls its own inbox and runs
relayed payloads through its normal slash pipeline, replying with its own bot token so Slack
shows the owning profile's app. Also holds the bot-token ephemeral delivery both reply paths use.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from typing import Any, Callable, Optional

from gateway.platforms._shared import extra_or_secret as _extra_or_secret
from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.slack.adapter")

_RELAY_OFF = {"false", "0", "no", "off"}
# raw_response marker on a failed relayed reply: see SlackSlashRelayMixin._send_retry_is_final.
_PRIVATE_REPLY = "private_slash_reply"


def _queue():
    """The relay queue module (late import: it reads the hermes root on use)."""
    try:
        from plugins.platforms.slack import slash_relay
    except ImportError:  # pragma: no cover - plugin loaded outside package context
        import slash_relay  # type: ignore
    return slash_relay


def _slash_name(command: dict) -> str:
    return (command.get("command") or "").lstrip("/")


class SlackSlashRelayMixin:
    config: Any
    _running: bool
    _team_clients: dict
    _get_client: Callable[..., Any]
    _metadata_team_id: Callable[..., str]
    _format_chunks: Callable[[str], list]
    _lazy_attr: Callable[..., Any]
    _handle_slash_command: Callable[..., Any]
    _send_slash_ephemeral: Callable[..., Any]
    _clear_thread_status_quietly: Callable[..., Any]

    # Consumer task, the profile name captured at connect, and the pending "owner never
    # claimed it" warning tasks (created on first use).
    _slash_relay_task: Optional[asyncio.Task] = None
    _slash_relay_poll_s = 1.5
    _slash_relay_profile_name: Optional[str] = None
    _slash_relay_warn_tasks: Optional[set] = None

    def _slash_relay_enabled(self) -> bool:
        """Kill switch: ``slack.slash_relay: false`` in config.yaml (bridged into
        ``config.extra`` by ``_YAML_BRIDGE``) or ``SLACK_SLASH_RELAY=false``. Default on."""
        raw = _extra_or_secret(self.config.extra, "slash_relay", "SLACK_SLASH_RELAY", None)
        return raw is None or str(raw).strip().lower() not in _RELAY_OFF

    def _slash_relay_profile(self) -> str:
        """The profile name this adapter represents: the name captured at
        connect, else the current scope's."""
        return self._slash_relay_profile_name or self._resolve_slash_relay_profile()

    @staticmethod
    def _resolve_slash_relay_profile() -> str:
        """Profile of the current scope. Under a multiplexed gateway a secondary
        profile's adapter connects inside that profile's ``HERMES_HOME`` override
        (a ContextVar), which ``current_profile_name`` reads first."""
        try:
            from hermes_cli.profiles import current_profile_name

            return current_profile_name("default") or "default"
        except Exception:
            logger.debug("[Slack] slash relay: profile name unresolved", exc_info=True)
            return "default"

    # ── Intake (receiving app) ──────────────────────────────────────────

    def _resolve_foreign_slash_owner(self, command: dict) -> Optional[str]:
        """Return the owning profile of the invoking channel when it is NOT
        this gateway's profile; None means handle the command locally."""
        if not self._slash_relay_enabled():
            return None
        channel_id = str(command.get("channel_id") or "")
        if not channel_id:
            return None
        try:
            owner = _queue().resolve_channel_owner(channel_id)
        except Exception:
            logger.debug("[Slack] Slash relay owner lookup failed", exc_info=True)
            return None
        if owner is None or owner == self._slash_relay_profile():
            return None
        return owner

    async def _maybe_relay_slash(self, ack: Callable[..., Any], command: dict) -> bool:
        """Relay a slash typed in another profile's channel; True when it was taken.

        Runs before the channel gate: a foreign channel is by definition outside this
        profile's allowed_channels, and the owner applies its own gate on arrival. The ack
        is silent because any text here (or any response_url POST) is attributed to THIS
        app; the owner replies with its own bot token."""
        # The lookup may rescan every profile's files: keep it off the loop (3 s ack window).
        owner = await asyncio.to_thread(self._resolve_foreign_slash_owner, command)
        if not owner:
            return False
        await ack()
        await self._forward_slash_to_profile(owner, command)
        return True

    async def _forward_slash_to_profile(self, owner: str, command: dict) -> None:
        """Enqueue a foreign-channel slash payload for *owner*'s gateway.

        A failed enqueue refuses the command privately. It never runs here: the owner is
        known, and this profile is the wrong one to act for it."""
        slash = _slash_name(command)
        try:
            row_id = await asyncio.to_thread(
                _queue().enqueue, owner, self._slash_relay_profile(), command)
        except Exception:
            logger.warning(
                "[Slack] Slash relay enqueue for profile %s failed; /%s was not run",
                owner, slash, exc_info=True)
            if command.get("response_url"):
                await self._send_slash_ephemeral(
                    {"response_url": command["response_url"], "replace_original": False},
                    f"⚠️ `/{slash}` could not be passed to *{owner}*'s gateway, "
                    "so it was not run. Try again.")
            return
        logger.info(
            "[Slack] Relayed /%s in %s to profile %s (row %d)",
            slash, command.get("channel_id"), owner, row_id)
        if command.get("response_url"):
            # Keep a reference (a bare create_task can be garbage-collected)
            # so disconnect() can cancel a warning that is still waiting.
            warn_tasks = self._lazy_attr("_slash_relay_warn_tasks", set)
            warn_task = asyncio.create_task(self._warn_if_slash_unclaimed(owner, command, row_id))
            warn_tasks.add(warn_task)
            warn_task.add_done_callback(warn_tasks.discard)

    async def _warn_if_slash_unclaimed(
        self, owner: str, command: dict, row_id: int, delay_s: float = 20.0) -> None:
        """Tell the user privately if *owner* never picks the relayed command
        up (its gateway is likely down)."""
        await asyncio.sleep(delay_s)
        try:
            claimed = await asyncio.to_thread(_queue().is_claimed, row_id)
        except Exception:
            logger.debug("[Slack] slash relay: claim check failed", exc_info=True)
            return
        if claimed:
            return
        # The ack was silent, so create a new ephemeral rather than replacing a
        # (nonexistent) original.
        await self._send_slash_ephemeral(
            {"response_url": command["response_url"], "replace_original": False},
            f"⚠️ *{owner}*'s gateway hasn't picked up `/{_slash_name(command)}` — is it running?")

    # ── Consumer (owning app) ───────────────────────────────────────────

    def _ensure_slash_relay_task(self) -> None:
        # Capture the profile now: connect() runs inside the profile's scope, and
        # Bolt's dispatch tasks must not depend on inheriting that ContextVar.
        self._slash_relay_profile_name = self._resolve_slash_relay_profile()
        if not self._slash_relay_enabled():
            return
        if self._slash_relay_task is None or self._slash_relay_task.done():
            self._slash_relay_task = asyncio.create_task(self._slash_relay_loop())

    async def _stop_slash_relay(self) -> None:
        """Cancel the consumer and any pending unclaimed-row warnings (disconnect)."""
        relay_task, self._slash_relay_task = self._slash_relay_task, None
        if relay_task is not None and not relay_task.done():
            relay_task.cancel()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("[Slack] Slash relay task raised during disconnect", exc_info=True)
        warn_tasks, self._slash_relay_warn_tasks = self._slash_relay_warn_tasks or set(), set()
        for warn_task in warn_tasks:
            warn_task.cancel()
        if warn_tasks:
            await asyncio.gather(*warn_tasks, return_exceptions=True)

    async def _slash_relay_loop(self) -> None:
        """Poll the shared relay for slash payloads addressed to this profile and
        run each through the normal slash pipeline, one row at a time. Each poll
        also purges rows older than 24 h (``slash_relay.claim_next``)."""
        profile = self._slash_relay_profile()
        logger.info("[Slack] Slash relay consumer started (profile %s)", profile)
        while self._running:
            try:
                row = await self._claim_relayed_slash(profile)
                if row is not None:
                    await self._run_relayed_slash(row)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("[Slack] Slash relay loop error", exc_info=True)
            await asyncio.sleep(self._slash_relay_poll_s)

    async def _claim_relayed_slash(self, profile: str) -> Optional[dict]:
        """Claim the next row. The claim finishes in its thread even if this task is
        cancelled meanwhile; a row claimed that way never started, so it goes back."""
        queue = _queue()
        claim = asyncio.ensure_future(asyncio.to_thread(queue.claim_next, profile))
        try:
            return await asyncio.shield(claim)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                row = await claim
                if row is not None:
                    await asyncio.to_thread(queue.release_claim, row["id"])
            raise

    async def _run_relayed_slash(self, row: dict) -> None:
        """Run one claimed row. A handler that returned or raised has run: the row is
        done. A cancelled handler may have acted, so its row stays claimed and is never
        re-delivered."""
        try:
            await self._handle_slash_command(row["payload"], relayed=True)
        except asyncio.CancelledError:
            logger.warning(
                "[Slack] Relayed slash row %s interrupted mid-run; left unfinished, not retried",
                row["id"])
            raise
        except Exception:
            logger.warning("[Slack] Relayed slash command failed", exc_info=True)
        await asyncio.to_thread(_queue().mark_done, row["id"])

    def _mark_relayed_slash_context(self, ctx: dict[str, Any], team_id: str) -> None:
        """Flag a stashed slash context as relayed from another profile's app: reply via
        THIS bot so Slack shows the owning profile's app name, in the invoking workspace,
        with response_url only as fallback. The receiving app acked silently, so a fallback
        POST must create a new message, not replace one."""
        ctx.update(via_bot=True, replace_original=False, team_id=str(team_id or ""))

    # ── Replies ─────────────────────────────────────────────────────────

    def _relayed_reply_client(
        self, chat_id: str, ctx: dict[str, Any], metadata: Optional[dict[str, Any]],
    ) -> tuple[Any, str]:
        """(bot client, workspace id) for the invoking workspace. The client is None when
        a workspace is named but this adapter has no client for it: never widen to
        another workspace's client."""
        team_id = self._metadata_team_id(metadata) or str(ctx.get("team_id") or "")
        if not team_id:
            return self._get_client(chat_id), ""
        return self._team_clients.get(team_id), team_id

    async def _send_relayed_slash_reply(
        self, chat_id: str, ctx: dict[str, Any], content: str,
        metadata: Optional[dict[str, Any]]) -> SendResult:
        """Private reply to a relayed slash (see ``_deliver_relayed_slash_reply``).
        Ephemerals don't auto-clear the Assistant status, so clear it on success."""
        result = await self._deliver_relayed_slash_reply(chat_id, ctx, content, metadata)
        if result.success:
            await self._clear_thread_status_quietly(chat_id, metadata)
            return result
        raw = result.raw_response if isinstance(result.raw_response, dict) else {}
        return dataclasses.replace(result, raw_response={**raw, _PRIVATE_REPLY: True})

    def _send_retry_is_final(self, result: SendResult) -> bool:
        """A failed relayed slash reply is final. ``_send_with_retry``'s resend and plain-text
        fallback go back through send() after the slash context is used up, so they would post
        the private reply publicly."""
        raw = getattr(result, "raw_response", None)
        if isinstance(raw, dict) and raw.get(_PRIVATE_REPLY):
            return True
        return super()._send_retry_is_final(result)  # type: ignore[misc]

    async def _deliver_relayed_slash_reply(
        self, chat_id: str, ctx: dict[str, Any], content: str,
        metadata: Optional[dict[str, Any]]) -> SendResult:
        """Private reply to a relayed slash, posted with THIS bot's token.

        A ``chat.postEphemeral`` from the owning profile's bot is attributed to that
        profile's app, unlike a ``response_url`` POST, which always shows the app that
        received the slash. Every chunk is posted. If nothing was delivered, the whole
        reply goes through ``response_url`` instead (still private). If a prefix was
        delivered, it is never replayed: the user is told the reply was cut off and the
        result is a partial delivery, which ``_send_with_retry`` does not re-send."""
        chunks = self._format_chunks(content)
        user_id = str(ctx.get("user_id") or "")
        sent, error, client = 0, "no user_id in slash context", None
        if user_id:
            client, team_id = self._relayed_reply_client(chat_id, ctx, metadata)
            error = None if client is not None else f"no Slack client for workspace {team_id}"
        if client is not None:
            sent, error = await self._post_ephemeral_chunks(client, chat_id, user_id, chunks)
            if error is None:
                return SendResult(success=True, message_id=None)
        if sent == 0:
            logger.warning(
                "[Slack] Relayed slash reply not posted by bot (%s); falling back to response_url",
                error)
            if ctx.get("response_url"):
                return await self._send_slash_ephemeral(ctx, content)
            return SendResult(success=False, error=error or "No ephemeral route available")
        logger.warning(
            "[Slack] Relayed slash reply cut off after %d of %d parts: %s", sent, len(chunks), error)
        if ctx.get("response_url"):
            await self._send_slash_ephemeral(
                {"response_url": ctx["response_url"], "replace_original": False},
                f"⚠️ This reply was cut off after part {sent} of {len(chunks)}.")
        return SendResult(
            success=False, error=f"reply cut off after {sent} of {len(chunks)} parts: {error}",
            raw_response={"partial_overflow": True, "delivered_parts": sent})

    async def _post_ephemeral_chunks(
        self, client: Any, chat_id: str, user_id: str, chunks: list) -> tuple[int, Optional[str]]:
        """Post *chunks* in order with ``chat.postEphemeral``, stopping at the first
        failure. Returns (parts delivered, error or None)."""
        try:
            from plugins.platforms.slack.adapter import _slack_response_payload
        except ImportError:  # pragma: no cover - plugin loaded outside package context
            from adapter import _slack_response_payload  # type: ignore

        for sent, chunk in enumerate(chunks):
            try:
                result = await client.chat_postEphemeral(channel=chat_id, user=user_id, text=chunk)
            except Exception as e:  # any SDK/transport error ends delivery at this chunk
                logger.warning("[Slack] chat.postEphemeral failed after %d part(s)", sent, exc_info=True)
                return sent, str(e)
            payload = _slack_response_payload(result)
            if not payload.get("ok"):
                err = payload.get("error", "unknown_error") if payload else "unexpected_response"
                return sent, f"chat.postEphemeral failed: {err}"
        return len(chunks), None

    async def _post_ephemeral_fallback(
        self, chat_id: str, ctx: dict[str, Any], content: str) -> SendResult:
        """Deliver a slash reply via ``chat.postEphemeral`` when ``response_url`` fails.
        Keeps the reply private (a public channel post must never happen for an ephemeral reply).
        Cannot ``replace_original``, so the ack stays; no 5-POST cap applies here.

        See #19688.
        """
        user_id = ctx.get("user_id", "")
        if not user_id:
            return SendResult(success=False, error="no user_id in slash context for postEphemeral")
        chunks = self._format_chunks(content)
        try:
            client = self._get_client(chat_id)
        except Exception as e:
            logger.warning("[Slack] No client for ephemeral slash reply in %s", chat_id, exc_info=True)
            return SendResult(success=False, error=str(e))
        _sent, error = await self._post_ephemeral_chunks(client, chat_id, user_id, chunks)
        if error is not None:
            return SendResult(success=False, error=error)
        return SendResult(success=True, message_id=None)
