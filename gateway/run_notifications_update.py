"""Gateway /update notification lifecycle for GatewayRunner.

Split from :mod:`gateway.run_notifications` so update IPC, terminal-receipt
correlation, and the notification watcher do not grow the general notification
facade past the repository's file-size/ownership limits.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform
# Completion notification metadata needs the topic-agnostic, non-conversational route.
from gateway.run_shutdown import _log_suppressed

logger = logging.getLogger("gateway.run")

# A failed /update leaves the previous version running; the full pip/git log stays on the host
# (`hermes update` re-runs it in the terminal) and only a short tail is quoted in chat.
_UPDATE_FAILED_NOTICE = (
    "❌ Hermes update failed; the previous version is still running. Run `hermes update` on the "
    "host to see the full error, or try /update again later.")

# An update's completion notice waits for its target platform adapter to (re)connect before it
# can be delivered. Nothing bounds that wait, so a marker naming a platform that is not
# configured at all — no adapter will ever appear — would keep itself on disk and re-log a
# deferred line on every poll, in every process, forever. Stop waiting past this age.
_UPDATE_NOTIFY_MAX_ADAPTER_WAIT_SECONDS = 3600.0


def _update_output_tail(output: str, limit: int) -> str:
    """Last ``limit`` chars of an update log, prefixed with an ellipsis when cut."""
    return output if len(output) <= limit else "…" + output[-limit:]



class GatewayUpdateNotificationsMixin:
    """Own the gateway-mode /update watcher and its durable completion exchange."""

    @dataclasses.dataclass
    class _UpdatePaths:
        """Marker files ``hermes update --gateway`` and its watcher exchange under HERMES_HOME."""

        pending: Path
        claimed: Path
        output: Path
        exit_code: Path
        prompt: Path
        response: Path

        def any_pending(self) -> bool:
            return self.pending.exists() or self.claimed.exists()

        def unlink_all(self) -> None:
            for p in (self.pending, self.claimed, self.output, self.exit_code, self.prompt, self.response):
                p.unlink(missing_ok=True)

    @dataclasses.dataclass
    class _UpdateTarget:
        """Resolved delivery target for update watcher messages."""

        adapter: Any
        chat_id: Any
        session_key: Optional[str]
        metadata: Any
        platform: Any

        def send_metadata(self):
            from gateway.run import _non_conversational_metadata
            return _non_conversational_metadata(self.metadata, platform=self.platform)

        async def send(self, text: str):
            return await self.adapter.send(self.chat_id, text, metadata=self.send_metadata())

    def _schedule_update_notification_watch(self) -> None:
        """Ensure a background task is watching for update completion."""
        existing_task = getattr(self, "_update_notification_task", None)
        if existing_task and not existing_task.done():
            return
        try:
            self._update_notification_task = asyncio.create_task(self._watch_update_progress())
        except RuntimeError:
            logger.debug("Skipping update notification watcher: no running event loop")

    @classmethod
    def _update_paths(cls) -> "GatewayUpdateNotificationsMixin._UpdatePaths":
        from gateway.run import _hermes_home
        return cls._UpdatePaths(
            pending=_hermes_home / ".update_pending.json",
            claimed=_hermes_home / ".update_pending.claimed.json", output=_hermes_home / ".update_output.txt",
            exit_code=_hermes_home / ".update_exit_code",
            prompt=_hermes_home / ".update_prompt.json", response=_hermes_home / ".update_response",
        )

    @staticmethod
    def _marker_profile(data: dict) -> Optional[str]:
        """Owning profile of a persisted restart/update marker: explicit ``profile``, else the
        ``agent:<profile>:`` lane of its ``session_key`` (markers written before ``profile`` was
        persisted); ``None`` = default profile."""
        profile = str(data.get("profile") or "").strip()
        if profile:
            return profile
        from gateway.session import profile_from_session_key_namespace
        parts = str(data.get("session_key") or "").split(":")
        if len(parts) >= 5 and parts[0] == "agent" and parts[1] not in ("main", ""):
            return profile_from_session_key_namespace(parts[1])
        return None

    @staticmethod
    def _marker_age_seconds(data: dict) -> Optional[float]:
        """Age of a persisted update marker, from the ``timestamp`` stamped by its writer.

        ``None`` when the marker carries no parseable stamp — the field is absent on markers
        written before it existed, and callers keep the old retry behavior rather than guess.
        """
        raw = str(data.get("timestamp") or "").strip()
        if not raw:
            return None
        try:
            stamped = datetime.fromisoformat(raw)
        except ValueError:
            return None
        # The writer stamps a naive local ``datetime.now()``; tolerate a tz-aware one too.
        now = datetime.now(stamped.tzinfo) if stamped.tzinfo else datetime.now()
        return (now - stamped).total_seconds()

    def _resolve_update_target(self, paths: "_UpdatePaths") -> Optional["_UpdateTarget"]:
        """Resolve adapter/chat/session for update watcher messages from the pending marker."""
        for path in (paths.claimed, paths.pending):
            if not path.exists():
                continue
            with suppress(Exception):
                pending = json.loads(path.read_text(encoding="utf-8-sig"))
                platform_str = pending.get("platform")
                chat_id = pending.get("chat_id")
                session_key = pending.get("session_key")
                if not (platform_str and chat_id):
                    continue  # BASE: an incomplete marker falls through to the next path, not "unresolved"
                platform = Platform(platform_str)
                # The requester's OWN profile bot (marker ``profile``, else the ``agent:<profile>:`` key
                # lane); a bare self.adapters lookup is the default bot under multiplex.
                adapter = self._authorization_adapter(platform, self._marker_profile(pending))
                if not adapter:
                    return None
                metadata = self._pending_marker_metadata(platform, chat_id, pending, adapter)
                # Fallback session key if not stored (old pending files)
                return self._UpdateTarget(
                    adapter, chat_id, session_key or f"{platform_str}:{chat_id}", metadata, platform,
                )
        return None

    def _pending_marker_metadata(self, platform, chat_id, data: dict, adapter):
        """Thread metadata for a persisted update/restart marker (thread_id/chat_type/message_id keys)."""
        return self._thread_metadata_for_target(
            platform, chat_id, data.get("thread_id"), chat_type=data.get("chat_type"),
            reply_to_message_id=data.get("message_id"), adapter=adapter,
        )

    async def _watch_update_completion_only(self, paths: "_UpdatePaths", deadline: float, poll_interval: float) -> None:
        """Fallback when no adapter/chat can be resolved: wait for the exit code, then notify."""
        logger.warning("Update watcher: cannot resolve adapter/chat_id, falling back to completion-only")
        # Poll until _send_update_notification delivers (it returns False while the platform reconnects).
        loop = asyncio.get_running_loop()
        while paths.any_pending() and loop.time() < deadline:
            if paths.exit_code.exists() and await self._send_update_notification():
                return
            await asyncio.sleep(poll_interval)
        if paths.any_pending() and not paths.exit_code.exists():
            paths.exit_code.write_text("124", encoding="utf-8")
            await self._send_update_notification()
        elif (
            paths.any_pending()
            and paths.exit_code.exists()
            and self._update_exit_code(paths) == 0
            and not self._gateway_update_finalized(paths)
        ):
            logger.warning(
                "Update watcher (completion-only) timed out with unfinished finalize; "
                "leaving markers for a later retry instead of reporting success",
            )

    @staticmethod
    def _update_exit_code(paths: "_UpdatePaths") -> int:
        return int(paths.exit_code.read_text(encoding="utf-8-sig").strip() or "1")

    @staticmethod
    def _gateway_update_finalized(paths: "_UpdatePaths") -> bool:
        """True when the exact run published a successful terminal receipt.

        ``.update_exit_code`` is published before fleet restart, so zero there
        proves only the pull/maintenance half. Resolve the receipt by the
        update id persisted for this invocation; never trust ``latest.json``
        or filesystem mtimes, which another update or a wrapper rewrite can
        reorder. Unknown or malformed state fails closed.
        """
        try:
            marker_path = paths.claimed if paths.claimed.exists() else paths.pending
            marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
            update_id = str(marker.get("update_id") or "").strip()
            if not update_id:
                return False
            directory = paths.exit_code.parent / "logs" / "update_receipts"
            for path in directory.glob("update_*.json"):
                try:
                    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError, TypeError):
                    continue
                if not (
                    str(receipt.get("update_id") or "") == update_id
                    and receipt.get("finished_at")
                    and receipt.get("outcome") == "success"
                    and receipt.get("exit_code") == 0
                ):
                    continue
                restart = receipt.get("gateway_restart")
                if isinstance(restart, dict) and (
                    restart.get("incomplete") is True or restart.get("phase_error")
                ):
                    continue
                from hermes_cli.update_cmd_fleet import _fleet_restart_obligation_armed
                return not _fleet_restart_obligation_armed()
        except Exception:
            return False
        return False

    @staticmethod
    def _read_update_output_since(path: Path, offset: int) -> tuple[str, int]:
        """Read update output defensively; logs may contain invalid UTF-8."""
        try:
            data = path.read_bytes()
        except OSError:
            return "", offset
        if len(data) <= offset:
            return "", len(data)
        return data[offset:].decode("utf-8", errors="replace"), len(data)

    async def _send_update_output(self, target: "_UpdateTarget", text: str) -> None:
        """Send buffered update output as fenced chunks that fit message limits (Telegram: 4096)."""
        from tools.ansi_strip import strip_ansi
        clean = strip_ansi(text).strip()
        if not clean:
            return
        max_chunk = 3500
        for i in range(0, len(clean), max_chunk):
            with _log_suppressed(logging.DEBUG, "Update stream send failed: %s"):
                await target.send(f"```\n{clean[i:i + max_chunk]}\n```")

    async def _forward_update_prompt(self, target: "_UpdateTarget", prompt_text: str, default: str) -> None:
        """Forward an update prompt: platform-native buttons first (Discord, Telegram), else text."""
        sent_buttons = False
        adapter = target.adapter
        if getattr(type(adapter), "send_update_prompt", None) is not None:
            with _log_suppressed(logging.DEBUG, "Button-based update prompt failed: %s"):
                await adapter.send_update_prompt(
                    chat_id=target.chat_id, prompt=prompt_text, default=default,
                    session_key=target.session_key, metadata=target.send_metadata(),
                )
                sent_buttons = True
        if not sent_buttons:
            default_hint = f" (default: {default})" if default else ""
            _p = getattr(adapter, "typed_command_prefix", "/")
            await target.send(
                f"☤ **Update needs your input:**\n\n{prompt_text}{default_hint}\n\n"
                f"Reply `{_p}approve` (yes) or `{_p}deny` (no), or type your answer directly."
            )
        # Keep the prompt marker on disk until answered so a restarted watcher can re-forward it.
        self._session_state(target.session_key).persistent.update_prompt_pending = True
        logger.info("Forwarded update prompt to %s: %s", target.session_key, prompt_text[:80])

    def _clear_update_markers(self, paths: "_UpdatePaths", session_key: Optional[str]) -> None:
        paths.unlink_all()
        state = self._peek_session_state(session_key)
        if state is not None:
            state.persistent.update_prompt_pending = False

    async def _watch_update_progress(
        self, poll_interval: float = 2.0, stream_interval: float = 4.0, timeout: float = 1800.0
    ) -> None:
        """Watch ``hermes update --gateway``, streaming output + forwarding prompts.

        Polls ``.update_output.txt`` for new content and sends chunks to the user periodically;
        detects ``.update_prompt.json`` (written when the update process needs input) and forwards it.
        """
        paths = self._update_paths()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        target = self._resolve_update_target(paths)
        if target is None:
            await self._watch_update_completion_only(paths, deadline, poll_interval)
            return
        session_key = target.session_key
        bytes_sent = 0
        last_stream_time = loop.time()
        buffer = ""

        async def _flush_buffer() -> None:
            nonlocal buffer, last_stream_time
            text, buffer = buffer, ""
            if text.strip():
                last_stream_time = loop.time()
                await self._send_update_output(target, text)

        def _read_new_output() -> None:
            nonlocal buffer, bytes_sent
            if paths.output.exists():
                with suppress(OSError):
                    chunk, bytes_sent = self._read_update_output_since(paths.output, bytes_sent)
                    buffer += chunk

        while loop.time() < deadline:
            if paths.exit_code.exists():
                _read_new_output()
                await _flush_buffer()
                with _log_suppressed(logging.WARNING, "Update final notification failed: %s"):
                    exit_code = self._update_exit_code(paths)
                    if exit_code != 0:
                        await target.send(_UPDATE_FAILED_NOTICE)
                        logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                        self._clear_update_markers(paths, session_key)
                        return
                    if self._gateway_update_finalized(paths):
                        await target.send("✅ Hermes update finished.")
                        logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                        self._clear_update_markers(paths, session_key)
                        return
                    logger.info(
                        "Update exit code %s seen but finalize evidence missing (receipt/marker), keep watching",
                        exit_code,
                    )
            _read_new_output()
            if buffer.strip() and (loop.time() - last_stream_time) >= stream_interval:
                await _flush_buffer()
            # Forward a prompt only when none is pending, else every poll re-forwards the same prompt.
            _pending_state = self._peek_session_state(session_key) if session_key else None
            if paths.prompt.exists() and session_key and not getattr(
                getattr(_pending_state, "persistent", None), "update_prompt_pending", False
            ):
                try:
                    prompt_data = json.loads(paths.prompt.read_text(encoding="utf-8-sig"))
                    prompt_text = prompt_data.get("prompt", "")
                    if prompt_text:
                        await _flush_buffer()  # user sees context before the prompt
                        await self._forward_update_prompt(target, prompt_text, prompt_data.get("default", ""))
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Failed to read update prompt: %s", e)
            await asyncio.sleep(poll_interval)
        if not paths.exit_code.exists():
            logger.warning("Update watcher timed out after %.0fs", timeout)
            paths.exit_code.write_text("124", encoding="utf-8")
            await _flush_buffer()
            with suppress(Exception):
                await target.send("❌ Hermes update timed out after 30 minutes.")
            self._clear_update_markers(paths, session_key)
        elif self._update_exit_code(paths) == 0 and not self._gateway_update_finalized(paths):
            logger.warning(
                "Update watcher timed out with unfinished finalize (receipt/marker) after %.0fs", timeout,
            )
            await _flush_buffer()
            with suppress(Exception):
                await target.send(
                    "❌ Hermes update did not finalize — a fleet restart is still pending."
                    " Run `hermes update` to retry, or `hermes gateway restart` after clearing the marker."
                )
            self._clear_update_markers(paths, session_key)

    async def _send_update_notification(self) -> bool:
        """If an update finished, notify the user.

        False while the update is still running (caller may retry); True after a definitive send/skip.
        """
        from gateway.run import _non_conversational_metadata
        paths = self._update_paths()
        if not paths.any_pending():
            return False
        cleanup = True
        active_pending_path = paths.claimed

        def _defer(reason: str, *args) -> bool:
            nonlocal cleanup, active_pending_path
            logger.info(reason, *args)
            cleanup = False
            active_pending_path = paths.pending
            paths.claimed.replace(paths.pending)
            return False

        try:
            if paths.pending.exists():
                try:
                    paths.pending.replace(paths.claimed)
                except FileNotFoundError:
                    if not paths.claimed.exists():
                        return True
            elif not paths.claimed.exists():
                return True
            pending = json.loads(paths.claimed.read_text(encoding="utf-8-sig"))
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")
            if not paths.exit_code.exists():
                return _defer("Update notification deferred: update still running")
            exit_code = self._update_exit_code(paths)
            if exit_code == 0 and not self._gateway_update_finalized(paths):
                return _defer(
                    "Update notification deferred: finalize evidence missing (receipt/marker) — wait for updater",
                )
            output = paths.output.read_bytes().decode("utf-8", errors="replace") if paths.output.exists() else ""
            platform = Platform(platform_str)
            adapter = self._authorization_adapter(platform, self._marker_profile(pending))
            if chat_id and not adapter:
                age = self._marker_age_seconds(pending)
                if age is not None and age > _UPDATE_NOTIFY_MAX_ADAPTER_WAIT_SECONDS:
                    # The platform never came back. Deferring forever leaks the markers and re-logs
                    # on every poll for the life of the install: the startup path reschedules this
                    # watcher whenever the markers are still on disk, so an undeliverable marker
                    # outlives every restart. Give up loudly, clear the markers, and report a
                    # definitive decision (True) so the caller stops rescheduling.
                    logger.warning(
                        "Post-update notification for %s:%s dropped after %.1fh: %s adapter never "
                        "connected", platform_str, chat_id, age / 3600.0, platform_str)
                    self._clear_update_markers(paths, pending.get("session_key"))
                    return True
                # Target platform not reconnected yet (common right after the update's restart): keep the
                # markers for a later retry instead of silently losing the notification.
                return _defer("Update notification deferred: %s adapter not connected yet", platform_str)
            if chat_id:
                metadata = self._pending_marker_metadata(platform, chat_id, pending, adapter)
                from tools.ansi_strip import strip_ansi
                output = strip_ansi(output).strip()
                if exit_code == 0:
                    msg = "✅ Hermes update finished successfully."
                    if output:
                        msg = f"{msg}\n\n```\n{_update_output_tail(output, 3500)}\n```"
                else:
                    msg = _UPDATE_FAILED_NOTICE
                    if output:
                        msg = f"{msg}\n\nLast lines:\n```\n{_update_output_tail(output, 800)}\n```"
                await adapter.send(chat_id, msg, metadata=_non_conversational_metadata(metadata, platform=platform))
                logger.info("Sent post-update notification to %s:%s (exit=%s)", platform_str, chat_id, exit_code)
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            if cleanup:
                for p in (active_pending_path, paths.claimed, paths.output, paths.exit_code):
                    p.unlink(missing_ok=True)
        return True
