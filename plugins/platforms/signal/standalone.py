"""Out-of-process Signal send (cron / send_message) — format, split, rate-limit, JSON-RPC.

Owned by the Signal plugin so tools/gateway callers only touch
``PlatformEntry.standalone_sender_fn`` after discovery.
"""

from __future__ import annotations

import logging
import os
import time

from . import signal_rate_limit as rl
from .signal_format import markdown_to_signal

logger = logging.getLogger(__name__)


def _error(message: str) -> dict:
    return {"error": str(message)}


def _success(platform: str, chat_id, warnings=None, **fields) -> dict:
    return {"success": True, "platform": platform, "chat_id": chat_id, **fields,
            **({"warnings": warnings} if warnings else {})}


async def _signal_send_batch(post, scheduler, idx, n_batches, att_batch, batch_message):
    """One Signal batch under the scheduler with rate-limit retries: None on success, False when
    retries were exhausted (batch lost), error dict for a non-rate-limit RPC error."""
    n, max_attempts = len(att_batch), rl.SIGNAL_RATE_LIMIT_MAX_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        try:
            await scheduler.acquire(n)
            _rpc_t0 = time.monotonic()
            data = await post(att_batch, batch_message)
            if "error" not in data:
                await scheduler.report_rpc_duration(time.monotonic() - _rpc_t0, n)
                return None
            err = data["error"]
            if not rl._is_signal_rate_limit_error(err):
                return _error(f"Signal RPC error on batch {idx + 1}/{n_batches}: {err}")
            server_retry_after = rl._extract_retry_after_seconds(err)
            scheduler.feedback(server_retry_after, n)
            retry_after_label = f"{server_retry_after:.0f}s" if server_retry_after else "unknown"
            if attempt >= max_attempts:
                logger.error("Signal: rate-limit retries exhausted on batch %d/%d (%d attachments lost, "
                             "server retry_after=%s)", idx + 1, n_batches, n, retry_after_label)
                return False
            logger.warning("Signal: rate-limited on batch %d/%d (attempt %d/%d, server retry_after=%s); "
                           "scheduler will pace the retry",
                           idx + 1, n_batches, attempt, max_attempts, retry_after_label)
        except Exception as e:
            if attempt >= max_attempts:
                logger.error("Signal: send error on batch %d/%d after %d attempts: %s",
                             idx + 1, n_batches, attempt, str(e))
                return False
            logger.warning("Signal: transient error on batch %d/%d (attempt %d/%d): %s; will retry",
                           idx + 1, n_batches, attempt, max_attempts, str(e))


async def _send_signal(extra, chat_id, message, media_files=None):
    """signal-cli JSON-RPC send; owns Signal message chunking (like Telegram).

    Convert Markdown once, then split with ``SignalAdapter._split_signal_formatted_message``.
    Attachments ride only the **last** text chunk, in SIGNAL_MAX_ATTACHMENTS_PER_MSG batches
    metered by the process-wide SignalAttachmentScheduler (shared with the gateway adapter).
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}
    from .adapter import MAX_MESSAGE_LENGTH, SignalAdapter
    try:
        http_url, account = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/"), extra.get("account", "")
        if not account:
            return {"error": "Signal account not configured"}
        valid_media = media_files or []
        attachment_paths = []
        for media_path, _is_voice in valid_media:
            if os.path.exists(media_path):
                attachment_paths.append(media_path)
            else:
                logger.warning("Signal media file not found, skipping: %s", media_path)
        # No attachments still means one (text-only) batch; text rides on batch #0 of the last chunk.
        per_batch = rl.SIGNAL_MAX_ATTACHMENTS_PER_MSG
        att_batches = [attachment_paths[i:i + per_batch] for i in range(0, len(attachment_paths), per_batch)] or [[]]
        n_batches = len(att_batches)
        text_chunks = SignalAdapter._split_signal_formatted_message(
            *markdown_to_signal(message), MAX_MESSAGE_LENGTH)
        recipient = {"groupId": chat_id[6:]} if chat_id.startswith("group:") else {"recipient": [chat_id]}

        async def _rpc_send(text, *, id_prefix, timeout, attachments=None, styles=None):
            params = {"account": account, "message": text, **recipient}
            if text and styles:
                params["textStyle" if len(styles) == 1 else "textStyles"] = (
                    styles[0] if len(styles) == 1 else styles)
            if attachments:
                params["attachments"] = attachments
            payload = {"jsonrpc": "2.0", "method": "send", "params": params,
                       "id": f"{id_prefix}_{int(time.time() * 1000)}"}
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(f"{http_url}/api/v1/rpc", json=payload)

        scheduler = rl.get_scheduler()
        logger.info("send_message Signal: scheduler state=%s, %d attachment(s) in %d batch(es), %d text chunk(s)",
                    scheduler.state(), len(attachment_paths), n_batches, len(text_chunks))

        # Non-final text chunks: text-only (attachments only on the last text chunk).
        for chunk_idx, (plain_text, text_styles) in enumerate(text_chunks[:-1]):
            async def _post_text(_atts, batch_message, _styles=text_styles):
                resp = await _rpc_send(batch_message, id_prefix="send", styles=_styles,
                                       timeout=rl._signal_send_timeout(0))
                resp.raise_for_status()
                return resp.json()
            outcome = await _signal_send_batch(
                _post_text, scheduler, chunk_idx, len(text_chunks), [], plain_text)
            if outcome is False:
                return _error(
                    f"Signal: text chunk {chunk_idx + 1}/{len(text_chunks)} hit rate limit")
            if outcome is not None:
                return outcome

        last_plain, last_styles = text_chunks[-1]

        async def _post(batch_attachments, batch_message):
            resp = await _rpc_send(
                batch_message, id_prefix="send", attachments=batch_attachments,
                styles=last_styles if batch_message else None,
                timeout=rl._signal_send_timeout(len(batch_attachments)))
            resp.raise_for_status()
            return resp.json()

        failed_batches: list[int] = []
        for idx, att_batch in enumerate(att_batches):
            n = len(att_batch)
            if n > 0 and (estimated := scheduler.estimate_wait(n)) >= rl.SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                # Best-effort one-shot RPC for a user-facing pacing notice.
                try:
                    await _rpc_send(f"(More images coming — pausing ~{rl._format_wait(estimated)} "
                                    f"for Signal rate limit, batch {idx + 1}/{n_batches}.)",
                                    id_prefix="notice", timeout=30.0)
                except Exception as _e:
                    logger.warning("Signal: inline notice failed: %s", _e)
            outcome = await _signal_send_batch(_post, scheduler, idx, n_batches, att_batch,
                                               last_plain if idx == 0 else "")
            if outcome is False:
                failed_batches.append(idx + 1)
            elif outcome is not None:
                return outcome
        warnings = []
        if len(attachment_paths) < len(valid_media):
            warnings.append("Some media files were skipped (not found on disk)")
        if failed_batches:
            warnings.append(f"Signal rate-limited {len(failed_batches)} batch(es) "
                            f"(#{', #'.join(str(b) for b in failed_batches)})")
        if failed_batches and len(failed_batches) == n_batches:
            return _error(f"Signal: every batch ({n_batches}) hit rate limit; no attachments delivered")
        # Result-safe chat identifier for tool transcripts/log consumers.
        return _success("signal", "group:***" if str(chat_id).startswith("group:") else chat_id, warnings)
    except Exception as e:
        return _error(f"Signal send failed: {e}")
