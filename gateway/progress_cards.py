"""Bounded, opt-in per-turn progress; platform adapters own card rendering.

Only public tool lifecycle and assistant commentary enter this surface. Never
feed reasoning deltas, tool results, full arguments, or approval prompts here.
"""
from collections import OrderedDict
import logging
import re

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


def safe_progress_text(value, limit=600):
    """Redact before truncating, including signed URL credentials and markup."""
    text = redact_sensitive_text(str(value or ""), force=True, redact_url_credentials=True)
    # Preview URLs are not navigation targets. Opaque tickets/signatures can
    # have arbitrary names; do not retain any URL query or fragment here.
    text = re.sub(r"(https?://[^\s?#]+)[?#][^\s]*", r"\1?[redacted]", text)
    # Escape card mentions/tags/links so telemetry cannot ping users or embed UI.
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"([\\`*\[\]])", r"\\\1", text)
    return text if len(text) <= limit else text[:limit - 1] + "…"


class TurnProgress:
    """One bounded ledger and one message identity, owned by a single turn."""
    MAX_DETAILS = 24

    def __init__(self, adapter, chat_id, *, session_id, reply_to=None, metadata=None):
        self.adapter = adapter
        self.chat_id = chat_id
        self.session_id = safe_progress_text(session_id, 100)
        self.reply_to = reply_to
        self.metadata = metadata
        self.entries = OrderedDict()
        self.omitted = 0
        self.sequence = 0
        self.status = "running"
        self.message_id = None
        self.failed = False
        self.finished = False
        self.had_tool_error = False
        self.pending_confirmation = False
        self.last_publish = float("-inf")
        self.published_sequence = -1

    async def publish(self, *, force=False):
        import asyncio
        import time
        if self.failed or (not force and self.finished):
            return
        if not force and (self.sequence == self.published_sequence or time.monotonic() - self.last_publish < 1.5):
            return
        self.last_publish = time.monotonic()
        self.published_sequence = self.sequence
        try:
            result = await asyncio.wait_for(self.adapter.send_progress_card(
                self.chat_id, self.snapshot(), message_id=self.message_id,
                reply_to=self.reply_to, metadata=self.metadata,
            ), timeout=5.0)
            if not result.success or (not self.message_id and not result.message_id):
                raise RuntimeError("Progress card unavailable")
            self.message_id = self.message_id or result.message_id
        except Exception:
            # Latch BEFORE attempting fallback, including on timeout or an
            # ambiguous successful create. Never fall through to text progress.
            self.failed = True
            logger.warning("Progress card unavailable session=%s", self.session_id)
            try:
                await asyncio.wait_for(self.adapter.send(
                    chat_id=self.chat_id,
                    content=("进度卡更新暂时失败，已有卡片可能停留在旧状态；任务仍在继续。"
                             "确认请求与最终结果仍会正常发送。需要执行记录可直接问我。"
                             f"（Session {self.session_id}；本地可用 hermes sessions browse）"),
                    reply_to=self.reply_to, metadata=self.metadata,
                ), timeout=3.0)
            except Exception:
                logger.warning("Progress fallback unavailable session=%s", self.session_id)

    async def finish(self, outcome):
        if self.finished:
            return
        self.finished = True
        self.status = "completed_with_warnings" if outcome == "completed" and self.had_tool_error else outcome
        for entry in self.entries.values():
            if entry["status"] == "running":
                entry["status"] = "interrupted" if outcome == "interrupted" else "unconfirmed"
        if not self.failed:
            await self.publish(force=True)
        if self.failed:
            # One last PATCH can recover a transient error. Never retry a
            # CREATE with unknown delivery or send another fallback bubble.
            if self.message_id:
                import asyncio
                try:
                    await asyncio.wait_for(self.adapter.send_progress_card(
                        self.chat_id, self.snapshot(), message_id=self.message_id,
                        reply_to=self.reply_to, metadata=self.metadata,
                    ), timeout=5.0)
                except Exception:
                    logger.warning("Final card recovery unavailable session=%s", self.session_id)
            return

    def record(self, event):
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind not in {"tool.started", "tool.completed", "commentary"}:
            return
        name = str(event.get("tool_name") or "tool")
        if name == "_thinking":
            return
        if name == "clarify":
            # Keep the actual question/answer on its existing confirmation UI.
            if kind in {"tool.started", "tool.completed"}:
                self.pending_confirmation = kind == "tool.started"
                self.status = "awaiting_confirmation" if self.pending_confirmation else "running"
                self.sequence += 1
                self.last_publish = float("-inf")
            return
        key = ("tool", str(event.get("tool_call_id") or self.sequence + 1))
        if kind == "tool.completed" and key not in self.entries:
            return  # evicted/uncorrelated starts must not become phantom rows
        self.sequence += 1
        if kind == "commentary":
            key = ("commentary", self.sequence)
            entry = {"text": safe_progress_text(event.get("text"), 570), "status": ""}
        elif kind == "tool.started":
            from agent.display import get_tool_emoji
            label = event.get("label") or f"{get_tool_emoji(name)} {name}: {event.get('preview') or ''}"
            entry = {"text": safe_progress_text(label, 570), "status": "running"}
        else:
            entry = self.entries[key]
            entry["status"] = "failed" if event.get("is_error") else "completed"
            self.had_tool_error |= bool(event.get("is_error"))
        self.entries[key] = entry
        if len(self.entries) > self.MAX_DETAILS:
            self.entries.popitem(last=False)
            self.omitted += 1
        # Existing profile-aware logging remains the inspectable full event
        # sequence even when the bounded card drops older entries.
        logger.info("Progress session=%s %s %s", self.session_id, entry["text"], entry["status"])

    def snapshot(self):
        details = [e["text"] + (f" — {e['status']}" if e["status"] else "") for e in self.entries.values()]
        omitted = self.omitted
        if self.had_tool_error:
            omitted += max(0, len(details) - (self.MAX_DETAILS - 1))
            details = ["本轮有工具异常记录，可能已通过后续步骤恢复；实际交付结果以最终答复为准。"] + details[-(self.MAX_DETAILS - 1):]
        return {
            "status": self.status,
            "session_id": self.session_id,
            "details": details,
            "omitted": omitted,
        }


async def run_progress_card(ctx, adapter):
    """Drain the existing thread-safe queue into one turn-owned card."""
    import asyncio
    import queue
    progress = TurnProgress(
        adapter, ctx.source.chat_id, session_id=ctx.session_id,
        reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata,
    )
    try:
        while True:
            for _ in range(1000):
                try:
                    progress.record(ctx.progress_queue.get_nowait())
                except queue.Empty:
                    break
            current = ctx._run_still_current()
            outcome = ctx._progress_outcome
            agent = ctx.agent_holder[0] if ctx.agent_holder else None
            if not current or (agent and getattr(agent, "is_interrupted", False)):
                outcome = "interrupted"
            if outcome:
                # A superseded turn may resolve its existing card, never
                # create a new stale bubble after /new or /stop.
                if current or progress.message_id:
                    if progress.entries or progress.message_id:
                        await progress.finish(outcome)
                return
            if progress.entries or progress.message_id or progress.pending_confirmation:
                await progress.publish()
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # No late create/retry on cancellation; the normal finish helper
        # communicates terminal state before cancelling this task.
        raise


async def finish_progress_card(ctx, task, *, cancelled=False, result=None):
    """Resolve the card without allowing UI errors to swallow final delivery."""
    import asyncio
    if ctx._progress_outcome is None:
        result = result if isinstance(result, dict) else (ctx.result_holder[0] or {})
        agent = ctx.agent_holder[0] if ctx.agent_holder else None
        if cancelled or result.get("interrupted") or (agent and getattr(agent, "is_interrupted", False)):
            ctx._progress_outcome = "interrupted"
        elif not result or result.get("failed") or result.get("error") or result.get("completed") is False:
            ctx._progress_outcome = "failed"
        else:
            ctx._progress_outcome = "completed"
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=12.0)
    except asyncio.TimeoutError:
        task.cancel()
    except asyncio.CancelledError:
        task.cancel()
        raise  # A new /stop during cleanup must still cancel the outer turn.
    except Exception:
        logger.warning("Progress card task failed session=%s", ctx.session_id)
    finally:
        if not task.done():
            task.cancel()
