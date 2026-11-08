"""Native plan snapshots for a single owned Slack task-card stream.

The stream remains the message: expiry changes its write method, not its identity.
Snapshots replace details; stream chunks append details. Only acknowledged stream
chunks seed a snapshot, while queued edit-mode chunks are applied exactly once.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import time


def slack_error_code(error):
    response = getattr(error, "response", None)
    if response is not None:
        return response.get("error")
    # The guarded client converts explicit ok:false responses into exact codes.
    return str(error) if type(error) is RuntimeError else None


def retry_delay(error):
    from agent.retry_utils import parse_retry_after_seconds
    value = parse_retry_after_seconds(getattr(getattr(error, "response", None), "headers", None))
    return max(3.0, value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else 3.0


def rich_text(text):
    """Preserve complete inline spans and literal newlines in native task details."""
    from plugins.platforms.slack.block_kit import _inline_elements
    return {"type": "rich_text", "elements": [
        {"type": "rich_text_section", "elements": _inline_elements(text) or [{"type": "text", "text": " "}]},
    ]}


class NativePlan:
    MAX_TASKS = 50
    MAX_CHARS = 40_000
    INTERVAL = 3.0

    def __init__(self, owner):
        self.owner = owner
        self.title = owner.header_label or "Working"
        self.tasks = {}
        self.editing = False
        self.dirty = False
        self.queue = []
        self.next_edit = 0.0
        self.retry_at = 0.0
        self.revision = 0
        self.closed_stream = False

    def live(self):
        checker = getattr(self.owner.client, "is_live", None)
        return not self.owner.disabled and (not callable(checker) or checker())

    def apply(self, chunk):
        if chunk["type"] == "plan_update":
            self.title = chunk["title"]
            return
        if chunk["type"] != "task_update":
            return
        task_id = str(chunk["id"])
        row = self.tasks.setdefault(task_id, {"task_id": task_id, "title": "", "status": "pending"})
        for key in ("title", "status", "output", "sources"):
            if key in chunk:
                row[key] = deepcopy(chunk[key])
        if chunk.get("details"):
            row["details"] = row.get("details", "") + chunk["details"]

    def blocks(self, *, continuing=False):
        tasks = []
        for row in self.tasks.values():
            item = deepcopy(row)
            for key in ("details", "output"):
                if item.get(key):
                    item[key] = rich_text(item[key])
            if continuing and item["status"] == "in_progress":
                item["status"] = "pending"
                item["title"] += " · continued below"
            tasks.append(item)
        title = self.title + (" · continued below" if continuing else "")
        return [{"type": "plan", "title": title[:250], "block_id": f"plan-{self.revision}", "tasks": tasks}]

    def fits(self, chunk):
        if chunk["type"] != "task_update":
            return True
        if chunk["id"] not in self.tasks and len(self.tasks) >= self.MAX_TASKS:
            return False
        size = sum(len(str(row.get(key, ""))) for row in self.tasks.values()
                   for key in ("title", "details", "output", "sources"))
        return not self.tasks or size + len(json.dumps(chunk, ensure_ascii=False)) <= self.MAX_CHARS

    async def activate(self, *, expired=False):
        if self.editing:
            return
        if not expired:
            try:
                await self.owner.client.chat_stopStream(channel=self.owner.channel, ts=self.owner.ts)
            except Exception as error:
                if slack_error_code(error) != "message_not_in_streaming_state":
                    raise
        self.closed_stream = True
        if self.owner._stopped or not self.live():
            return
        self.editing = True
        self.dirty = True

    async def offer(self, chunk):
        self.queue.append(deepcopy(chunk))
        await self.flush()

    async def flush(self, *, final=False):
        """Called under the owner's send lock; never sleeps or outlives its turn."""
        if not self.editing or not self.live() or (self.owner._stopped and not final):
            return
        now = time.monotonic()
        if now < self.retry_at or (not final and now < self.next_edit):
            return
        while self.queue and self.fits(self.queue[0]):
            self.apply(self.queue.pop(0))
            self.dirty = True
        if not self.dirty and not self.queue:
            return
        continuing = bool(self.queue)
        self.revision += 1
        try:
            await self.owner.client.chat_update(
                channel=self.owner.channel, ts=self.owner.ts,
                text=self.title, blocks=self.blocks(continuing=continuing),
            )
        except Exception as error:
            from aiohttp import ClientError
            if (slack_error_code(error) in {"ratelimited", "rate_limited"}
                    or isinstance(error, (OSError, TimeoutError, ClientError))):
                self.retry_at = time.monotonic() + retry_delay(error)
                # The guarded rail records errors for its publisher. This known
                # idempotent retry remains owned here, not by a text fallback.
                if hasattr(self.owner.client, "failure"):
                    self.owner.client.failure = None
                return
            raise
        self.dirty = False
        self.retry_at = 0.0
        self.next_edit = time.monotonic() + self.INTERVAL
        if continuing and not self.owner._stopped and self.live():
            self.owner._rollovers += 1
            if self.owner._rollovers > self.owner.MAX_ROLLOVERS:
                raise RuntimeError("native plan capacity continuation limit exceeded")
            self.closed_stream = False
            await self.owner._open_stream()
            if self.owner._stopped or not self.live():
                return
            await self.owner.client.chat_stopStream(channel=self.owner.channel, ts=self.owner.ts)
            self.tasks = {}
            self.title = f"⤵ {self.owner.header_label or 'Working'}"
            self.next_edit = 0.0
            self.closed_stream = True
            # Populate the new bounded page now; subsequent updates still coalesce.
            await self.flush(final=final)

    def reset_stream(self):
        """An explicit missing-message recovery owns a fresh stream, not an edit."""
        self.tasks = {}
        self.editing = False
        self.dirty = False
        self.queue = []
        self.next_edit = self.retry_at = 0.0
        self.closed_stream = False
