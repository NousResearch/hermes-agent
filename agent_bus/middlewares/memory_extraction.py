"""MemoryExtractionMiddleware — async fact extraction from conversation.

Inspired by DeerFlow's MemoryMiddleware (§1.B #13 + §1.E). Unlike DeerFlow,
this middleware does NOT replace Hermes's wiki/memory/ manual authoring; it
AUGMENTS it by dedicated auto-extracted facts stored in a separate JSON
file so the two channels don't collide.

Flow
----
1. `after_model`: filter to (user messages + final AI responses), enqueue
   for the current thread_id.
2. Background debounce: 30s default. When idle long enough, batch messages
   and call `_extract_facts()` (pluggable — defaults to simple heuristic;
   real LLM extractor is off by default and opt-in via env var).
3. Atomic write to `~/.hermes/memory-auto.json` with dedup
   (whitespace-normalized fact content comparison).
4. Injection (future): `before_model` can later prepend top-N facts to the
   system prompt `<memory>` tag. For MVP we only capture; injection is a
   downstream step once fact quality is validated.

Env vars
--------
HERMES_MW_MEMORY_EXTRACT     off | core (default core) — master gate
HERMES_AUTO_MEMORY_DEBOUNCE  seconds, default 30
HERMES_AUTO_MEMORY_PATH      default ~/.hermes/memory-auto.json
HERMES_AUTO_MEMORY_LLM       on | off (default off for MVP safety)
HERMES_AUTO_MEMORY_MAX_FACTS default 100
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent_bus.middleware import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SEC = 30
DEFAULT_MAX_FACTS = 100
DEFAULT_STORAGE = Path.home() / ".hermes" / "memory-auto.json"


# -------- Fact schema --------
@dataclass
class Fact:
    id: str
    content: str
    category: str
    confidence: float
    created_at: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -------- Queue (per-thread debounce) --------
@dataclass
class _PendingBatch:
    thread_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_enqueued_at: float = 0.0
    timer: threading.Timer | None = None


class _ExtractionQueue:
    def __init__(self, debounce_sec: int, worker: Callable[[str, list[dict]], None]):
        self.debounce_sec = debounce_sec
        self.worker = worker
        self._batches: dict[str, _PendingBatch] = {}
        self._lock = threading.Lock()

    def enqueue(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        with self._lock:
            batch = self._batches.get(thread_id)
            if batch is None:
                batch = _PendingBatch(thread_id=thread_id)
                self._batches[thread_id] = batch
            batch.messages.extend(messages)
            batch.last_enqueued_at = time.time()
            if batch.timer is not None:
                batch.timer.cancel()
            batch.timer = threading.Timer(self.debounce_sec, self._fire, args=(thread_id,))
            batch.timer.daemon = True
            batch.timer.start()

    def _fire(self, thread_id: str) -> None:
        with self._lock:
            batch = self._batches.pop(thread_id, None)
        if not batch or not batch.messages:
            return
        try:
            self.worker(thread_id, batch.messages)
        except Exception as exc:  # pragma: no cover — never crash the daemon thread
            logger.warning("memory-extract worker failed for %s: %s", thread_id, exc)

    def flush_now(self, thread_id: str | None = None) -> None:
        """Force processing of pending batches (for tests or shutdown)."""
        with self._lock:
            if thread_id:
                batch = self._batches.pop(thread_id, None)
                to_fire = [batch] if batch else []
            else:
                to_fire = list(self._batches.values())
                self._batches.clear()
        for b in to_fire:
            if b.timer is not None:
                b.timer.cancel()
            if b.messages:
                try:
                    self.worker(b.thread_id, b.messages)
                except Exception as exc:
                    logger.warning("flush_now worker failed for %s: %s", b.thread_id, exc)


# -------- Storage --------
def _load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"facts": [], "updated_at": 0.0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"facts": [], "updated_at": 0.0}


def _save_store(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _normalize_for_dedup(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip().lower()


# -------- Extractors --------
def heuristic_extract(messages: list[dict[str, Any]]) -> list[Fact]:
    """Simple heuristic fallback when LLM extractor is off.

    Grabs user messages that look like stated preferences / facts (contain
    "我", "I ", "preference", etc.) — not great quality but safe and offline.
    """
    facts: list[Fact] = []
    now = time.time()
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").strip()
        if not content or len(content) > 300:
            continue
        # Filter: must contain a statement-like word
        if not re.search(
            r"(我|I\s|prefer|like|dislike|每天|always|never|要|不要)",
            content, re.IGNORECASE,
        ):
            continue
        facts.append(Fact(
            id=f"h-{int(now)}-{i}",
            content=content[:200],
            category="preference",
            confidence=0.5,  # low because heuristic
            created_at=now,
            source="heuristic",
        ))
    return facts


# -------- Middleware --------
class MemoryExtractionMiddleware(BaseMiddleware):
    """Queue conversation snippets, debounce, extract facts, dedup, persist."""

    name = "memory-extraction"

    def __init__(self) -> None:
        debounce = int(os.environ.get("HERMES_AUTO_MEMORY_DEBOUNCE", str(DEFAULT_DEBOUNCE_SEC)))
        self._queue = _ExtractionQueue(
            debounce_sec=max(1, debounce),
            worker=self._process_batch,
        )

    # -------- Hooks --------
    def after_model(self, ctx: MiddlewareContext) -> MiddlewareContext:
        if not ctx.messages or not ctx.thread_id:
            return ctx

        # DeerFlow filter: user messages + final AI responses (strip tool churn)
        relevant = [
            m for m in ctx.messages
            if m.get("role") == "user"
            or (m.get("role") == "assistant" and not m.get("tool_calls"))
        ]
        if not relevant:
            return ctx

        self._queue.enqueue(ctx.thread_id, relevant)
        ctx.record(self.name, "after_model", "enqueued", f"count={len(relevant)}")
        return ctx

    def on_session_end(self, ctx: MiddlewareContext) -> MiddlewareContext:
        # Force flush for tests / graceful shutdown
        if ctx.thread_id:
            self._queue.flush_now(ctx.thread_id)
            ctx.record(self.name, "on_session_end", "flushed")
        return ctx

    # -------- Worker --------
    def _process_batch(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        use_llm = os.environ.get("HERMES_AUTO_MEMORY_LLM", "off").lower() == "on"
        if use_llm:
            try:
                facts = self._llm_extract(thread_id, messages)
            except Exception as exc:
                logger.warning("LLM extract failed, falling back to heuristic: %s", exc)
                facts = heuristic_extract(messages)
        else:
            facts = heuristic_extract(messages)

        if not facts:
            return

        path = Path(os.environ.get("HERMES_AUTO_MEMORY_PATH", str(DEFAULT_STORAGE))).expanduser()
        store = _load_store(path)
        existing = store.get("facts", []) or []
        existing_dedup_keys = {_normalize_for_dedup(f.get("content", "")) for f in existing}

        added = 0
        for f in facts:
            key = _normalize_for_dedup(f.content)
            if key in existing_dedup_keys:
                continue
            existing.append(f.to_dict())
            existing_dedup_keys.add(key)
            added += 1

        # Cap facts to max
        max_facts = int(os.environ.get("HERMES_AUTO_MEMORY_MAX_FACTS", str(DEFAULT_MAX_FACTS)))
        if len(existing) > max_facts:
            existing = sorted(existing, key=lambda f: f.get("created_at", 0), reverse=True)[:max_facts]

        store["facts"] = existing
        store["updated_at"] = time.time()
        _save_store(path, store)
        logger.info("memory-extract: thread=%s added=%d total=%d", thread_id, added, len(existing))

    def _llm_extract(self, thread_id: str, messages: list[dict[str, Any]]) -> list[Fact]:
        """Extract facts via Anthropic SDK if available, else return [].

        Uses minimal prompt that asks Claude to emit a JSON array of facts.
        Degrades silently (log-only) on any error — caller falls back to
        heuristic extractor.
        """
        try:
            import anthropic  # type: ignore
        except ImportError:
            logger.debug("anthropic SDK not installed — LLM extract skipped")
            return []

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("ANTHROPIC_API_KEY missing — LLM extract skipped")
            return []

        # Build a compact conversation text
        lines: list[str] = []
        for m in messages[-30:]:  # last 30 only, avoid blowing context
            role = m.get("role", "?")
            content = m.get("content") or ""
            if isinstance(content, str) and content.strip():
                lines.append(f"{role}: {content.strip()[:500]}")
        if not lines:
            return []

        model = os.environ.get("HERMES_AUTO_MEMORY_MODEL", "claude-3-5-haiku-latest")
        prompt = (
            "You are extracting durable facts about the user from this "
            "conversation. Return ONLY a JSON array (no prose) of at most 5 "
            "objects, each with keys:\n"
            '  content (string, <160 chars, third-person statement of fact)\n'
            '  category (one of: preference, knowledge, context, behavior, goal)\n'
            '  confidence (float 0..1)\n'
            "Skip anything speculative. Prefer stable traits over momentary "
            "mentions.\n\n"
            "Conversation:\n---\n" + "\n".join(lines) + "\n---"
        )

        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text if resp.content else ""
        except Exception as exc:
            logger.warning("LLM extract call failed: %s", exc)
            return []

        import json as _json
        import re as _re

        # Extract JSON array from response (model may wrap in fences)
        m = _re.search(r"\[.*\]", raw, _re.DOTALL)
        if not m:
            logger.debug("LLM extract: no JSON array in response")
            return []
        try:
            raw_facts = _json.loads(m.group(0))
        except _json.JSONDecodeError as exc:
            logger.warning("LLM extract: JSON parse failed: %s", exc)
            return []

        now = time.time()
        out: list[Fact] = []
        for i, f in enumerate(raw_facts or []):
            if not isinstance(f, dict):
                continue
            content = (f.get("content") or "").strip()
            if not content or len(content) > 200:
                continue
            category = f.get("category", "context")
            if category not in ("preference", "knowledge", "context", "behavior", "goal"):
                category = "context"
            try:
                confidence = float(f.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7
            confidence = max(0.0, min(1.0, confidence))
            out.append(Fact(
                id=f"llm-{int(now)}-{i}",
                content=content,
                category=category,
                confidence=confidence,
                created_at=now,
                source=f"llm:{model}",
            ))
        logger.info("LLM extract: %d facts from %s", len(out), thread_id)
        return out

    # For tests
    def _flush_all(self) -> None:
        self._queue.flush_now()
