"""Custom context engine — enhances the default compressor with:
  1. Proactive context warnings (75% threshold)
  2. Lessons-learned extraction before compression
  3. "Things not to redo" persistence across sessions

Wraps the default ContextCompressor; all compression logic is delegated.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LESSONS_FILE = Path.home() / ".hermes" / "context_lessons.jsonl"
_PROACTIVE_WARN_RATIO = 0.75


def register(ctx):
    """Plugin entry point — called by the plugin loader."""
    from agent.context_compressor import ContextCompressor

    # Read model from config; fallback to a safe default
    try:
        from hermes_cli.config import get_config
        cfg = get_config()
        model = cfg.get("model", "gpt-4o-mini")
    except Exception:
        model = "gpt-4o-mini"

    base = ContextCompressor(model=model)
    engine = CustomContextEngine(base=base)
    ctx.register_context_engine(engine)


class CustomContextEngine:
    """Wraps ContextCompressor with enhanced behavior.

    Delegates all ABC methods to the base compressor.
    Adds proactive warnings and lessons-learned persistence.
    """

    def __init__(self, base):
        self._base = base
        self._warned_this_session = False
        self._lessons_cache: List[str] = []
        self._load_lessons()

    # ── Identity ─────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "custom"

    # ── Token state (read by run_agent.py) ───────────────────

    @property
    def last_prompt_tokens(self):
        return self._base.last_prompt_tokens

    @last_prompt_tokens.setter
    def last_prompt_tokens(self, v):
        self._base.last_prompt_tokens = v

    @property
    def last_completion_tokens(self):
        return self._base.last_completion_tokens

    @last_completion_tokens.setter
    def last_completion_tokens(self, v):
        self._base.last_completion_tokens = v

    @property
    def last_total_tokens(self):
        return self._base.last_total_tokens

    @last_total_tokens.setter
    def last_total_tokens(self, v):
        self._base.last_total_tokens = v

    @property
    def threshold_tokens(self):
        return self._base.threshold_tokens

    @threshold_tokens.setter
    def threshold_tokens(self, v):
        self._base.threshold_tokens = v

    @property
    def context_length(self):
        return self._base.context_length

    @context_length.setter
    def context_length(self, v):
        self._base.context_length = v

    @property
    def compression_count(self):
        return self._base.compression_count

    @compression_count.setter
    def compression_count(self, v):
        self._base.compression_count = v

    @property
    def threshold_percent(self):
        return self._base.threshold_percent

    @threshold_percent.setter
    def threshold_percent(self, v):
        self._base.threshold_percent = v

    @property
    def protect_first_n(self):
        return self._base.protect_first_n

    @property
    def protect_last_n(self):
        return self._base.protect_last_n

    @property
    def emit_automatic_compaction_status(self):
        return self._base.emit_automatic_compaction_status

    # ── Core interface ───────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self._base.update_from_response(usage)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return self._base.should_compress(prompt_tokens)

    def should_compress_info(self, prompt_tokens: int = None):
        return self._base.should_compress_info(prompt_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]] = None) -> bool:
        return self._base.should_compress_preflight(messages)

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Compress with lessons-learned extraction."""
        # Extract lessons before compression destroys the context
        new_lessons = self._extract_lessons(messages)
        if new_lessons:
            self._lessons_cache.extend(new_lessons)
            self._save_lessons()
            logger.info("Extracted %d lessons before compression", len(new_lessons))

        # Delegate to base compressor
        result = self._base.compress(
            messages,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )

        # Inject persisted lessons into compressed result
        if self._lessons_cache and result:
            self._inject_lessons(result)

        return result

    # ── Lifecycle ────────────────────────────────────────────

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._warned_this_session = False
        self._load_lessons()
        if hasattr(self._base, "on_session_start"):
            self._base.on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Extract lessons from final messages
        final_lessons = self._extract_lessons(messages)
        if final_lessons:
            self._lessons_cache.extend(final_lessons)
            self._save_lessons()
        if hasattr(self._base, "on_session_end"):
            self._base.on_session_end(session_id, messages)

    def on_session_reset(self) -> None:
        self._warned_this_session = False
        if hasattr(self._base, "on_session_reset"):
            self._base.on_session_reset()

    # ── Lessons extraction ───────────────────────────────────

    def _extract_lessons(self, messages) -> List[str]:
        """Extract 'AVOID:' patterns from messages."""
        lessons = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("- AVOID:") or stripped.startswith("AVOID:"):
                    lessons.append(stripped)
                elif "REASON:" in stripped and lessons and "AVOID" not in stripped:
                    lessons[-1] += " " + stripped
        return lessons

    def _inject_lessons(self, result: List[Dict[str, Any]]):
        """Inject persisted lessons into compressed summary."""
        if not self._lessons_cache:
            return
        lessons_text = "\n".join(f"  {l}" for l in self._lessons_cache[-20:])
        injection = f"\n\n## Persisted Lessons (cross-session)\n{lessons_text}\n"

        for msg in result:
            if msg.get("role") in ("system", "assistant") and isinstance(msg.get("content"), str):
                content = msg["content"]
                if "compacted" in content.lower() or "summary" in content.lower():
                    msg["content"] += injection
                    break

    # ── Disk persistence ─────────────────────────────────────

    def _load_lessons(self):
        self._lessons_cache = []
        if not _LESSONS_FILE.exists():
            return
        try:
            for line in _LESSONS_FILE.read_text().strip().split("\n"):
                if line.strip():
                    entry = json.loads(line)
                    lesson = entry.get("lesson", "")
                    if lesson:
                        self._lessons_cache.append(lesson)
        except Exception as e:
            logger.debug("Failed to load lessons: %s", e)

    def _save_lessons(self):
        if not self._lessons_cache:
            return
        try:
            _LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            seen = set()
            unique = []
            for l in self._lessons_cache:
                if l not in seen:
                    seen.add(l)
                    unique.append(l)
            with open(_LESSONS_FILE, "w") as f:
                for lesson in unique[-50:]:
                    f.write(json.dumps({"lesson": lesson, "ts": time.time()}) + "\n")
            self._lessons_cache = unique[-50:]
        except Exception as e:
            logger.debug("Failed to save lessons: %s", e)
