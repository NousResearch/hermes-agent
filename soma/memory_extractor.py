"""Post-turn memory extraction.

After each Hermes turn, Soma feeds the user message + the assistant
response into a second LLM call asking "what here is durable enough
to remember?". The model in the foreground never knows memories
exist — extraction is Soma's responsibility, not the agent's.

The extractor returns the list of MemoryRecords that were actually
persisted (after the store's embedding-fuse drops near-duplicates).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .memory_store import IMMORTAL_TAGS, MEMORY_TYPES, MemoryRecord, MemoryStore

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM = """You extract durable memories from a single conversation turn.

You receive a USER message and the ASSISTANT response that followed it.
Return ONLY a JSON array (no prose, no markdown fence) of memory objects.
Empty array `[]` is valid and expected when nothing in the turn is durable.

Each memory object has:
  "type":    one of "semantic" | "procedural" | "episodic"
  "content": one factual sentence in the language the user wrote in
  "tags":    array of short lowercase tags

Use tags from this set when they apply (they make memories immortal):
  domain      — the user's field / project / company
  role        — the user's job, position, expertise
  identity    — names, locations, demographics
  preference  — how the user wants to be addressed or answered
  behavior    — rules the agent must follow next time

Other tags are free-form (e.g. "deadline", "stack", "pet").

Rules:
- Extract ONLY information that will still be true / useful next week.
- Skip transient details: greetings, weather, jokes, momentary moods.
- Skip anything the assistant invented; only what the USER stated.
- Skip if you're not sure. Empty array is better than noise.
- Each memory MUST stand alone — no pronouns referring to the turn.
- "semantic" = facts about the user or world. "procedural" = how to
  do something. "episodic" = events that happened at a specific time.

Return the JSON array and nothing else."""


@dataclass
class ExtractedMemory:
    """A validated, ready-to-store memory candidate."""
    type: str
    content: str
    tags: List[str]


class MemoryExtractor:
    def __init__(
        self,
        store: MemoryStore,
        *,
        call_llm=None,
        model: Optional[str] = None,
        max_per_turn: int = 8,
        max_content_chars: int = 400,
        source: str = "extractor",
    ):
        self.store = store
        self.model = model
        self.max_per_turn = max_per_turn
        self.max_content_chars = max_content_chars
        self.source = source
        if call_llm is None:
            from agent.auxiliary_client import call_llm as default_call_llm
            call_llm = default_call_llm
        self._call_llm = call_llm

    # -- Public API ----------------------------------------------------------

    def extract(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text and not assistant_text:
            return []

        try:
            raw = self._invoke_llm(user_text, assistant_text)
        except Exception as exc:
            logger.warning("soma: extractor LLM call failed: %s", exc)
            return []

        candidates = self._parse_and_validate(raw)
        if not candidates:
            return []

        written: List[MemoryRecord] = []
        for cand in candidates[: self.max_per_turn]:
            try:
                rec = self.store.write(
                    cand.content,
                    type=cand.type,
                    tags=cand.tags,
                    source=self.source,
                    session_id=session_id,
                )
                written.append(rec)
            except Exception as exc:
                logger.warning("soma: failed to write memory %r: %s", cand.content[:60], exc)
        return written

    async def extract_async(
        self,
        user_text: str,
        assistant_text: str,
        *,
        session_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        return await asyncio.to_thread(
            self.extract, user_text, assistant_text, session_id=session_id
        )

    # -- Internals -----------------------------------------------------------

    def _invoke_llm(self, user_text: str, assistant_text: str) -> str:
        payload = (
            f"USER:\n{user_text}\n\n"
            f"ASSISTANT:\n{assistant_text}\n\n"
            f"Return the JSON array now."
        )
        kwargs: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": payload},
            ],
            "temperature": 0.0,
            "max_tokens": 800,
        }
        if self.model:
            kwargs["model"] = self.model
        response = self._call_llm(**kwargs)
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    def _parse_and_validate(self, raw: str) -> List[ExtractedMemory]:
        data = _coerce_json_array(raw)
        if not isinstance(data, list):
            return []
        out: List[ExtractedMemory] = []
        for item in data:
            cand = self._validate_item(item)
            if cand is not None:
                out.append(cand)
        return out

    def _validate_item(self, item: Any) -> Optional[ExtractedMemory]:
        if not isinstance(item, dict):
            return None
        type_ = str(item.get("type") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        raw_tags = item.get("tags") or []
        if type_ not in MEMORY_TYPES:
            return None
        if not content:
            return None
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars].rstrip()
        tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
        # Cap tag count; preserve order, drop dupes.
        seen: set = set()
        unique_tags: List[str] = []
        for t in tags:
            if t in seen:
                continue
            seen.add(t)
            unique_tags.append(t)
            if len(unique_tags) >= 8:
                break
        return ExtractedMemory(type=type_, content=content, tags=unique_tags)


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _coerce_json_array(raw: str) -> Any:
    """Forgivingly extract a JSON array from model output.

    Models sometimes wrap output in ```json ... ``` fences or prepend
    a sentence. We greedy-match the first [ ... ] block.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# Re-export for callers that want to inspect the immortal tag set.
__all__ = ["MemoryExtractor", "ExtractedMemory", "EXTRACTION_SYSTEM", "IMMORTAL_TAGS"]
