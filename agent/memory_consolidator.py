# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/memory/consolidator.js
#             src/memory/consolidation-loop.js
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""Long-term memory consolidator + round-robin loop.

**Not a memory *store*.** This module is the *janitor* — it cleans up
redundant / stale memories in a store you inject. The store owns
persistence, retrieval, vector indexes, and every actual mutation.
This module just:

* Picks one entity's worth of memories at a time.
* Feeds them to an LLM with a constrained tool set
  (``merge_memories``, ``downgrade_memory``, ``skip_consolidation``).
* Applies each tool call against the injected store.

## Design invariants

* **Never write new content.** The consolidator can only merge, hide
  (soft-delete via ``visibility=0`` + ``merged_into=keep_id``), or
  downgrade salience. It never invents facts.
* **Contradictions are signal.** Two memories that flatly disagree are
  left alone. Merging them would erase evidence.
* **Salience 5 is identity-level.** Salience-5 memories are protected
  from downgrade unless the LLM has overwhelming counter-evidence in
  the same batch.
* **Round-robin fairness.** The loop cycles through eligible entities
  (fact/person with ≥ 3 memories) so no single hot entity monopolises
  the cleanup budget.
* **Hidden ≠ deleted.** ``visibility=0`` memories remain in the FTS
  index + vector store and are recoverable by explicit rehydration.
  The default retrieval flows just stop returning them.

## Relationship to Hermes's ``sleep-consolidation`` skill

This module is the *engine* the skill can call. The skill orchestrates
*when* consolidation runs (idle detection, weekly deep pass, do-not-
disturb windows) and *what strategy* to use (per-user vs global,
priority pruning first vs merges first). The engine keeps its focus
narrow: given a batch of memories, ask the LLM what to do, apply the
verdict. Everything policy-shaped stays in the skill or in the loop
config below.

Ported from BaiLongma's ``consolidator.js`` + ``consolidation-loop.js``
(MIT).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


logger = logging.getLogger(__name__)


# ── Prompt + tool schema ────────────────────────────────────────────


CONSOLIDATOR_SYSTEM_PROMPT = (
    "You are the memory consolidator. Your job is to clean up "
    "redundant or stale long-term memories for ONE entity at a time. "
    "You do not write new memories. You only call tools to merge or "
    "downgrade existing ones.\n\n"
    "## What you're given\n\n"
    "A batch of memories about one entity, each with:\n"
    "- mem_id\n"
    "- type (fact / person / etc.)\n"
    "- title\n"
    "- content\n"
    "- salience (1-5)\n"
    "- timestamp\n\n"
    "## What to do\n\n"
    "Read the batch. Identify:\n\n"
    "1. SEMANTIC DUPLICATES — two or more memories that say the same "
    "thing in different words. Pick the best-phrased one as keep, "
    "merge the rest into it via merge_memories. merged_content should "
    "preserve any unique facts from drops. Drop memories are NOT "
    "deleted: they become hidden (visibility=0, "
    "merged_into=keep_mem_id). The row + FTS index + embedding are "
    "fully preserved and remain reachable by future recovery flows; "
    "routine search/get* simply stops returning them.\n\n"
    "2. SUPERSEDED FACTS — an older memory whose claim is strictly "
    "contained in a newer, more complete one. Merge the older into "
    "the newer.\n\n"
    "3. STALE LOW-VALUE MEMORIES — memories that haven't been "
    "reinforced and seem ephemeral in hindsight. Use "
    "downgrade_memory to lower salience (do NOT delete).\n\n"
    "4. PROTECTED — salience=5 memories represent identity-level "
    "beliefs. Do NOT downgrade or drop them unless there is "
    "overwhelming evidence in this batch they are wrong. When in "
    "doubt, leave them alone.\n\n"
    "## What NOT to do\n\n"
    "- Do not invent new content unsupported by the batch.\n"
    "- Do not merge memories that contradict each other — leave "
    "both; contradiction is signal, not noise.\n"
    "- Do not downgrade everything to clean up \"clutter\" — only "
    "downgrade when a memory has clearly aged out.\n"
    "- If nothing in this batch needs cleanup, call "
    "skip_consolidation. Do not force action.\n\n"
    "## Tool usage\n\n"
    "- merge_memories({ keep_mem_id, drop_mem_ids: [...], "
    "merged_content, merged_salience?, reason })\n"
    "- downgrade_memory({ mem_id, new_salience, reason })\n"
    "- skip_consolidation({ reason })\n\n"
    "You may call multiple merges/downgrades in one session. Always "
    "include reason.\n\n"
    "## Output\n\n"
    "Tool calls only. No prose."
)

CONSOLIDATOR_TOOL_NAMES: tuple[str, ...] = (
    "merge_memories",
    "downgrade_memory",
    "skip_consolidation",
)


# ── Data model ─────────────────────────────────────────────────────


@dataclass
class Memory:
    """A memory row as fed to the LLM — deliberately minimal.

    ``event_type`` mirrors the upstream JS shape (fact / person / …).
    ``salience`` is 1..5 with 5 = identity-level.
    """

    mem_id: str
    event_type: str
    title: str
    content: str
    salience: int = 3
    timestamp: str = ""


@dataclass
class CandidateEntity:
    """One entity eligible for a consolidation pass.

    ``memory_count`` is used only for logging / cursor tie-breaking;
    eligibility gating (usually memory_count ≥ 3) lives in the store.
    """

    entity: str
    memory_count: int = 0


# ── Store + LLM protocols (inject at seam) ─────────────────────────


@runtime_checkable
class MemoryStore(Protocol):
    """Contract the consolidator expects from a memory backend.

    Every mutation returns ``bool`` — the LLM's tool-call handler
    checks this to count successful actions. A ``False`` return means
    "the store refused this specific action" (invalid ids, integrity
    violation) and is *not* treated as a fatal error for the pass.
    """

    def get_candidate_entities(self, limit: int) -> Sequence[CandidateEntity]:
        ...

    def get_memories_by_entity(
        self, entity: str, limit: int
    ) -> Sequence[Memory]:
        ...

    def merge_memories(
        self,
        *,
        keep_mem_id: str,
        drop_mem_ids: Sequence[str],
        merged_content: str,
        merged_salience: Optional[int],
        reason: str,
    ) -> bool:
        ...

    def downgrade_memory(
        self, *, mem_id: str, new_salience: int, reason: str
    ) -> bool:
        ...


ConsolidatorLLM = Callable[..., Awaitable[Any]]
"""Async LLM adapter. Called with keyword args:

    system_prompt, message, temperature, max_tokens, tools,
    on_tool_call

``on_tool_call`` is a synchronous callable ``(name, args) -> str``
returning a JSON tool result string. Adapters must invoke it for each
tool call the model makes.
"""


# ── Consolidator runner ────────────────────────────────────────────


def _format_memory(m: Memory) -> str:
    ts = (m.timestamp or "")[:10]
    return (
        f"mem_id={m.mem_id} | type={m.event_type} | "
        f"salience={m.salience} | {ts}\n"
        f"  title: {m.title}\n"
        f"  content: {m.content}"
    )


def format_input(entity: str, memories: Sequence[Memory]) -> str:
    """Public helper — exposed for tests and skill callers who want
    to preview what the LLM will see without running the pass.
    """
    header = f"[Entity] {entity}\n[Memory count] {len(memories)}\n\n"
    return header + "\n\n".join(_format_memory(m) for m in memories)


@dataclass
class ConsolidationResult:
    actions: int
    skipped: bool
    error: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)


class _RateLimitedError(Exception):
    """Raised by the LLM adapter (or synthesised by the runner) when a
    429 is seen so the caller's ``on_rate_limited`` hook fires.
    """


def _looks_like_rate_limited(err: BaseException) -> bool:
    if isinstance(err, _RateLimitedError):
        return True
    msg = str(err) or ""
    status = getattr(err, "status", None) or getattr(err, "status_code", None)
    return status == 429 or "429" in msg


def _make_tool_handler(
    store: MemoryStore,
    result_ref: "ConsolidationResult",
) -> Callable[[str, dict], str]:
    """Build the tool-call handler that mutates the store and updates
    the outer result counter. Returns a JSON string per invocation so
    the LLM adapter has something to feed back.
    """

    def handler(name: str, args: dict) -> str:
        args = args or {}
        payload: dict[str, Any] = {}
        try:
            if name == "skip_consolidation":
                result_ref.skipped = True
                payload = {"ok": True}
            elif name == "merge_memories":
                keep = str(args.get("keep_mem_id") or "")
                drops = list(args.get("drop_mem_ids") or [])
                merged_content = str(args.get("merged_content") or "")
                merged_salience_raw = args.get("merged_salience")
                merged_salience = (
                    int(merged_salience_raw)
                    if merged_salience_raw is not None
                    else None
                )
                reason = str(args.get("reason") or "")
                if not keep or not drops:
                    payload = {"ok": False, "error": "missing keep/drop ids"}
                else:
                    ok = bool(
                        store.merge_memories(
                            keep_mem_id=keep,
                            drop_mem_ids=[str(d) for d in drops],
                            merged_content=merged_content,
                            merged_salience=merged_salience,
                            reason=reason,
                        )
                    )
                    payload = {"ok": ok}
                    if ok:
                        result_ref.actions += 1
            elif name == "downgrade_memory":
                mem_id = str(args.get("mem_id") or "")
                new_salience_raw = args.get("new_salience")
                if not mem_id or new_salience_raw is None:
                    payload = {"ok": False, "error": "missing mem_id/salience"}
                else:
                    ok = bool(
                        store.downgrade_memory(
                            mem_id=mem_id,
                            new_salience=int(new_salience_raw),
                            reason=str(args.get("reason") or ""),
                        )
                    )
                    payload = {"ok": ok}
                    if ok:
                        result_ref.actions += 1
            else:
                payload = {"ok": False, "error": f"unknown tool {name!r}"}
        except Exception as err:  # noqa: BLE001 — never let a handler
            # bubble; it would abort mid-batch and lose progress on
            # the other tool calls this LLM turn already made.
            logger.warning(
                "[consolidator] tool %s raised %s (args=%r)",
                name,
                err,
                args,
            )
            payload = {"ok": False, "error": str(err)}

        result_ref.tool_calls.append(
            {"name": name, "args": args, "result": payload}
        )
        return json.dumps(payload, ensure_ascii=False)

    return handler


async def run_consolidator(
    *,
    call_llm: ConsolidatorLLM,
    store: MemoryStore,
    entity: str,
    memories: Sequence[Memory],
    on_rate_limited: Optional[Callable[[], None]] = None,
) -> ConsolidationResult:
    """Feed one entity's memory batch to the consolidator LLM.

    Returns immediately with ``skipped=True`` if the batch is empty
    (nothing to do). Any exception from the LLM adapter is captured
    into ``result.error``; a 429-shaped error also fires the
    ``on_rate_limited`` hook (Hermes's rate-limit controller subscribes
    here to back off the whole loop).
    """
    result = ConsolidationResult(actions=0, skipped=False)
    if not memories:
        result.skipped = True
        return result

    message = format_input(entity, memories)
    handler = _make_tool_handler(store, result)

    try:
        await call_llm(
            system_prompt=CONSOLIDATOR_SYSTEM_PROMPT,
            message=message,
            temperature=0,
            tools=list(CONSOLIDATOR_TOOL_NAMES),
            thinking=False,
            must_reply=False,
            on_tool_call=handler,
            tool_context={"source": "consolidator", "entity": entity},
        )
    except Exception as err:  # noqa: BLE001
        logger.error("[consolidator] LLM call failed: %s", err)
        result.error = str(err)
        if _looks_like_rate_limited(err) and on_rate_limited is not None:
            try:
                on_rate_limited()
            except Exception as hook_err:  # noqa: BLE001
                logger.warning(
                    "[consolidator] on_rate_limited hook raised %s", hook_err
                )
        return result

    logger.info(
        "[consolidator] entity=%s memories=%d actions=%d%s",
        entity,
        len(memories),
        result.actions,
        " (explicit skip)" if result.skipped else "",
    )
    return result


# ── Loop ──────────────────────────────────────────────────────────


DEFAULT_RUN_INTERVAL_SECONDS = 30 * 60
DEFAULT_STARTUP_DELAY_SECONDS = 5 * 60
DEFAULT_BATCH_SIZE = 20
DEFAULT_CANDIDATE_LIMIT = 10


@dataclass
class ConsolidationLoopConfig:
    run_interval_seconds: float = DEFAULT_RUN_INTERVAL_SECONDS
    startup_delay_seconds: float = DEFAULT_STARTUP_DELAY_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT


class ConsolidationLoop:
    """Round-robin async loop over consolidation candidates.

    Cursor is in-memory only (v1 upstream doesn't persist either).
    On restart the cursor resets to 0; over time round-robin fairness
    is preserved because the candidate list itself rotates as entities
    are consolidated.

    Usage::

        loop = ConsolidationLoop(
            call_llm=my_llm,
            store=my_store,
            config=ConsolidationLoopConfig(run_interval_seconds=1800),
        )
        loop.start()
        ...
        await loop.stop()
    """

    def __init__(
        self,
        *,
        call_llm: ConsolidatorLLM,
        store: MemoryStore,
        config: Optional[ConsolidationLoopConfig] = None,
        on_rate_limited: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = None,  # type: ignore[assignment]
    ) -> None:
        self._call_llm = call_llm
        self._store = store
        self._config = config or ConsolidationLoopConfig()
        self._on_rate_limited = on_rate_limited
        self._cursor = 0
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def _tick(self) -> Optional[ConsolidationResult]:
        try:
            candidates = list(
                self._store.get_candidate_entities(self._config.candidate_limit)
            )
        except Exception as err:  # noqa: BLE001
            logger.error("[consolidation-loop] candidate query failed: %s", err)
            return None

        if not candidates:
            logger.info(
                "[consolidation-loop] no eligible entities (fact/person "
                "counts all < threshold)"
            )
            return None

        pick = candidates[self._cursor % len(candidates)]
        self._cursor = (self._cursor + 1) % len(candidates)

        try:
            memories = list(
                self._store.get_memories_by_entity(
                    pick.entity, self._config.batch_size
                )
            )
        except Exception as err:  # noqa: BLE001
            logger.error(
                "[consolidation-loop] memory fetch for entity=%s failed: %s",
                pick.entity,
                err,
            )
            return None

        if not memories:
            logger.info(
                "[consolidation-loop] entity=%s has no memories", pick.entity
            )
            return None

        logger.info(
            "[consolidation-loop] consolidating entity=%s (candidates=%d)",
            pick.entity,
            len(candidates),
        )
        return await run_consolidator(
            call_llm=self._call_llm,
            store=self._store,
            entity=pick.entity,
            memories=memories,
            on_rate_limited=self._on_rate_limited,
        )

    async def run_once(self) -> Optional[ConsolidationResult]:
        """Force an immediate tick. Useful for tests and for the skill
        that wants an on-demand consolidation pass outside the loop.
        """
        return await self._tick()

    def start(self) -> None:
        """Register the loop; first tick fires after
        ``startup_delay_seconds`` to avoid piling on the app's
        boot-time self-checks.
        """
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.get_event_loop().create_task(self._run_forever())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping.set()
        task = self._task
        self._task = None
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
        except asyncio.CancelledError:
            pass

    async def _run_forever(self) -> None:
        try:
            await asyncio.wait_for(
                self._stopping.wait(),
                timeout=self._config.startup_delay_seconds,
            )
            return
        except asyncio.TimeoutError:
            pass

        while not self._stopping.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                logger.error("[consolidation-loop] tick failed: %s", err)

            try:
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=self._config.run_interval_seconds,
                )
                return
            except asyncio.TimeoutError:
                continue

    # Diagnostics helpers used by tests -----------------------------

    @property
    def cursor(self) -> int:
        return self._cursor


__all__ = [
    "CONSOLIDATOR_SYSTEM_PROMPT",
    "CONSOLIDATOR_TOOL_NAMES",
    "CandidateEntity",
    "ConsolidationLoop",
    "ConsolidationLoopConfig",
    "ConsolidationResult",
    "ConsolidatorLLM",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CANDIDATE_LIMIT",
    "DEFAULT_RUN_INTERVAL_SECONDS",
    "DEFAULT_STARTUP_DELAY_SECONDS",
    "Memory",
    "MemoryStore",
    "format_input",
    "run_consolidator",
]
