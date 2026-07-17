# SPDX-License-Identifier: Apache-2.0
"""Tests for the memory consolidator + round-robin loop.

The consolidator is dependency-injected end-to-end: the store is a
simple in-memory dict-backed fake, the LLM adapter is a callable that
replays a scripted list of tool calls. This lets the tests verify:

* Merge / downgrade / skip tool dispatch (state changes in the fake
  store, action counter reflects successful ops only).
* Rate-limit hook fires on 429-shaped exceptions.
* Errors in one tool call don't abort the batch.
* Round-robin cursor advances across candidate entities and wraps.
* Empty candidate list / empty batch short-circuits with no LLM call.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from agent.memory_consolidator import (
    CONSOLIDATOR_SYSTEM_PROMPT,
    CONSOLIDATOR_TOOL_NAMES,
    CandidateEntity,
    ConsolidationLoop,
    ConsolidationLoopConfig,
    ConsolidationResult,
    Memory,
    format_input,
    run_consolidator,
)


# ── Fakes ──────────────────────────────────────────────────────────


@dataclass
class FakeStore:
    """In-memory memory store. Deliberately mutable so tests can
    assert the exact rows that survive after a consolidation pass.
    """

    candidates: list[CandidateEntity] = field(default_factory=list)
    memories_by_entity: dict[str, list[Memory]] = field(default_factory=dict)
    merges: list[dict] = field(default_factory=list)
    downgrades: list[dict] = field(default_factory=list)
    fail_next_merge: bool = False

    def get_candidate_entities(self, limit: int) -> list[CandidateEntity]:
        return list(self.candidates[:limit])

    def get_memories_by_entity(self, entity: str, limit: int) -> list[Memory]:
        return list(self.memories_by_entity.get(entity, [])[:limit])

    def merge_memories(
        self,
        *,
        keep_mem_id: str,
        drop_mem_ids,
        merged_content: str,
        merged_salience,
        reason: str,
    ) -> bool:
        if self.fail_next_merge:
            self.fail_next_merge = False
            return False
        self.merges.append(
            {
                "keep": keep_mem_id,
                "drops": list(drop_mem_ids),
                "content": merged_content,
                "salience": merged_salience,
                "reason": reason,
            }
        )
        # Apply the merge to the fake row set (any entity that has
        # these ids). We soft-delete drops by removing them from the
        # active list; a real store would set visibility=0.
        for mems in self.memories_by_entity.values():
            for drop_id in drop_mem_ids:
                mems[:] = [m for m in mems if m.mem_id != drop_id]
        return True

    def downgrade_memory(
        self, *, mem_id: str, new_salience: int, reason: str
    ) -> bool:
        self.downgrades.append(
            {"mem_id": mem_id, "salience": new_salience, "reason": reason}
        )
        for mems in self.memories_by_entity.values():
            for m in mems:
                if m.mem_id == mem_id:
                    m.salience = new_salience
                    return True
        return False


class ScriptedLLM:
    """Async LLM adapter that replays a scripted list of tool calls.

    Each script entry is ``(tool_name, args_dict)``. The adapter feeds
    them to the caller's ``on_tool_call`` handler in order — this is
    exactly the seam the real LLM adapter drives.
    """

    def __init__(self, script: list[tuple[str, dict]] | Exception):
        self._script = script
        self.calls: list[dict] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        script = self._script
        if isinstance(script, Exception):
            raise script
        handler = kwargs.get("on_tool_call")
        assert callable(handler), "on_tool_call handler must be provided"
        for name, args in script:
            handler(name, args)
        return {"content": ""}


class _StatusError(Exception):
    """LLM error carrying an HTTP status attribute (matches real
    provider SDK behaviour).
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ── format_input + prompt sanity ────────────────────────────────────


def test_prompt_lists_tool_names_the_runner_registers() -> None:
    for name in CONSOLIDATOR_TOOL_NAMES:
        assert name in CONSOLIDATOR_SYSTEM_PROMPT


def test_format_input_shows_entity_and_all_memories() -> None:
    mems = [
        Memory(
            mem_id="m1",
            event_type="fact",
            title="喜欢简洁",
            content="用户偏好简洁回答",
            salience=4,
            timestamp="2026-07-01T12:00:00Z",
        ),
        Memory(
            mem_id="m2",
            event_type="fact",
            title="偏好直接",
            content="用户希望回答直接",
            salience=3,
            timestamp="2026-07-05T09:00:00Z",
        ),
    ]
    text = format_input("user:qzl", mems)
    assert "[Entity] user:qzl" in text
    assert "[Memory count] 2" in text
    assert "m1" in text and "m2" in text
    # Timestamp is trimmed to date.
    assert "2026-07-01\n" in text


# ── run_consolidator ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_consolidator_empty_batch_shortcircuits() -> None:
    store = FakeStore()
    llm = ScriptedLLM([])
    result = await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=[],
    )
    assert result.skipped is True
    assert result.actions == 0
    # LLM was never called.
    assert llm.calls == []


@pytest.mark.asyncio
async def test_run_consolidator_dispatches_merge_and_downgrade() -> None:
    store = FakeStore(
        memories_by_entity={
            "user:qzl": [
                Memory("m1", "fact", "喜欢简洁", "简洁", 4, ""),
                Memory("m2", "fact", "偏好直接", "直接", 3, ""),
                Memory("m3", "fact", "旧习惯", "已过期", 2, ""),
            ]
        }
    )
    llm = ScriptedLLM(
        [
            (
                "merge_memories",
                {
                    "keep_mem_id": "m1",
                    "drop_mem_ids": ["m2"],
                    "merged_content": "用户偏好简洁直接的回答",
                    "merged_salience": 4,
                    "reason": "semantic duplicate",
                },
            ),
            (
                "downgrade_memory",
                {
                    "mem_id": "m3",
                    "new_salience": 1,
                    "reason": "stale, unreinforced",
                },
            ),
        ]
    )
    result = await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=store.get_memories_by_entity("user:qzl", 20),
    )
    assert result.actions == 2
    assert result.skipped is False
    assert result.error is None
    assert len(store.merges) == 1 and store.merges[0]["keep"] == "m1"
    assert len(store.downgrades) == 1 and store.downgrades[0]["salience"] == 1


@pytest.mark.asyncio
async def test_run_consolidator_records_skip() -> None:
    store = FakeStore(
        memories_by_entity={
            "user:qzl": [Memory("m1", "fact", "t", "c", 3, "")]
        }
    )
    llm = ScriptedLLM(
        [("skip_consolidation", {"reason": "nothing to do"})]
    )
    result = await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=store.get_memories_by_entity("user:qzl", 20),
    )
    assert result.skipped is True
    assert result.actions == 0


@pytest.mark.asyncio
async def test_run_consolidator_failed_merge_does_not_count() -> None:
    store = FakeStore(
        memories_by_entity={
            "user:qzl": [
                Memory("m1", "fact", "t", "c", 3, ""),
                Memory("m2", "fact", "t", "c", 3, ""),
            ]
        },
        fail_next_merge=True,
    )
    llm = ScriptedLLM(
        [
            (
                "merge_memories",
                {
                    "keep_mem_id": "m1",
                    "drop_mem_ids": ["m2"],
                    "merged_content": "x",
                    "merged_salience": 3,
                    "reason": "dup",
                },
            ),
        ]
    )
    result = await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=store.get_memories_by_entity("user:qzl", 20),
    )
    # Store said "no" → action not counted, no exception.
    assert result.actions == 0
    assert store.merges == []


@pytest.mark.asyncio
async def test_run_consolidator_invalid_args_are_captured_not_raised() -> None:
    store = FakeStore(
        memories_by_entity={
            "user:qzl": [Memory("m1", "fact", "t", "c", 3, "")]
        }
    )
    # Missing drop_mem_ids for merge; missing new_salience for downgrade.
    llm = ScriptedLLM(
        [
            ("merge_memories", {"keep_mem_id": "m1", "drop_mem_ids": []}),
            ("downgrade_memory", {"mem_id": "m1"}),
            ("nonsense_tool", {"foo": "bar"}),
        ]
    )
    result = await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=store.get_memories_by_entity("user:qzl", 20),
    )
    # None of the malformed calls should have incremented actions.
    assert result.actions == 0
    # But they should all show up in the tool call log for auditing.
    assert len(result.tool_calls) == 3
    for entry in result.tool_calls:
        assert entry["result"]["ok"] is False


@pytest.mark.asyncio
async def test_run_consolidator_captures_llm_error() -> None:
    store = FakeStore(memories_by_entity={"user:qzl": [Memory("m1", "fact", "t", "c", 3, "")]})
    llm = ScriptedLLM(RuntimeError("boom"))
    hits: list[int] = []
    result = await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=store.get_memories_by_entity("user:qzl", 20),
        on_rate_limited=lambda: hits.append(1),
    )
    assert result.error and "boom" in result.error
    assert result.actions == 0
    # Non-429 error should NOT fire the rate-limit hook.
    assert hits == []


@pytest.mark.asyncio
async def test_run_consolidator_fires_rate_limit_hook_on_429_status() -> None:
    store = FakeStore(memories_by_entity={"user:qzl": [Memory("m1", "fact", "t", "c", 3, "")]})
    llm = ScriptedLLM(_StatusError(429, "too many"))
    hits: list[int] = []
    await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=store.get_memories_by_entity("user:qzl", 20),
        on_rate_limited=lambda: hits.append(1),
    )
    assert hits == [1]


@pytest.mark.asyncio
async def test_run_consolidator_fires_rate_limit_hook_on_429_message() -> None:
    store = FakeStore(memories_by_entity={"user:qzl": [Memory("m1", "fact", "t", "c", 3, "")]})
    llm = ScriptedLLM(RuntimeError("HTTP 429 rate limit"))
    hits: list[int] = []
    await run_consolidator(
        call_llm=llm,
        store=store,
        entity="user:qzl",
        memories=store.get_memories_by_entity("user:qzl", 20),
        on_rate_limited=lambda: hits.append(1),
    )
    assert hits == [1]


# ── ConsolidationLoop ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_run_once_advances_cursor_round_robin() -> None:
    store = FakeStore(
        candidates=[
            CandidateEntity(entity="user:a", memory_count=5),
            CandidateEntity(entity="user:b", memory_count=5),
        ],
        memories_by_entity={
            "user:a": [Memory("a1", "fact", "t", "c", 3, "")],
            "user:b": [Memory("b1", "fact", "t", "c", 3, "")],
        },
    )
    seen: list[str] = []

    async def call_llm(**kwargs):
        # Peek at the batch by reading the message payload.
        seen.append(kwargs["message"].splitlines()[0])
        handler = kwargs["on_tool_call"]
        handler("skip_consolidation", {"reason": "test"})
        return {"content": ""}

    loop = ConsolidationLoop(
        call_llm=call_llm,
        store=store,
        config=ConsolidationLoopConfig(
            run_interval_seconds=0,
            startup_delay_seconds=0,
            batch_size=20,
            candidate_limit=10,
        ),
    )
    await loop.run_once()
    await loop.run_once()
    await loop.run_once()  # wraps back
    # We processed a, b, a in that order.
    assert [line for line in seen] == [
        "[Entity] user:a",
        "[Entity] user:b",
        "[Entity] user:a",
    ]


@pytest.mark.asyncio
async def test_loop_run_once_skips_when_no_candidates() -> None:
    store = FakeStore(candidates=[])
    called: list[int] = []

    async def call_llm(**kwargs):  # noqa: ARG001
        called.append(1)
        return {"content": ""}

    loop = ConsolidationLoop(
        call_llm=call_llm,
        store=store,
        config=ConsolidationLoopConfig(
            run_interval_seconds=0,
            startup_delay_seconds=0,
        ),
    )
    result = await loop.run_once()
    assert result is None
    assert called == []  # never invoked


@pytest.mark.asyncio
async def test_loop_run_once_skips_when_entity_has_no_memories() -> None:
    store = FakeStore(
        candidates=[CandidateEntity(entity="user:empty")],
        memories_by_entity={},
    )
    called: list[int] = []

    async def call_llm(**kwargs):  # noqa: ARG001
        called.append(1)
        return {"content": ""}

    loop = ConsolidationLoop(
        call_llm=call_llm,
        store=store,
        config=ConsolidationLoopConfig(
            run_interval_seconds=0,
            startup_delay_seconds=0,
        ),
    )
    result = await loop.run_once()
    assert result is None
    assert called == []


@pytest.mark.asyncio
async def test_loop_start_stop_cleanly() -> None:
    """The background loop task must start and stop without leaks or
    orphan warnings when startup_delay is short.
    """
    store = FakeStore(candidates=[])

    async def call_llm(**kwargs):  # noqa: ARG001
        return {"content": ""}

    loop = ConsolidationLoop(
        call_llm=call_llm,
        store=store,
        config=ConsolidationLoopConfig(
            run_interval_seconds=0.05,
            startup_delay_seconds=0.01,
        ),
    )
    loop.start()
    # Let it fire a couple of no-candidate ticks.
    await asyncio.sleep(0.1)
    await loop.stop()
    # After stop, calling stop again is a no-op.
    await loop.stop()
