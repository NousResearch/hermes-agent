"""Host commit contracts for pure external-engine sanitation."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import pytest

_SANITATION_GROWTH_BOUND = 1024
_DEFAULT_OPERATION = object()


def _placeholder(pattern: str, secret: str) -> str:
    parts = [
        f"[LCM sensitive redaction: name={pattern}; "
        f"chars={len(secret)}; bytes={len(secret.encode())}"
    ]
    if pattern != "password_assignment":
        parts.append(f"sha256={hashlib.sha256(secret.encode()).hexdigest()[:16]}")
    return "; ".join(parts) + "]"


def _placeholder_growth_fixture(rounds: int) -> tuple[list[dict], list[dict]]:
    """Exercise every host-visible container with shortest accepted secrets."""
    original: list[dict] = []
    sanitized: list[dict] = []
    for index in range(rounds):
        password = f"a{index:05d}"
        token = f"k{index:011d}"
        call_id = f"call-{index}"
        original.extend([
            {"role": "user", "content": f'password="{password}"'},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": (
                                f'{{"password":"{password}","api_key":"{token}"}}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": [
                    f"Bearer {token}",
                    {"client_secret": token, f"password={password}": password},
                ],
            },
            {"role": "assistant", "content": f"checked api_key={token}"},
        ])
        sanitized.extend([
            {
                "role": "user",
                "content": (
                    f'password="{_placeholder("password_assignment", password)}"'
                ),
            },
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": (
                                '{"password":"'
                                + _placeholder("password_assignment", password)
                                + '","api_key":"'
                                + _placeholder("api_key", token)
                                + '"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": [
                    f"Bearer {_placeholder('bearer_token', token)}",
                    {
                        "client_secret": _placeholder("api_key", token),
                        (
                            "password=" + _placeholder("password_assignment", password)
                        ): _placeholder("password_assignment", password),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": f"checked api_key={_placeholder('api_key', token)}",
            },
        ])
    return original, sanitized


class _MemoryManager:
    def __init__(self) -> None:
        self.pre_compress_calls = 0
        self.pre_compress_kwargs: list[dict[str, Any]] = []

    def on_pre_compress(self, _messages, **kwargs):
        self.pre_compress_calls += 1
        self.pre_compress_kwargs.append(kwargs)
        return "memory context"

    def on_session_switch(self, *_args, **_kwargs):
        raise AssertionError("sanitation must not switch memory sessions")


class _ExternalEngine:
    name = "fixture-external-engine"
    _last_compress_aborted = False
    _last_summary_error = None
    _last_compression_made_progress = True
    _last_summary_fallback_used = False
    _last_feasibility_skip = False
    compression_count = 1
    last_compression_rough_tokens = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    awaiting_real_usage_after_compression = False

    def __init__(
        self,
        candidate: list[dict],
        status: str | None,
        *,
        initial_status: str | None = "idle",
        current_operation: str | None = "sanitize",
        updates_status: bool = True,
    ) -> None:
        self.candidate = candidate
        if status is not None:
            self.last_compression_status = initial_status
        self.current_operation = current_operation
        self.updates_status = updates_status
        self.calls = 0
        self.call_options: list[dict[str, bool]] = []
        self.after_compress = None
        self.failure_cooldown_calls = 0

    def pending_compression_operation(self, _messages):
        return self.current_operation

    def compress(
        self,
        _messages,
        current_tokens=None,
        focus_topic=None,
        force=False,
        bypass_cooldown=False,
    ):
        self.calls += 1
        self.call_options.append(
            {"force": force, "bypass_cooldown": bypass_cooldown}
        )
        if self.updates_status and hasattr(self, "last_compression_status"):
            self.last_compression_status = self._result_status
        if self.after_compress is not None:
            self.after_compress()
        return copy.deepcopy(self.candidate)

    def _record_compression_failure_cooldown(self, *_args, **_kwargs):
        self.failure_cooldown_calls += 1

    _result_status = "sanitized"


@dataclass
class _Harness:
    agent: Any
    db: Any
    messages: list[dict]
    candidate: list[dict]
    memory: _MemoryManager
    session_end_calls: list[list[dict]]


def _make_harness(
    tmp_path,
    *,
    rounds: int,
    status: str | None = "sanitized",
    initial_status: str | None = "idle",
    current_operation: str | None | object = _DEFAULT_OPERATION,
    updates_status: bool = True,
) -> _Harness:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    session_id = f"sanitize-{rounds}-{status or 'legacy'}"
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = cast(
            Any,
            AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id=session_id,
                skip_context_files=True,
                skip_memory=True,
            ),
        )
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    agent._ensure_db_session()
    messages, candidate = _placeholder_growth_fixture(rounds)
    agent._flush_messages_to_session_db(messages, [])
    for original_message, candidate_message in zip(messages, candidate):
        for key in ("_row_id", "timestamp"):
            if key in original_message:
                candidate_message[key] = original_message[key]
    resolved_operation = (
        "sanitize"
        if current_operation is _DEFAULT_OPERATION and status == "sanitized"
        else cast(str | None, current_operation)
    )
    engine = _ExternalEngine(
        candidate,
        status,
        initial_status=initial_status,
        current_operation=resolved_operation,
        updates_status=updates_status,
    )
    engine._result_status = status
    agent.context_compressor = engine
    memory = _MemoryManager()
    agent._memory_manager = memory
    session_end_calls: list[list[dict]] = []
    agent.commit_memory_session = lambda value: session_end_calls.append(value)
    return _Harness(agent, db, messages, candidate, memory, session_end_calls)


def _without_persistence_markers(messages: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in message.items()
            if key not in {"_db_persisted", "_row_id", "timestamp"}
        }
        for message in messages
    ]


def test_required_checkpoint_fails_closed_before_pure_sanitation(tmp_path):
    """A result-only sanitation bridge cannot bypass a mandatory checkpoint."""
    from agent.conversation_compression import (
        CompressionCheckpointUnavailable,
        compress_context,
    )

    harness = _make_harness(
        tmp_path,
        rounds=1,
        initial_status="stale",
    )
    harness.agent.compression_checkpoint_required = True
    durable_before = harness.db.get_messages_as_conversation(
        harness.agent.session_id
    )

    with pytest.raises(
        CompressionCheckpointUnavailable,
        match="BLOCKED_MISSING_PREREQUISITE",
    ):
        compress_context(
            harness.agent,
            harness.messages,
            "system",
            approx_tokens=100_000,
        )

    assert harness.agent.context_compressor.calls == 0
    assert harness.memory.pre_compress_calls == 0
    assert harness.db.get_messages_as_conversation(
        harness.agent.session_id
    ) == durable_before


def test_required_checkpoint_runs_before_pure_sanitation_when_supported(tmp_path):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1, initial_status="stale")
    harness.agent.compression_checkpoint_required = True
    harness.memory.supports_pre_compress_checkpoint = lambda _version: True

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert harness.agent.context_compressor.calls == 1
    assert harness.memory.pre_compress_calls == 1
    assert harness.memory.pre_compress_kwargs[0]["require_checkpoint"] is True
    assert _without_persistence_markers(returned) == _without_persistence_markers(
        harness.candidate
    )


@pytest.mark.parametrize("result_status", ["reassembled", "stale", "exception"])
def test_statusless_external_engine_preserves_generic_memory_for_non_sanitation(
    tmp_path,
    result_status,
):
    """Memory timing follows the result, not a missing or stale pre-call status."""
    import agent.conversation_compression as compression

    harness = _make_harness(
        tmp_path,
        rounds=1,
        status=result_status,
        initial_status=None,
    )

    compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert harness.memory.pre_compress_calls == 1


def test_statusless_external_engine_can_report_pure_sanitation_without_memory_hook(
    tmp_path,
):
    import agent.conversation_compression as compression

    harness = _make_harness(
        tmp_path,
        rounds=1,
        initial_status=None,
    )

    compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert harness.memory.pre_compress_calls == 0


def test_stale_sanitized_status_cannot_classify_current_placeholder_result(
    tmp_path,
    monkeypatch,
):
    """A previous status cannot turn a generic current result into sanitation."""
    import agent.conversation_compression as compression

    harness = _make_harness(
        tmp_path,
        rounds=1,
        initial_status="sanitized",
        current_operation=None,
        updates_status=False,
    )
    generic_boundary_calls = 0

    def _fold(*_args):
        nonlocal generic_boundary_calls
        generic_boundary_calls += 1

    monkeypatch.setattr(compression, "_fold_todo_snapshot", _fold)
    harness.agent.context_compressor.after_compress = lambda: (
        harness.memory.pre_compress_calls == 1
        or (_ for _ in ()).throw(
            AssertionError("generic memory context must be gathered before compress")
        )
    )

    compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert generic_boundary_calls == 1
    assert harness.memory.pre_compress_calls == 1
    assert len(harness.session_end_calls) == 1


def test_known_generic_engine_without_memory_context_keeps_pre_call_memory_semantics(
    tmp_path,
):
    import agent.conversation_compression as compression

    harness = _make_harness(
        tmp_path,
        rounds=1,
        status="reassembled",
        current_operation=None,
    )
    harness.agent.context_compressor.candidate = [
        {"role": "user", "content": "generic compressed context"}
    ]
    harness.agent.context_compressor.after_compress = lambda: (
        harness.memory.pre_compress_calls == 1
        or (_ for _ in ()).throw(
            AssertionError("generic memory context must be gathered before compress")
        )
    )

    compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert harness.memory.pre_compress_calls == 1


def test_automatic_sanitation_commits_exact_candidate_without_boundary_side_effects(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Seven full shape rounds establish the smallest binary growth envelope."""
    import agent.context_compressor as context_compressor
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=7)
    growth = compression._sanitation_rough_tokens(
        harness.candidate
    ) - compression._sanitation_rough_tokens(harness.messages)
    assert 0 < growth <= _SANITATION_GROWTH_BOUND

    calls = {"todo": 0, "user": 0, "salvage": 0}
    monkeypatch.setattr(
        compression,
        "_fold_todo_snapshot",
        lambda *_args: calls.__setitem__("todo", calls["todo"] + 1),
    )

    def _user(*_args):
        calls["user"] += 1
        return "already_present"

    monkeypatch.setattr(compression, "_ensure_compressed_has_user_turn", _user)

    def _salvage(*_args, **_kwargs):
        calls["salvage"] += 1
        return None

    monkeypatch.setattr(context_compressor, "salvage_grown_transcript", _salvage)

    captured_commit = {}
    real_archive = harness.db.archive_and_compact

    def _archive(*args, **kwargs):
        captured_commit.update(kwargs)
        return real_archive(*args, **kwargs)

    monkeypatch.setattr(harness.db, "archive_and_compact", _archive)
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert _without_persistence_markers(returned) == _without_persistence_markers(
        harness.candidate
    )
    assert calls == {"todo": 0, "user": 0, "salvage": 0}
    assert harness.memory.pre_compress_calls == 0
    assert harness.session_end_calls == []
    assert captured_commit["watermark"] is not None
    assert captured_commit["lock_holder"]
    assert captured_commit["model_config_patch"] is None
    assert _without_persistence_markers(
        harness.db.get_messages_as_conversation(harness.agent.session_id)
    ) == _without_persistence_markers(harness.candidate)
    assert "operation=sanitize" in caplog.text
    assert "measurement=rough_message_tokens" in caplog.text
    assert f"growth_delta={growth}" in caplog.text
    assert "salvage=false" in caplog.text
    assert "terminal_result=committed" in caplog.text


def test_pure_sanitation_preserves_prompt_and_skips_generic_boundary_hooks(
    tmp_path,
    monkeypatch,
    caplog,
):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1)
    prompt = "".join(["stable-system-", "prompt"])
    harness.agent._cached_system_prompt = prompt
    harness.agent.context_compressor.compression_count = 2
    harness.agent.context_compressor._last_summary_error = "generic warning"
    harness.agent.event_callback = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("sanitation must not emit session:compress")
    )
    harness.agent._emit_warning = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("sanitation must not emit generic compression warnings")
    )
    statuses: list[str] = []
    harness.agent._emit_status = statuses.append
    monkeypatch.setattr(
        harness.db,
        "update_system_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sanitation must not rewrite the system prompt")
        ),
    )
    for name in (
        "_rebuild_system_prompt_at_boundary",
        "_notify_context_engine_compression_complete",
        "_queue_context_engine_compression_notification",
        "_reset_read_dedup_caches",
    ):
        monkeypatch.setattr(
            compression,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"sanitation called {_name}")
            ),
        )
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, returned_prompt = compression.compress_context(
        harness.agent,
        harness.messages,
        "different builder input",
        approx_tokens=100_000,
    )

    assert returned is not harness.messages
    assert returned_prompt is prompt
    assert harness.agent._cached_system_prompt is prompt
    assert harness.agent._last_compaction_in_place is True
    assert not any("accuracy may degrade" in status for status in statuses)
    assert "context compression done:" not in caplog.text
    assert not hasattr(
        harness.agent.context_compressor,
        "_verify_compaction_cleared_threshold",
    )


def test_pure_sanitation_is_forced_in_place_when_rotation_is_configured(tmp_path):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1)
    original_session_id = harness.agent.session_id
    harness.agent.compression_in_place = False

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert harness.agent.session_id == original_session_id
    assert harness.agent._last_compaction_in_place is True
    assert _without_persistence_markers(returned) == _without_persistence_markers(
        harness.candidate
    )


def test_structural_growth_scales_with_declared_redactions(tmp_path, caplog):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=20)
    aggregate_growth = compression._sanitation_rough_tokens(
        harness.candidate
    ) - compression._sanitation_rough_tokens(harness.messages)
    assert aggregate_growth > _SANITATION_GROWTH_BOUND
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert _without_persistence_markers(returned) == _without_persistence_markers(
        harness.candidate
    )
    assert "changed_fields=" in caplog.text
    assert "declared_placeholders=" in caplog.text
    assert "terminal_result=committed" in caplog.text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda candidate: candidate[0].__setitem__("unexpected", "addition"),
        lambda candidate: candidate[0].__setitem__(
            "content",
            candidate[0]["content"] + " arbitrary suffix",
        ),
        lambda candidate: candidate[0].__setitem__(
            "content",
            candidate[0]["content"].replace("chars=6", "chars=999"),
        ),
    ],
)
def test_structural_growth_rejects_undeclared_changes(tmp_path, mutate, caplog):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1)
    durable_before = harness.db.get_messages_as_conversation(
        harness.agent.session_id
    )
    mutate(harness.agent.context_compressor.candidate)
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert returned is harness.messages
    assert harness.db.get_messages_as_conversation(
        harness.agent.session_id
    ) == durable_before
    assert "terminal_result=refused_invalid_structure" in caplog.text


def test_engine_preflight_threshold_path_commits_sanitation(tmp_path):
    from agent.turn_context_compaction import (
        CompactionOutcome,
        _engine_preflight_maintenance,
    )

    harness = _make_harness(tmp_path, rounds=1)
    harness.agent.context_compressor.should_compress_preflight = (
        lambda _messages: True
    )
    outcome = CompactionOutcome(
        messages=harness.messages,
        active_system_prompt="system",
        conversation_history=[],
        current_turn_user_idx=0,
    )

    _engine_preflight_maintenance(
        harness.agent,
        outcome,
        harness.agent.context_compressor,
        100_000,
        "system",
        "default",
    )

    assert outcome.compressed is True
    assert outcome.messages is not harness.messages
    assert _without_persistence_markers(
        outcome.messages
    ) == _without_persistence_markers(harness.candidate)


@pytest.mark.parametrize(
    ("force", "bypass_cooldown"),
    [(True, False), (False, True)],
)
def test_manual_and_overflow_modes_keep_generic_boundary_behavior(
    tmp_path,
    force,
    bypass_cooldown,
):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1)
    harness.agent.context_compressor.candidate = [
        {"role": "user", "content": "generic compressed context"}
    ]
    events: list[tuple[str, dict]] = []
    harness.agent.event_callback = lambda name, payload: events.append((name, payload))

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
        force=force,
        bypass_cooldown=bypass_cooldown,
    )

    assert harness.agent.context_compressor.call_options == [
        {"force": force, "bypass_cooldown": bypass_cooldown}
    ]
    assert harness.memory.pre_compress_calls == 1
    assert len(harness.session_end_calls) == 1
    assert [event[0] for event in events] == ["session:compress"]
    assert _without_persistence_markers(returned) == [
        {"role": "user", "content": "generic compressed context"}
    ]


def test_sanitation_fence_cancellation_preserves_original(tmp_path, caplog):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1)
    durable_before = harness.db.get_messages_as_conversation(
        harness.agent.session_id
    )
    fence = compression.CompressionCommitFence()
    harness.agent.context_compressor.after_compress = fence.cancel_before_commit
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
        commit_fence=fence,
    )

    assert returned is harness.messages
    assert harness.db.get_messages_as_conversation(
        harness.agent.session_id
    ) == durable_before
    assert "cancelled before session mutation" in caplog.text


def test_sanitation_supersession_preserves_original(tmp_path, caplog):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1)
    durable_before = harness.db.get_messages_as_conversation(
        harness.agent.session_id
    )

    def supersede():
        harness.agent.context_compressor._compression_attempt_generation += 1

    harness.agent.context_compressor.after_compress = supersede
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert returned is harness.messages
    assert harness.db.get_messages_as_conversation(
        harness.agent.session_id
    ) == durable_before
    assert "superseded by a newer attempt" in caplog.text


def test_sanitation_commit_failure_rolls_back_without_boundary_hooks(
    tmp_path,
    monkeypatch,
    caplog,
):
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1)
    durable_before = harness.db.get_messages_as_conversation(
        harness.agent.session_id
    )
    monkeypatch.setattr(
        harness.db,
        "archive_and_compact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("commit failed")
        ),
    )
    harness.agent.event_callback = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("failed sanitation must not emit session:compress")
    )
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    assert returned is harness.messages
    assert harness.db.get_messages_as_conversation(
        harness.agent.session_id
    ) == durable_before
    assert harness.agent._last_compaction_in_place is False
    assert harness.agent.context_compressor.failure_cooldown_calls == 0
    assert "terminal_result=commit_failed" in caplog.text


@pytest.mark.parametrize(
    ("status", "watermark_failure", "terminal_result"),
    [
        ("sanitized", True, "refused_missing_watermark"),
        ("reassembled", False, None),
        (None, False, None),
    ],
)
def test_sanitation_bridge_is_status_narrow(
    tmp_path,
    monkeypatch,
    caplog,
    status,
    watermark_failure,
    terminal_result,
):
    """Ambiguous and legacy results stay generic; sanitation requires a watermark."""
    import agent.context_compressor as context_compressor
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=1, status=status)
    if watermark_failure:
        monkeypatch.setattr(
            harness.db,
            "get_active_message_watermark",
            lambda _session_id: (_ for _ in ()).throw(
                RuntimeError("watermark unavailable")
            ),
        )
    calls = {"todo": 0, "user": 0, "salvage": 0}
    monkeypatch.setattr(
        compression,
        "_fold_todo_snapshot",
        lambda *_args: calls.__setitem__("todo", calls["todo"] + 1),
    )

    def _user(*_args):
        calls["user"] += 1
        return "already_present"

    monkeypatch.setattr(compression, "_ensure_compressed_has_user_turn", _user)

    def _salvage(_messages, _candidate, *, budget):
        calls["salvage"] += 1
        return [{"role": "user", "content": "generic salvage"}]

    monkeypatch.setattr(context_compressor, "salvage_grown_transcript", _salvage)
    caplog.set_level(logging.INFO, logger="agent.conversation_compression")

    returned, _ = compression.compress_context(
        harness.agent,
        harness.messages,
        "system",
        approx_tokens=100_000,
    )

    if terminal_result is not None:
        assert returned is harness.messages
        assert calls == {"todo": 0, "user": 0, "salvage": 0}
        assert harness.memory.pre_compress_calls == 0
        assert harness.session_end_calls == []
        assert harness.db.get_messages_as_conversation(harness.agent.session_id)
        assert f"terminal_result={terminal_result}" in caplog.text
        assert harness.agent.context_compressor.calls == 1
    else:
        assert calls == {"todo": 1, "user": 1, "salvage": 1}
        assert harness.memory.pre_compress_calls == 1
        assert len(harness.session_end_calls) == 1
        assert _without_persistence_markers(returned) == [
            {"role": "user", "content": "generic salvage"}
        ]
