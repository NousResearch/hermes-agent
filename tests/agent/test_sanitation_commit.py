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

    def on_pre_compress(self, _messages, **_kwargs):
        self.pre_compress_calls += 1
        return "memory context"


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

    def __init__(self, candidate: list[dict], status: str | None) -> None:
        self.candidate = candidate
        if status is not None:
            self.last_compression_status = "idle"
        self.calls = 0

    def compress(
        self,
        _messages,
        current_tokens=None,
        focus_topic=None,
        force=False,
        bypass_cooldown=False,
    ):
        self.calls += 1
        if hasattr(self, "last_compression_status"):
            self.last_compression_status = self._result_status
        return copy.deepcopy(self.candidate)

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
    tmp_path, *, rounds: int, status: str | None = "sanitized"
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
    engine = _ExternalEngine(candidate, status)
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
    assert _without_persistence_markers(
        harness.db.get_messages_as_conversation(harness.agent.session_id)
    ) == _without_persistence_markers(harness.candidate)
    assert "operation=sanitize" in caplog.text
    assert "measurement=rough_message_tokens" in caplog.text
    assert f"growth_delta={growth}" in caplog.text
    assert "salvage=false" in caplog.text
    assert "terminal_result=committed" in caplog.text


@pytest.mark.parametrize(
    ("status", "force", "bypass_cooldown", "rounds", "watermark_failure", "terminal_result"),
    [
        ("sanitized", False, False, 8, False, "refused_growth_bound"),
        ("sanitized", False, False, 1, True, "refused_missing_watermark"),
        ("sanitized", True, False, 1, False, None),
        ("sanitized", False, True, 1, False, None),
        ("reassembled", False, False, 1, False, None),
        (None, False, False, 1, False, None),
    ],
)
def test_sanitation_bridge_is_bounded_and_narrow(
    tmp_path,
    monkeypatch,
    caplog,
    status,
    force,
    bypass_cooldown,
    rounds,
    watermark_failure,
    terminal_result,
):
    """Beyond-bound sanitation refuses; force, overflow, ambiguous, and legacy results stay generic."""
    import agent.context_compressor as context_compressor
    import agent.conversation_compression as compression

    harness = _make_harness(tmp_path, rounds=rounds, status=status)
    if watermark_failure:
        monkeypatch.setattr(
            harness.db,
            "get_active_message_watermark",
            lambda _session_id: (_ for _ in ()).throw(
                RuntimeError("watermark unavailable")
            ),
        )
    growth = compression._sanitation_rough_tokens(
        harness.candidate
    ) - compression._sanitation_rough_tokens(harness.messages)
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
        force=force,
        bypass_cooldown=bypass_cooldown,
    )

    if terminal_result is not None:
        if terminal_result == "refused_growth_bound":
            assert growth > _SANITATION_GROWTH_BOUND
        else:
            assert growth <= _SANITATION_GROWTH_BOUND
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
