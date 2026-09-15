"""Real Hermes/LCM sanitation integration contract.

Set ``HERMES_LCM_REPO`` to a hermes-lcm checkout.  The canonical Hermes test
runner forwards that explicit, non-secret opt-in so this file imports the
checkout as the standalone ``hermes_lcm`` plugin package.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest


def _load_real_lcm():
    checkout_value = os.environ.get("HERMES_LCM_REPO", "").strip()
    if not checkout_value:
        pytest.skip("set HERMES_LCM_REPO to a hermes-lcm checkout")
    checkout = Path(checkout_value).resolve()
    package_init = checkout / "__init__.py"
    if not package_init.is_file() or not (checkout / "engine.py").is_file():
        pytest.skip(f"HERMES_LCM_REPO is not a hermes-lcm checkout: {checkout}")

    loaded = sys.modules.get("hermes_lcm")
    if loaded is not None:
        loaded_root = Path(loaded.__path__[0]).resolve()
        if loaded_root != checkout:
            raise AssertionError(
                f"hermes_lcm already imported from {loaded_root}, expected {checkout}"
            )
    else:
        spec = importlib.util.spec_from_file_location(
            "hermes_lcm",
            package_init,
            submodule_search_locations=[str(checkout)],
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules["hermes_lcm"] = package
        spec.loader.exec_module(package)

    return (
        importlib.import_module("hermes_lcm.engine"),
        importlib.import_module("hermes_lcm.config"),
        importlib.import_module("hermes_lcm.tokens"),
    )


def _real_engine(home: Path, lcm_engine_module, lcm_config_module):
    config = lcm_config_module.LCMConfig(
        database_path=str(home / "lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=120,
        large_output_externalization_path=str(home / "externalized"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20_000,
        context_threshold=0.95,
        sensitive_patterns_enabled=True,
        sensitive_patterns=[
            "api_key",
            "bearer_token",
            "password_assignment",
            "private_key",
        ],
    )
    engine = lcm_engine_module.LCMEngine(config=config, hermes_home=str(home))
    engine.threshold_tokens = 90_000
    return engine, config


def _sensitive_messages() -> tuple[list[dict[str, Any]], list[str]]:
    raw_values = [
        "sk-synthetic-host-scalar-000000000000",
        "sk-synthetic-host-list-00000000000000",
        "sk-synthetic-host-dict-00000000000000",
        "sk-synthetic-host-key-000000000000000",
        "sk-synthetic-host-json-00000000000000",
        "sk-synthetic-host-toolcall-0000000000",
        "hostpw7",
    ]
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "old production request " * 20},
        {"role": "assistant", "content": "old production response " * 20},
        {
            "role": "user",
            "content": f"api_key={raw_values[0]} password={raw_values[6]}",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Bearer {raw_values[1]}"},
                {"type": "metadata", "value": {"api_key": raw_values[2]}},
                {f"api_key={raw_values[3]}": "safe-value"},
            ],
            "tool_calls": [
                {
                    "id": "call-sensitive",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"api_key": raw_values[5]}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-sensitive",
            "content": (
                json.dumps({"api_key": raw_values[4]})
                + " externalized filler "
                + ("safe-filler " * 30)
            ),
        },
        {"role": "user", "content": "fresh follow-up"},
    ]
    return messages, raw_values


def _without_persistence_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in message.items()
            if key not in {"_db_persisted", "_row_id", "timestamp"}
        }
        for message in messages
    ]


def _sqlite_values(path: Path) -> str:
    with sqlite3.connect(path) as conn:
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        values = []
        for table_name in table_names:
            quoted = table_name.replace('"', '""')
            values.extend(repr(row) for row in conn.execute(f'SELECT * FROM "{quoted}"'))
    return "\n".join(values)


def _make_agent(home: Path, messages: list[dict[str, Any]], engine):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(home / "state.db")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id="paired-sanitation",
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    agent._ensure_db_session()
    agent._flush_messages_to_session_db(messages, [])
    engine.on_session_start(
        agent.session_id,
        hermes_home=str(home),
        platform="synthetic",
        conversation_id="paired-sanitation-conversation",
        context_length=100_000,
    )
    agent.context_compressor = engine
    return agent, db


def test_real_lcm_sanitation_commits_without_secret_residue(
    tmp_path,
    monkeypatch,
    caplog,
):
    lcm_engine_module, lcm_config_module, token_module = _load_real_lcm()
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    messages, raw_values = _sensitive_messages()
    engine, config = _real_engine(home, lcm_engine_module, lcm_config_module)
    agent, db = _make_agent(home, messages, engine)

    summary_spy = Mock(
        side_effect=AssertionError("pure sanitation must not summarize or use network")
    )
    monkeypatch.setattr(
        lcm_engine_module,
        "summarize_with_escalation",
        summary_spy,
    )

    prepared_claims: list[object] = []
    received_claims: list[object] = []
    real_prepare = engine.prepare_compression_operation
    real_compress = engine.compress

    def observe_prepare(*args, **kwargs):
        prepared = real_prepare(*args, **kwargs)
        if prepared is not None:
            prepared_claims.append(prepared[1])
        return prepared

    def observe_compress(*args, **kwargs):
        received_claims.append(kwargs.get("operation_claim"))
        return real_compress(*args, **kwargs)

    monkeypatch.setattr(engine, "prepare_compression_operation", observe_prepare)
    monkeypatch.setattr(engine, "compress", observe_compress)

    forbidden_calls = {
        "todo": 0,
        "user": 0,
        "salvage": 0,
        "memory": 0,
        "session_boundary": 0,
    }

    def forbid(name):
        def _forbidden(*_args, **_kwargs):
            forbidden_calls[name] += 1
            raise AssertionError(f"sanitation called forbidden {name} hook")

        return _forbidden

    import agent.context_compressor as context_compressor
    import agent.conversation_compression as compression

    monkeypatch.setattr(compression, "_fold_todo_snapshot", forbid("todo"))
    monkeypatch.setattr(
        compression,
        "_ensure_compressed_has_user_turn",
        forbid("user"),
    )
    monkeypatch.setattr(
        context_compressor,
        "salvage_grown_transcript",
        forbid("salvage"),
    )
    agent._memory_manager = SimpleNamespace(on_pre_compress=forbid("memory"))
    agent.commit_memory_session = forbid("session_boundary")
    agent.event_callback = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("sanitation emitted a generic compression event")
    )
    initial_toolsets = copy.deepcopy(agent.enabled_toolsets)
    prompt = "".join(["byte-stable-", "system-prompt"])
    agent._cached_system_prompt = prompt
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

    commit_calls: list[tuple[str, dict[str, Any]]] = []
    commit_method_name = (
        "sanitize_and_compact"
        if hasattr(db, "sanitize_and_compact")
        else "archive_and_compact"
    )
    real_commit = getattr(db, commit_method_name)

    def observe_commit(*args, **kwargs):
        commit_calls.append((commit_method_name, dict(kwargs)))
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(db, commit_method_name, observe_commit)
    caplog.set_level(logging.INFO)

    assert token_module.count_messages_tokens(messages) < engine.threshold_tokens
    assert engine.should_compress_preflight(copy.deepcopy(messages)) is True
    returned, returned_prompt = compression.compress_context(
        agent,
        messages,
        "different prompt builder input",
        approx_tokens=token_module.count_messages_tokens(messages),
    )

    assert len(prepared_claims) == 1
    assert received_claims == [prepared_claims[0]]
    assert forbidden_calls == {
        "todo": 0,
        "user": 0,
        "salvage": 0,
        "memory": 0,
        "session_boundary": 0,
    }
    assert summary_spy.call_count == 0
    assert returned_prompt is prompt
    assert agent._cached_system_prompt is prompt
    assert agent.enabled_toolsets == initial_toolsets
    assert agent.session_id == "paired-sanitation"
    assert agent._last_compaction_in_place is True
    assert len(commit_calls) == 1
    assert commit_calls[0][1]["watermark"] > 0
    assert commit_calls[0][1]["lock_holder"]
    assert "operation=sanitize" in caplog.text
    assert "terminal_result=committed" in caplog.text

    expected_replay = _without_persistence_markers(returned)
    durable_replay = _without_persistence_markers(
        db.get_messages_as_conversation(agent.session_id)
    )
    assert durable_replay == expected_replay
    assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    engine.shutdown()

    restarted = lcm_engine_module.LCMEngine(config=config, hermes_home=str(home))
    restarted.threshold_tokens = 90_000
    restarted.on_session_start(
        agent.session_id,
        hermes_home=str(home),
        platform="synthetic",
        conversation_id="paired-sanitation-conversation",
        context_length=100_000,
    )
    assert restarted.should_compress_preflight(copy.deepcopy(expected_replay)) is False
    replay = restarted._ingest_messages(copy.deepcopy(expected_replay))
    assert json.dumps(replay, ensure_ascii=False, separators=(",", ":")).encode() == (
        json.dumps(expected_replay, ensure_ascii=False, separators=(",", ":")).encode()
    )
    assert restarted._dag.get_session_node_count(restarted.current_session_id) == 0

    state_values = _sqlite_values(home / "state.db")
    lcm_values = _sqlite_values(home / "lcm.db")
    externalized_values = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (home / "externalized").glob("*.json")
    )
    for raw in raw_values:
        assert raw not in state_values
        assert db.search_messages(raw, include_inactive=True) == []
        assert raw not in lcm_values
        assert restarted._store.search(
            raw,
            session_id=restarted.current_session_id,
        ) == []
        assert raw not in externalized_values
        assert raw not in caplog.text
    assert commit_calls[0][0] == "sanitize_and_compact"
    restarted.shutdown()


@pytest.mark.parametrize(
    "invalidation",
    ["messages", "session", "generation", "replay"],
)
def test_real_lcm_claim_invalidation_is_fail_closed(
    tmp_path,
    monkeypatch,
    invalidation,
):
    lcm_engine_module, lcm_config_module, _ = _load_real_lcm()
    home = tmp_path / invalidation
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    engine, _ = _real_engine(home, lcm_engine_module, lcm_config_module)
    engine.on_session_start(
        "claim-session",
        platform="synthetic",
        conversation_id="claim-conversation",
        context_length=100_000,
    )
    messages = [
        {
            "role": "user",
            "content": "api_key=sk-synthetic-invalid-integration-000000000",
        },
        {"role": "assistant", "content": "fresh answer"},
    ]
    monkeypatch.setattr(
        lcm_engine_module,
        "summarize_with_escalation",
        Mock(side_effect=AssertionError("invalid claims must not summarize")),
    )
    engine._compression_attempt_generation = 41
    assert engine.should_compress_preflight(copy.deepcopy(messages)) is True
    prepared = engine.prepare_compression_operation(
        copy.deepcopy(messages),
        session_id=engine.bound_session_id,
        attempt_generation=41,
    )
    assert prepared is not None
    _, claim = prepared

    invocation_messages = copy.deepcopy(messages)
    if invalidation == "messages":
        invocation_messages[0]["content"] += " changed"
    elif invalidation == "session":
        engine.on_session_start(
            "other-session",
            platform="synthetic",
            conversation_id="other-conversation",
            context_length=100_000,
        )
    elif invalidation == "generation":
        engine._compression_attempt_generation = 42

    if invalidation == "replay":
        first = engine.compress(invocation_messages, operation_claim=claim)
        assert isinstance(first, tuple)
        assert first[1] is claim
        second = engine.compress(copy.deepcopy(messages), operation_claim=claim)
        assert isinstance(second, list)
    else:
        first = engine.compress(invocation_messages, operation_claim=claim)
        assert isinstance(first, list)
    engine.shutdown()


@pytest.mark.parametrize(
    ("force", "bypass_cooldown"),
    [(True, False), (False, True)],
)
def test_real_lcm_manual_and_overflow_paths_remain_generic(
    tmp_path,
    monkeypatch,
    force,
    bypass_cooldown,
):
    lcm_engine_module, lcm_config_module, _ = _load_real_lcm()
    home = tmp_path / f"{force}-{bypass_cooldown}"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    messages, _ = _sensitive_messages()
    engine, _ = _real_engine(home, lcm_engine_module, lcm_config_module)
    agent, db = _make_agent(home, messages, engine)
    prepare_calls = 0
    real_prepare = engine.prepare_compression_operation

    def observe_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(engine, "prepare_compression_operation", observe_prepare)
    monkeypatch.setattr(
        lcm_engine_module,
        "summarize_with_escalation",
        Mock(return_value=("generic deterministic summary", 1)),
    )

    import agent.conversation_compression as compression

    returned, _ = compression.compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100_000,
        force=force,
        bypass_cooldown=bypass_cooldown,
    )

    assert prepare_calls == 0
    assert isinstance(returned, list)
    assert engine.last_compression_status != "sanitized"
    engine.shutdown()
    db.close()
