import json
import sqlite3
from pathlib import Path

import pytest

from scripts.render_training_config import render_config
from scripts.hermes_training_corpus import (
    CorpusRedactor,
    build_dpo_record,
    build_sft_record,
    iter_codex_rollout_records,
    iter_harness_result_records,
    iter_state_db_records,
    write_jsonl,
)


def test_redactor_masks_secrets_paths_tunnels_and_session_ids(tmp_path):
    redactor = CorpusRedactor(
        user_home=Path("C:/Users/downl"),
        hermes_home=Path("C:/Users/downl/.hermes"),
        repo_root=Path("C:/Users/downl/Documents/New project/hermes-agent"),
    )

    text = (
        "OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012 "
        "C:/Users/downl/.hermes/state.db "
        "D:/private/model.gguf "
        "https://example.ngrok-free.app/webhook "
        "019eb9fa071e7072a954cfa2945e3e87"
    )

    result = redactor.redact_text(text)

    assert "abc123def456" not in result
    assert "<HERMES_HOME>/state.db" in result
    assert "<LOCAL_PATH:1>" in result
    assert "<PUBLIC_TUNNEL_URL>" in result
    assert "<SESSION_ID:1>" in result


def test_build_sft_refuses_unredacted_record():
    with pytest.raises(ValueError, match="unredacted"):
        build_sft_record({"source": "state_db", "redacted": False, "messages": []})


def test_build_sft_includes_tool_calls_from_redacted_record():
    record = {
        "source": "state_db",
        "redacted": True,
        "session": {"id": "<SESSION_ID:1>", "source": "cli", "title": "restart"},
        "messages": [
            {"role": "user", "content": "Restart Hermes"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "tool_name": "terminal", "content": "ok"},
            {"role": "assistant", "content": "Done"},
        ],
    }

    sft = build_sft_record(record)

    assert sft is not None
    assert sft["messages"][0]["role"] == "system"
    assert sft["tools"] == ["terminal"]
    assert "tool-calling" in sft["metadata"]["tags"]


def test_iter_codex_rollout_records_maps_messages_and_tool_calls(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rows = [
        {"type": "session_meta", "timestamp": 1.0, "payload": {"payload": {"id": "abc123", "cwd": "C:/Users/downl/project"}}},
        {
            "type": "response_item",
            "timestamp": 2.0,
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "restart"}]},
        },
        {
            "type": "response_item",
            "timestamp": 3.0,
            "payload": {"type": "function_call", "call_id": "call_1", "name": "terminal", "arguments": "{\"cmd\":\"date\"}"},
        },
        {
            "type": "response_item",
            "timestamp": 4.0,
            "payload": {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        },
        {
            "type": "response_item",
            "timestamp": 5.0,
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        },
    ]
    rollout.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    record = next(iter_codex_rollout_records([rollout]))
    redacted = CorpusRedactor(user_home=Path("C:/Users/downl")).redact_record(record)
    sft = build_sft_record(redacted)

    assert record["source"] == "codex_rollout"
    assert record["messages"][1]["role"] == "assistant"
    assert record["messages"][1]["tool_calls"][0]["function"]["name"] == "terminal"
    assert sft is not None
    assert sft["metadata"]["source"] == "codex_rollout"
    assert sft["tools"] == ["terminal"]


def test_build_dpo_record_requires_redacted_preference():
    raw = {
        "redacted": True,
        "preference": {
            "prompt_messages": [{"role": "user", "content": "Restart Hermes"}],
            "chosen": "I will verify gateway and scheduled tasks.",
            "rejected": "It is probably fine.",
            "metadata": {"tags": ["restart"]},
        },
    }

    dpo = build_dpo_record(raw)

    assert dpo is not None
    assert dpo["schema"] == "hermes.operator.dpo.v1"
    assert dpo["chosen"].startswith("I will verify")


def test_build_dpo_record_refuses_unredacted_preference():
    with pytest.raises(ValueError, match="unredacted"):
        build_dpo_record({"redacted": False, "preference": {}})


def test_iter_harness_result_records_builds_trainable_sft(tmp_path):
    result_path = tmp_path / "harness-result.json"
    result_path.write_text(
        json.dumps({
            "id": "health",
            "status": "success",
            "service": "harness",
            "checks": [{"name": "health", "ok": True}],
            "path": "C:/Users/downl/.hermes/logs/harness.log",
        }),
        encoding="utf-8",
    )

    record = next(iter_harness_result_records([result_path]))
    redacted = CorpusRedactor(user_home=Path("C:/Users/downl")).redact_record(record)
    sft = build_sft_record(redacted)

    assert record["source"] == "harness_result"
    assert sft is not None
    assert sft["metadata"]["source"] == "harness_result"
    assert "harness" in sft["metadata"]["tags"]
    assert "<HERMES_HOME>" in sft["messages"][-1]["content"]


def test_iter_state_db_records_reads_messages_without_mutating(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            cwd TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            active INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, title) VALUES (?, ?, ?, ?)",
        ("s1", "cli", 1.0, "demo"),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("s1", "user", "hello", 2.0),
    )
    conn.commit()
    conn.close()

    records = list(iter_state_db_records(db_path))

    assert records[0]["session"]["id"] == "s1"
    assert records[0]["messages"][0]["content"] == "hello"


def test_write_jsonl_uses_utf8_newlines(tmp_path):
    output = tmp_path / "out.jsonl"

    count = write_jsonl([{"text": "ok"}], output)

    assert count == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {"text": "ok"}


def test_render_training_config_replaces_local_training_paths(tmp_path):
    template = tmp_path / "qlora.yaml"
    template.write_text(
        "\n".join([
            "base_model: placeholder/model",
            "datasets:",
            "  - path: old.jsonl",
            "output_dir: training/runs/old",
        ]),
        encoding="utf-8",
    )
    output = tmp_path / "local.yaml"

    render_config(
        template,
        output,
        base_model="H:\\models\\hf-checkpoint",
        sft_path=Path("training/corpora/sft.jsonl"),
        output_dir=Path("training/runs/new"),
    )

    text = output.read_text(encoding="utf-8")
    assert "base_model: H:/models/hf-checkpoint" in text
    assert "  - path: training/corpora/sft.jsonl" in text
    assert "output_dir: training/runs/new" in text
