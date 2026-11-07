"""Pre-summarize lossless offload of the compacted region (A1, deepagents pattern).

The state.db archive written at commit stays the primary store; the per-session
JSONL is an independent greppable copy written BEFORE the summarizer call, named
by the summary's Context Recovery footer. A failed offload must never block
compaction, and ``compression.offload_evicted_history`` is the behaviour gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_init import _parse_compression_config
from agent.compaction_offload import MAX_OFFLOAD_FILES_PER_SESSION, offload_evicted_region
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from hermes_cli.config import DEFAULT_CONFIG


def _compressor(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="test/model", quiet_mode=True, **kwargs)


def _transcript(count=40):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + ("lorem ipsum " * 300)}
        for i in range(count)
    ]


def test_offload_roundtrip_is_lossless(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    turns = [
        {"role": "user", "content": "check PR #12345678 and file agent/foo.py"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "raw é output ✓"},
    ]
    path = offload_evicted_region("sess-lossless", turns)
    assert path
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["offload_meta"]["session_id"] == "sess-lossless"
    assert header["offload_meta"]["message_count"] == 3
    assert [json.loads(line) for line in lines[1:]] == turns


def test_offload_prunes_to_newest_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for _ in range(MAX_OFFLOAD_FILES_PER_SESSION + 4):
        assert offload_evicted_region("sess-prune", [{"role": "user", "content": "x"}])
    directory = next((tmp_path / "offloads").glob("sess-prune*"))
    remaining = sorted(directory.glob("evicted-*.jsonl"))
    assert len(remaining) == MAX_OFFLOAD_FILES_PER_SESSION


def test_offload_failure_returns_none_and_never_raises(tmp_path, monkeypatch):
    # HERMES_HOME points AT a regular file: mkdir must fail, and the failure must
    # land as None rather than escaping into compress().
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    monkeypatch.setenv("HERMES_HOME", str(blocker))
    assert offload_evicted_region("sess-fail", [{"role": "user", "content": "x"}]) is None


def test_compress_offloads_region_before_summary(tmp_path, monkeypatch):
    c = _compressor(tmp_path, monkeypatch)
    c.bind_session_state(session_db=None, session_id="OFFLOAD_E2E")
    messages = _transcript()
    c._call_summary_llm = lambda prompt, prompt_started_at: "## Historical Task Snapshot\nNone\n\nBody."
    compressed = c.compress(messages)

    files = list((tmp_path / "offloads").glob("OFFLOAD_E2E*/evicted-*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()[1:]]
    assert rows, "offload must capture the compacted region"
    originals = {json.dumps(m, sort_keys=True) for m in messages}
    assert all(json.dumps(row, sort_keys=True) in originals for row in rows), "offloaded rows verbatim from the transcript"
    summary_msg = next(m for m in compressed if m.get(COMPRESSED_SUMMARY_METADATA_KEY))
    assert "## Context Recovery" in summary_msg["content"]
    assert files[0].name in summary_msg["content"], "the footer names the offload file"


def test_offload_gate_disabled_writes_no_file(tmp_path, monkeypatch):
    c = _compressor(tmp_path, monkeypatch, offload_evicted=False)
    c.bind_session_state(session_db=None, session_id="OFFLOAD_OFF")
    c._call_summary_llm = lambda prompt, prompt_started_at: "## Historical Task Snapshot\nNone\n\nBody."
    c.compress(_transcript())
    assert not list((tmp_path / "offloads").glob("OFFLOAD_OFF*"))


def test_offload_gate_reads_compression_config_key():
    agent = SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)
    # Shipped default is ON, matching DEFAULT_CONFIG (config-load-failure fallback contract).
    assert DEFAULT_CONFIG["compression"]["offload_evicted_history"] is True
    assert _parse_compression_config(agent, {}).offload_evicted is True
    off = _parse_compression_config(agent, {"compression": {"offload_evicted_history": False}})
    assert off.offload_evicted is False
