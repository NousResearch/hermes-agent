"""Opt-in structured pruning preserves evidence, scope and archive-before-commit."""

import concurrent.futures
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.tool_result_storage import ARCHIVE_POINTER_SEP


def _round(number, name, body):
    call_id = f"call-{number}"
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def _body(label="fixture"):
    return json.dumps({"exit_code": 1, "output": (
        f"FAILED {label}: expected 403 got 200\n"
        + "diagnostic heartbeat\r\n" * 600
        + "DO NOT RESTART THE DATABASE\nRollback migration: 016\n"),
        "stderr": "assertion failed", "timed_out": False})


def _invalid_json_field(field):
    return _body()[:-1] + "," + field + "}"


def _history(label="fixture"):
    rows = [{"role": "system", "content": "immutable policy"}, {"role": "user", "content": "inspect"}]
    for i in range(8):
        rows += _round(i, "terminal", _body(f"{label}-{i}"))
    rows += [{"role": "assistant", "content": "Await instruction."}]
    return rows


def _compressor():
    return ContextCompressor(
        model="fixture", config_context_length=100_000, quiet_mode=True,
        protect_first_n=2, protect_last_n=4, proactive_prune_tokens=1000,
        proactive_prune_min_result_chars=1000, proactive_prune_min_reclaim_tokens=1000,
        structured_tool_pruning=True,
    )


def _projected_body(content):
    return json.loads(content.split(ARCHIVE_POINTER_SEP)[0])


def test_preserves_all_evidence_and_archives_the_exact_original(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    history = _history()
    snapshot = copy.deepcopy(history)
    with patch("agent.context_compressor.call_llm", side_effect=AssertionError("No LLM")):
        pruned, count = _compressor().prune_tool_results_only(history, current_tokens=30_000)
    assert count > 0 and history == snapshot
    assert pruned[0] == history[0] and pruned[-4:] == history[-4:]
    for original, shortened in zip(history, pruned):
        if original == shortened:
            continue
        body = _projected_body(shortened["content"])
        expanded = "".join(run["line"] * run["repeat"] for run in body["output_runs"])
        source = json.loads(original["content"])
        assert expanded == source.pop("output")
        assert body["result"] == source
        assert "expected 403 got 200" in expanded
        assert "DO NOT RESTART THE DATABASE" in expanded
        assert "Rollback migration: 016" in expanded
        archive = Path(os.path.expanduser(shortened["content"].split(ARCHIVE_POINTER_SEP)[1]))
        assert archive.read_text() == original["content"]


@pytest.mark.parametrize("name,body", [
    ("read_file", "Rollback migration must remain 016.\n" + "sample\n" * 1000),
    ("mcp_support_ticket", json.dumps({"restriction": "DO NOT RESTART", "noise": ["sample"] * 1000})),
    ("terminal", json.dumps({"exit_code": 1, "output": "\n".join(str(i) for i in range(5000))})),
    ("terminal", _invalid_json_field('"output":"conflicting output"')),
    ("terminal", json.dumps({"exit_code": True, "output": "sample\n" * 1000})),
    ("terminal", _invalid_json_field('"elapsed":NaN')),
    ("terminal", _invalid_json_field('"elapsed":1e999')),
    ("terminal", _invalid_json_field('"value":1.234567890123456789')),
])
def test_unknown_or_ambiguous_content_stays_whole(name, body):
    history = [{"role": "user", "content": "inspect"}] + _round(0, name, body)
    history += [{"role": "assistant", "content": "Result consumed."}]
    projected, count = _compressor()._prune_old_tool_results(history, protect_tail_count=0)
    assert count == 0 and projected == history


def test_the_negative_case_window_actually_admits_a_valid_projection():
    history = [{"role": "user", "content": "inspect"}] + _round(0, "terminal", _body())
    history += [{"role": "assistant", "content": "Result consumed."}]
    projected, count = _compressor()._prune_old_tool_results(history, protect_tail_count=0)
    assert count == 1 and projected[2] != history[2]


@pytest.mark.parametrize("field", ["output", "stderr"])
def test_projection_remains_utf8_transport_safe_with_escaped_surrogates(field):
    source = json.loads(_body())
    source[field] = "diagnostic \ud800\n" * 600
    history = [{"role": "user", "content": "inspect"}] + _round(0, "terminal", json.dumps(source))
    history += [{"role": "assistant", "content": "Result consumed."}]
    projected, count = _compressor()._prune_old_tool_results(history, protect_tail_count=0)
    assert count == 1
    content = projected[2]["content"]
    assert content.encode("utf-8").decode("utf-8") == content
    decoded = json.loads(content)
    assert decoded["result"] == {k: v for k, v in source.items() if k != "output"}
    assert "".join(run["line"] * run["repeat"] for run in decoded["output_runs"]) == source["output"]


def test_distinct_results_are_not_dropped_on_a_digest_collision():
    history = _round(0, "terminal", _body("first assertion")) + _round(1, "terminal", _body("different restriction"))
    snapshot = copy.deepcopy(history)
    with patch("agent.context_compressor.hashlib.md5", return_value=SimpleNamespace(hexdigest=lambda: "0" * 32)):
        assert ContextCompressor._dedupe_tool_results(history) == 0
    assert history == snapshot


def test_running_turn_stays_whole(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    history = _history()
    turn_start = len(history)
    history += [{"role": "user", "content": "inspect current failure"}]
    for index in range(8, 18):
        history += _round(index, "terminal", _body(str(index)))
    pruned, count = _compressor().prune_tool_results_only(history, current_tokens=60_000, turn_start=turn_start)
    assert count and pruned[turn_start:] == history[turn_start:]


def test_archive_failure_preserves_originals(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    history = _history()
    with patch("tools.tool_result_storage.write_pruned_original", return_value=None):
        projected, count = _compressor().prune_tool_results_only(history, current_tokens=30_000)
    assert count == 0 and projected is history


def test_reused_ids_do_not_turn_an_unknown_tool_into_a_terminal():
    history = [{"role": "user", "content": "inspect"}]
    history += _round(0, "mcp_support_ticket", _body("unknown"))
    history += _round(0, "terminal", _body("terminal"))
    history += [{"role": "assistant", "content": "Both results were inspected."}]
    projected, count = _compressor()._prune_old_tool_results(history, protect_tail_count=0)
    assert projected[2] == history[2]
    assert count == 1 and projected[4] != history[4]


def test_archive_is_recoverable_through_the_worker_read_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    from tools.file_tools import read_file_tool

    pruned, count = _compressor().prune_tool_results_only(_history(), current_tokens=30_000)
    assert count
    content = next(m["content"] for m in pruned if ARCHIVE_POINTER_SEP in m.get("content", ""))
    archive = os.path.expanduser(content.split(ARCHIVE_POINTER_SEP)[1])
    first = json.loads(read_file_tool(archive, offset=1, limit=30))
    last = json.loads(read_file_tool(archive, offset=580, limit=100))
    assert "expected 403 got 200" in first["content"]
    assert "DO NOT RESTART THE DATABASE" in last["content"]
    assert "Rollback migration: 016" in last["content"]


def test_projection_is_idempotent_and_preserves_pending_parallel_results(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    history = _history()
    compressor = _compressor()
    pruned, count = compressor.prune_tool_results_only(history, current_tokens=30_000)
    assert count
    again, count = compressor._prune_old_tool_results(pruned, protect_tail_count=4)
    assert count == 0 and again == pruned
    pending = _round(20, "terminal", _body("unread"))
    pending[0]["tool_calls"] += _round(21, "terminal", "")[0]["tool_calls"]
    pending += [_round(21, "terminal", _body("unread-other"))[1]]
    projected, _ = compressor.prune_tool_results_only(history + pending, current_tokens=30_000)
    assert projected[-3:] == pending


def test_three_profiles_never_mix_archive_paths_or_content(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    def worker(label):
        home = tmp_path / label
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            history = _history(label)
            pruned, count = _compressor().prune_tool_results_only(history, current_tokens=30_000)
            assert count
            for original, shortened in zip(history, pruned):
                if original == shortened:
                    continue
                path = Path(os.path.expanduser(shortened["content"].split(ARCHIVE_POINTER_SEP)[1]))
                assert path.parent == home / "cache/pruned"
                assert json.loads(path.read_text())["output"].startswith(f"FAILED {label}-")
            return count
        finally:
            reset_hermes_home_override(token)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        assert all(pool.map(worker, ["worker-a", "worker-b", "worker-c"]))


def test_config_loader_and_live_reload_share_opt_in_and_unset_semantics():
    from agent.agent_init import _parse_compression_config
    from tui_gateway.session_compression import _apply_live_compression_config

    agent = SimpleNamespace(model="fixture", provider="openai", api_mode="chat_completions")
    settings = _parse_compression_config(agent, {"compression": {"structured_tool_pruning": True}})
    assert settings.structured_tool_pruning is True
    assert _parse_compression_config(agent, {}).structured_tool_pruning is False
    agent.context_compressor = _compressor()
    _apply_live_compression_config(agent, {"compression": {"structured_tool_pruning": True}})
    assert agent.context_compressor.structured_tool_pruning is True
    _apply_live_compression_config(agent, {})
    assert agent.context_compressor.structured_tool_pruning is False


@pytest.mark.platforms("posix")
def test_archived_original_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    from tools.tool_result_storage import write_pruned_original

    path, _ = write_pruned_original(_body(), "session", "terminal", "call")
    assert Path(path).stat().st_mode & 0o777 == 0o600
