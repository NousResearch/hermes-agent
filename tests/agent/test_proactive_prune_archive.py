"""The proactive prune loses no information and never shortens the turn in progress.

Every result it shortens is first written byte-exact to ``$HERMES_HOME/cache/pruned/`` and the stub
names that file; a result whose write fails stays whole. The archive shares the spillover sweeper:
kept while state.db points at it, retired with its session.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB
from tools.tool_result_storage import cleanup_spillover_cache


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _compressor(db=None):
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=4,
                              proactive_prune_tokens=1_000, proactive_prune_min_result_chars=1_000,
                              proactive_prune_min_reclaim_tokens=1_000)
    if db is not None:
        c.bind_session_state(session_db=db, session_id="s1")
    return c


def _round(i, content, name="read_file"):
    cid = f"call_{i}"
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps({"path": f"f{i}.py"})}}]},
        {"role": "tool", "tool_call_id": cid, "content": content}]


def _transcript(n=8, chars=12_000):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read them"}]
    for i in range(n):
        body = json.dumps({"content": f"{i}|" + chr(65 + i) * chars}) if i % 2 else f"line {i}\n" + "x" * chars
        msgs += _round(i, body)
    return msgs


def _archive_path(stub: str) -> Path:
    return Path(os.path.expanduser(stub.rsplit("full result: ", 1)[1]))


def test_every_shortened_result_is_archived_byte_exact_and_named_by_its_stub(home):
    msgs = _transcript()
    pruned, count = _compressor().prune_tool_results_only(msgs, current_tokens=100_000)

    assert count
    changed = [(o, n) for o, n in zip(msgs, pruned) if o["content"] != n["content"]]
    assert len(changed) == count
    for original, stub in changed:
        path = _archive_path(stub["content"])
        assert path.parent == home / "cache" / "pruned"
        assert path.read_bytes() == original["content"].encode("utf-8")
        assert path.suffix == (".json" if original["content"].startswith("{") else ".txt")
        assert "_read_file_" in path.name


def test_a_failed_archive_write_leaves_that_result_unpruned(home):
    msgs = _transcript()
    real = __import__("tools.tool_result_storage", fromlist=["x"])._write_to_spillover
    failed = {"n": 0}

    def flaky(content, filename, subdir="cache/spillover", exclusive=False):
        if subdir == "cache/pruned" and failed["n"] == 0 and filename.endswith(".txt"):
            failed["n"] += 1
            return None
        return real(content, filename, subdir, exclusive)

    with patch("tools.tool_result_storage._write_to_spillover", side_effect=flaky):
        pruned, count = _compressor().prune_tool_results_only(msgs, current_tokens=100_000)

    assert failed["n"] == 1 and count
    kept = [o for o, n in zip(msgs, pruned) if o["role"] == "tool" and o["content"] == n["content"]
            and len(o["content"]) > 1_000 and not o["content"].startswith("{")]
    assert kept, "the result whose archive failed must keep its full body"
    for original, stub in zip(msgs, pruned):
        if original["content"] != stub["content"]:
            assert _archive_path(stub["content"]).read_bytes() == original["content"].encode("utf-8")


def test_the_running_turn_is_never_shortened(home):
    """Post-tool site in a long tool loop: everything from this turn's user row stays whole."""
    history = _transcript(6) + [{"role": "assistant", "content": "done"}]
    turn_start = len(history)
    msgs = history + [{"role": "user", "content": "now read more"}]
    for i in range(6, 30):
        msgs += _round(i, f"line {i}\n" + "y" * 12_000)

    pruned, count = _compressor().prune_tool_results_only(msgs, current_tokens=200_000, turn_start=turn_start)

    assert count
    assert pruned[turn_start:] == msgs[turn_start:]
    assert any(o["content"] != n["content"] for o, n in zip(msgs[:turn_start], pruned[:turn_start]))


def test_the_first_turn_is_never_shortened(home):
    """The loop's transcript carries no system row, so a session's first turn starts at index 0."""
    msgs = [{"role": "user", "content": "read them all"}]
    for i in range(30):
        msgs += _round(i, f"line {i}\n" + "y" * 12_000)

    pruned, count = _compressor().prune_tool_results_only(msgs, current_tokens=200_000, turn_start=0)

    assert (pruned, count) == (msgs, 0)


def test_pruned_archive_follows_its_session(home):
    """Referenced archives survive the age sweep; deleting the session retires them."""
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("s1", "cli")
        msgs = _transcript()
        db.append_messages_batch("s1", msgs[1:])
        stored = [{"role": "system", "content": "sys"},
                  *db.get_messages_as_conversation("s1", include_row_ids=True)]
        from agent.context_compressor import _DB_PERSISTED_MARKER
        for m in stored[1:]:
            m[_DB_PERSISTED_MARKER] = True
        _, count = _compressor(db).prune_tool_results_only(stored, current_tokens=100_000)
        assert count
        files = sorted((home / "cache" / "pruned").iterdir())
        assert len(files) == count
        stale = time.time() - 400 * 86_400
        for f in files:
            os.utime(f, (stale, stale))

        assert cleanup_spillover_cache(max_age_hours=24) == 0  # state.db still points at each
        assert all(f.exists() for f in files)
        db.delete_session("s1")
        assert not any(f.exists() for f in files)
    finally:
        db.close()


def test_read_file_pages_a_json_archive_by_the_text_it_wraps(home):
    """A JSON result is archived byte-exact (one long line); read_file must still reach every line of it,
    not cut it at the per-line cap."""
    from tools.file_tools import read_file_tool

    body = "\n".join(f"{i}|line {i} " + "z" * 60 for i in range(1, 200)) + "\n198|IMPORTANT: the fact."
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(8):
        msgs += _round(i, json.dumps({"content": body.replace("fact", f"fact {i}"), "total_lines": 199}))
    pruned, _ = _compressor().prune_tool_results_only(msgs, current_tokens=100_000)
    stub = next(n["content"] for o, n in zip(msgs, pruned) if o["content"] != n["content"])

    page = json.loads(read_file_tool(str(_archive_path(stub)), offset=1, limit=500))
    assert "IMPORTANT: the fact" in page["content"]


def test_a_repeated_tool_call_id_never_overwrites_an_earlier_original(home):
    """Resumed histories can repeat a tool_call_id: each original gets its own file."""
    from tools.tool_result_storage import write_pruned_original

    first = write_pruned_original("first body", "s1", "terminal", "dup")
    second = write_pruned_original("second body", "s1", "terminal", "dup")

    assert first and second and first[0] != second[0]
    assert Path(first[0]).read_text() == "first body" and Path(second[0]).read_text() == "second body"


def test_a_sandboxed_backend_keeps_the_result_whole(home):
    """The model on a remote backend reads its own copy; with no environment to sync it the stub could
    name a file the model cannot open, so the result is not shortened."""
    msgs = _transcript()
    with patch("tools.credential_files.to_agent_visible_cache_path", side_effect=lambda p: "/root/.hermes/x"):
        pruned, count = _compressor().prune_tool_results_only(msgs, current_tokens=100_000)

    assert (pruned, count) == (msgs, 0)
    assert not (home / "cache" / "pruned").exists() or not any((home / "cache" / "pruned").iterdir())


def test_a_plugin_engine_without_the_builtins_fields_still_prunes():
    """The store-row gate is the built-in's own; a plugin engine is called without those fields."""
    from types import SimpleNamespace

    from agent.turn_preflight import _proactive_tool_result_prune

    class Engine:
        def prune_tool_results_only(self, messages, current_tokens=None):
            return messages[:1], 1

    agent = SimpleNamespace(compression_checkpoint_required=False, session_id="s1")
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    with patch("agent.compaction_events.publish_prune"):
        assert _proactive_tool_result_prune(agent, Engine(), msgs, 100_000) == msgs[:1]
