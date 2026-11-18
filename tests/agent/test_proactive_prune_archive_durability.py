"""Prune archives stay reachable (#156081 Greptile round): an uncertain store commit keeps the files the
transcript may already name, and a budget smaller than the line cap still pages every archive character."""

import json
import os
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return tmp_path


def test_an_uncertain_commit_keeps_the_archives(home):
    """A worker may commit and then lose the reply: the files its stubs point at must survive."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from persistence.conversation.messages import ConversationRequest
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import conversation
    from tests.fixtures.storage_seed import seed_messages, seed_session
    db = open_persistence(profile_scope(home=home), eager=True)
    try:
        seed_session(db, "s1", "cli", model="m")
        for i in range(6):
            seed_messages(db, "s1", {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "content": chr(65 + i) * 30_000, "tool_call_id": f"c{i}"})
        live = [{"role": "system", "content": "sys"}] + conversation(db, "s1", ConversationRequest(include_row_ids=True))
        for m in live[1:]:
            m[_DB_PERSISTED_MARKER] = True
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="m", quiet_mode=True, protect_first_n=1, protect_last_n=2)
        c.bind_session_state(db, "s1", persistence=db)
        with patch.object(c, "_rewrite_pruned_rows_in_place", side_effect=RuntimeError("reply lost")):
            out, n = c.prune_tool_results_only(live, current_tokens=100_000, cold_cache=True)
        assert (out, n) == (live, 0)
        assert any((home / "cache" / "pruned").iterdir())
    finally:
        db.close()


def test_archive_pages_fit_a_small_read_budget(home):
    from tools.file_tools import read_file_tool
    from tools.tool_result_storage import write_pruned_original

    body = "".join(f"<{i:06d}>" for i in range(5_000))
    _host, stub = write_pruned_original(body, "s1", "terminal", "call_1")
    seen, offset = [], 1
    with patch("tools.file_tools._get_max_read_chars", return_value=1_000):
        for _ in range(500):
            page = json.loads(read_file_tool(os.path.expanduser(stub), offset=offset, limit=1))
            seen += [line.split("|", 1)[1] for line in page["content"].split("\n") if line]
            nxt = page.get("next_offset") or (offset + 1 if page.get("truncated") else None)
            if not nxt:
                break
            offset = nxt
    assert "".join(seen) == body
