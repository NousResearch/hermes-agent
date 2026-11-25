"""A compaction gives no archive file and no summary list entry to a replaced row the session store holds.

The store archives every row a compaction replaces (``active=0``), so the original stays reachable through
``session_search`` and the retention sweep still reads the pointers it holds. A file for that row would be
named only by the summary's ``Archived tool output`` list: one path line per replaced result (hundreds per
compaction on a long tool-heavy session), with the same bytes stored twice. Results the compaction shortens in
the rows it keeps still get their file and pointer, and a session without a store keeps today's list.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from session_rules.turns import _DB_PERSISTED_MARKER
from tools.tool_result_storage import ARCHIVE_LIST_HEADING, pruned_archive_refs


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return tmp_path


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        c = ContextCompressor(model="test/model", quiet_mode=True, protect_first_n=2, protect_last_n=4)
    c._session_id = "s1"
    return c


def _transcript(n=24, chars=12_000, stored=lambda i: True):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read them"}]
    for i in range(n):
        cid = f"call_{i}"
        call = {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]}
        result = {"role": "tool", "tool_call_id": cid, "content": f"NEEDLE-{i:02d}\n" + chr(65 + i % 26) * chars}
        for row in (call, result):
            if stored(i):
                row[_DB_PERSISTED_MARKER] = True
        msgs += [call, result]
    return msgs + [{"role": "assistant", "content": "all read"}, {"role": "user", "content": "now summarize"}]


def _no_pointer_summary():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nRead the files.\n## Progress\nRead them all."
    return patch("agent.context_compressor.call_llm", return_value=response)


def _archived(home) -> set[str]:
    d = home / "cache" / "pruned"
    return {f"cache/pruned/{p.name}" for p in d.iterdir()} if d.exists() else set()


def _summary(rows) -> str:
    from session_rules.turns import is_compaction_summary_message

    (row,) = [m for m in rows if is_compaction_summary_message(m)]
    return row["content"]


def _needle_of(home, ref: str) -> int:
    return int((home / ref).read_text(encoding="utf-8").split("NEEDLE-", 1)[1][:2])


def test_stored_replaced_rows_get_no_file_and_the_summary_lists_no_paths(home):
    """Before the fix the summary listed one path per replaced result and each had a file the store duplicated."""
    c = _compressor()
    with _no_pointer_summary():
        out = c.compress(_transcript(), current_tokens=c.threshold_tokens + 1)

    summary = _summary(out)
    assert ARCHIVE_LIST_HEADING not in summary and not pruned_archive_refs([{"content": summary}])
    kept = [m for m in out if not m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    # Every file that exists is named by a row the chat keeps (a shortened kept result's own pointer).
    assert _archived(home) == pruned_archive_refs(kept)


def test_a_stored_row_that_already_points_at_an_archive_adds_no_list_entry(home):
    """An earlier prune left a stub naming its file on a stored row; the store keeps that row (and the reference
    the sweep reads) when the summary replaces it, so the summary must not re-list the path."""
    msgs = _transcript()
    stub = "[read_file] read f3.py (12,010 chars) — full result: ~/.hermes/cache/pruned/s1_earlier_read_file_00.txt"
    msgs[2 * 3 + 3]["content"] = stub  # call_3's result, already shortened and stored by a proactive prune
    c = _compressor()
    with _no_pointer_summary():
        out = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert stub not in json.dumps(out)  # the summary replaced it
    assert "s1_earlier_read_file_00" not in _summary(out)


def test_without_a_store_the_replaced_results_are_still_archived_and_listed(home):
    c = _compressor()
    with _no_pointer_summary():
        out = c.compress(_transcript(stored=lambda i: False), current_tokens=c.threshold_tokens + 1)

    listed = pruned_archive_refs([{"content": _summary(out)}])
    assert listed and listed <= _archived(home)


def test_a_mixed_window_lists_only_the_unstored_results(home):
    """Rows the store does not hold yet (a turn not flushed when compaction runs) keep their file and entry."""
    unstored = set(range(0, 24, 3))
    c = _compressor()
    with _no_pointer_summary():
        out = c.compress(_transcript(stored=lambda i: i not in unstored), current_tokens=c.threshold_tokens + 1)

    listed = pruned_archive_refs([{"content": _summary(out)}])
    assert listed and {_needle_of(home, ref) for ref in listed} <= unstored
    kept = [m for m in out if not m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert _archived(home) == listed | pruned_archive_refs(kept)


def test_a_real_store_commit_keeps_each_replaced_result_searchable_without_a_file(home):
    """Real AIAgent, real SQLite and the in-place commit: the replaced tool result's exact text comes back from
    session_search, so dropping its file and list line loses nothing the model can reach."""
    import os

    from agent.conversation_compression import compress_context
    from persistence.conversation.messages import ConversationRequest
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session
    from tools.session_search_tool import session_search

    sid = "20261125_000000_store"
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), \
            open_persistence(profile_scope(home=Path(home))) as db, _no_pointer_summary(), \
            patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        seed_session(db, sid, "cli", model="test/model")
        from run_agent import AIAgent

        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, persistence=db, session_id=sid, skip_context_files=True, skip_memory=True)
        agent.compression_in_place = True
        agent.context_compressor.protect_first_n, agent.context_compressor.protect_last_n = 2, 4
        msgs = _transcript(stored=lambda i: False)[1:]
        agent._persist_session(msgs, conversation_history=None)  # stamps each row it writes
        assert all(m.get(_DB_PERSISTED_MARKER) for m in msgs)
        compressed, _ = compress_context(agent, msgs, approx_tokens=agent.context_compressor.threshold_tokens + 1,
                                         system_message="sys", force=True)

        active = [m.as_openai_message() for m in db.messages.read_conversation(sid, ConversationRequest())]
        summary = _summary(active)
        assert ARCHIVE_LIST_HEADING not in summary
        replaced = sorted({int(m["content"][7:9]) for m in msgs if m.get("role") == "tool"}
                          - {int(m["content"][7:9]) for m in compressed
                             if m.get("role") == "tool" and str(m.get("content")).startswith("NEEDLE-")})
        assert replaced, "the window replaced no tool result whole"
        assert not _archived(home) - pruned_archive_refs(active)  # no file only a dropped row would name
        hit = json.loads(session_search(query=f"NEEDLE-{replaced[0]:02d}", role_filter="tool", persistence=db))
        assert f"NEEDLE-{replaced[0]:02d}" in json.dumps(hit)
