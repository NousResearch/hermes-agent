"""A compaction archives each dropped tool result exactly once, after its summary exists, and the
committed chat names every archive whatever the generated summary says.

Planning (``_plan_compaction_window``) only decides whether work can start, so it writes nothing; a
prepared pass (refused or completed) summarizes the originals and writes nothing either: the compaction
that splices its summary archives the window once.
The summary is model text and may drop a pointer, so the committed summary row lists every archive the
replaced rows named; the retention sweep then keeps those files.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import prepared_compaction as pc
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from agent.prepared_compaction import PreparedCompaction


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


def _transcript(n=24, chars=12_000):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read them"}]
    for i in range(n):
        cid = f"call_{i}"
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {
                     "name": "read_file", "arguments": json.dumps({"path": f"f{i}.py"})}}]},
                 {"role": "tool", "tool_call_id": cid, "content": f"line {i}\n" + chr(65 + i % 26) * chars}]
    return msgs + [{"role": "assistant", "content": "all read"}, {"role": "user", "content": "now summarize"}]


def _archived(home) -> list[Path]:
    d = home / "cache" / "pruned"
    return sorted(d.iterdir()) if d.exists() else []


def _no_pointer_summary():
    """A summarizer that, like a real model may, names none of the archive paths it was shown."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nRead the files.\n## Progress\nRead them all."
    return patch("agent.context_compressor.call_llm", return_value=response)


def test_planning_and_the_prepared_pass_write_nothing_and_the_splice_archives_each_result_once(home):
    msgs = _transcript()
    c = _compressor()
    for _ in range(3):  # every tool batch plans; none of them is admitted work
        c._plan_compaction_window(msgs)
    assert _archived(home) == []

    c.prepared_compaction = PreparedCompaction()
    c._generate_summary = lambda turns, **kw: c._with_summary_prefix("prepared")
    c.last_prompt_tokens = c.threshold_tokens - 1
    agent = SimpleNamespace(context_compressor=c, compression_enabled=True, api_mode="chat_completions")
    with patch.object(pc._slots, "acquire", return_value=False):  # every summary slot busy: refused
        assert pc.maybe_prepare(agent, msgs, origin="tool-batch") is None
    assert _archived(home) == []

    thread = pc.maybe_prepare(agent, msgs, origin="tool-batch")
    thread.join(5.0)
    assert c.prepared_compaction._entry is not None and _archived(home) == []

    compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    bodies = [p.read_bytes() for p in _archived(home)]
    assert bodies and len(bodies) == len(set(bodies))  # no tool result was written twice


def test_the_committed_summary_names_every_archive_the_generated_text_omits_and_the_sweep_keeps_them(home):
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session
    from tools.tool_result_storage import cleanup_spillover_cache

    msgs = _transcript()
    c = _compressor()
    with _no_pointer_summary():
        compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)
    files = _archived(home)
    assert files

    committed = "\n".join(str(m.get("content")) for m in compressed)
    for path in files:
        assert f"cache/pruned/{path.name}" in committed
    (summary_row,) = [m for m in compressed if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert "Archived tool output" in summary_row["content"]

    db = open_persistence(profile_scope(home=home), eager=True)
    try:
        seed_session(db, "s1", "cli")
        seed_messages(db, "s1", *(m for m in compressed if m.get("role") != "system"))
    finally:
        db.close()
    old = 1_000_000_000
    for path in files:
        os.utime(path, (old, old))
    assert cleanup_spillover_cache(max_age_hours=1) == 0
    assert _archived(home) == files


def _next_batch(start, n=24, chars=12_000):
    rows = _transcript(n, chars)[2:-2]
    for row in rows:  # fresh call ids so each batch archives new files
        for call in row.get("tool_calls") or ():
            call["id"] = f"b{start}_{call['id']}"
        if row.get("tool_call_id"):
            row["tool_call_id"] = f"b{start}_{row['tool_call_id']}"
    return rows + [{"role": "assistant", "content": f"batch {start} read"}, {"role": "user", "content": f"go on {start}"}]


@pytest.mark.parametrize("stored", [True, False])
def test_a_stored_handoff_keeps_its_own_list_and_an_unstored_one_carries_it(home, stored):
    """A stored handoff's archive list stays on its state.db row, so the next summary does not repeat it
    (before the fix each compaction re-listed every earlier path and the summary grew without bound). With no
    session store the old summary is the list's only copy, so its paths carry forward instead of vanishing."""
    from session_rules.turns import _DB_PERSISTED_MARKER
    from tools.tool_result_storage import ARCHIVE_LIST_HEADING, pruned_archive_refs
    msgs = _transcript()
    previous: set[str] = set()
    for rnd in range(3):
        c = _compressor()  # a fresh process per round: the prior summary comes only from the handoff row
        with _no_pointer_summary() as llm:
            msgs = c.compress(msgs, current_tokens=c.threshold_tokens + 1)
        request = json.dumps(llm.call_args.kwargs.get("messages"), ensure_ascii=False)
        assert (rnd == 0 or "Read them all." in request) and ARCHIVE_LIST_HEADING not in request, rnd
        (summary_row,) = [m for m in msgs if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
        listed = pruned_archive_refs([summary_row])
        assert listed and (not listed & previous if stored else previous <= listed), rnd
        previous = listed
        if stored:
            summary_row[_DB_PERSISTED_MARKER] = True  # what the commit stamps on the live handoff
        msgs = msgs + _next_batch(rnd + 1)


def test_an_unstored_handoff_in_the_protected_head_keeps_its_archives_listed():
    """Assembly drops a standalone handoff from the protected head; its archive paths must then move into
    the new list rather than be counted as "kept" by a row the committed transcript no longer has."""
    from agent.context_compressor import _HandoffScan
    from tools.tool_result_storage import ARCHIVE_LIST_HEADING, pruned_archive_refs
    c = _compressor()
    old = {"role": "user", COMPRESSED_SUMMARY_METADATA_KEY: True, "content": c._with_summary_prefix("old")
           + f"\n\n{ARCHIVE_LIST_HEADING}\n- ~/.hermes/cache/pruned/old.txt"}
    rows = [old, {"role": "assistant", "content": "work"}, {"role": "user", "content": "continue"}]
    scan = _HandoffScan(turns_to_summarize=[], summary_indices={0}, tail_start=2,
                        previous_summary_before=None, has_user_turn_before=True)
    out = c._assemble_with_archive_list(rows, 1, 2, scan, c._with_summary_prefix("new"))
    assert pruned_archive_refs(out) == {"cache/pruned/old.txt"}


def test_stripping_removes_only_generated_list_blocks_not_a_quoted_heading():
    from tools.tool_result_storage import ARCHIVE_LIST_HEADING, strip_archive_list
    block = f"\n\n{ARCHIVE_LIST_HEADING}\n- ~/.hermes/cache/pruned/a.txt\n- ~/.hermes/cache/pruned/b.json"
    quoted = f'## User said\n"why does {ARCHIVE_LIST_HEADING} keep growing?"\n## Recovery\nread_file the paths'
    assert strip_archive_list(quoted + block + block) == quoted
