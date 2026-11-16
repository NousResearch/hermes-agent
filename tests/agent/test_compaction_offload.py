"""Every compaction archives the tool results it drops, and the summarizer reads the pointers.

Threshold ``compress()``, the prepared pass (``compression.prepare_ahead``) and manual ``/compress``
(``compress_now``, also the cache-expiry and gateway-hygiene owners) all plan through
``_plan_compaction_window``: each tool result that plan shortens is first written byte-exact to
``$HERMES_HOME/cache/pruned/`` and its stub names the file, so the summary input carries the pointer.
A failed write keeps that result whole and the compaction still commits; a sandboxed backend (which
cannot ``read_file`` host paths) keeps today's plain stubs and writes nothing.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import prepared_compaction as pc
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from agent.context_compressor_prune_archive import ARCHIVE_POINTER_SEP
from agent.prepared_compaction import PreparedCompaction

_SUMMARY = "## Goal\nRead the files.\n## Progress\nRead them."


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return tmp_path


@pytest.fixture
def summarizer():
    """The summary LLM boundary; records every prompt it is sent."""
    prompts: list[str] = []
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = _SUMMARY

    def _call(**kw):
        prompts.append("\n".join(str(m.get("content")) for m in kw["messages"]))
        return response

    with patch("agent.context_compressor.call_llm", side_effect=_call):
        yield prompts


def _compressor(session_id="s1"):
    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        c = ContextCompressor(model="test/model", quiet_mode=True, protect_first_n=2, protect_last_n=4)
    c._session_id = session_id
    return c


def _round(i, content, name="read_file"):
    cid = f"call_{i}"
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps({"path": f"f{i}.py"})}}]},
        {"role": "tool", "tool_call_id": cid, "content": content}]


def _transcript(n=24, chars=12_000):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read them"}]
    for i in range(n):
        body = json.dumps({"content": f"{i}|" + chr(65 + i % 26) * chars}) if i % 2 else f"line {i}\n" + "x" * chars
        msgs += _round(i, body)
    return msgs + [{"role": "assistant", "content": "all read"}, {"role": "user", "content": "now summarize"}]


def _pointer_paths(text: str) -> list[Path]:
    return [Path(os.path.expanduser(part.split()[0].rstrip("]")))
            for part in text.split(ARCHIVE_POINTER_SEP)[1:]]


def _originals(msgs):
    return {m["tool_call_id"]: m["content"] for m in msgs if m.get("role") == "tool"}


def _archived(home) -> list[Path]:
    d = home / "cache" / "pruned"
    return sorted(d.iterdir()) if d.exists() else []


def _assert_pointers_name_byte_exact_originals(home, originals, rows):
    archived = 0
    for row in rows:
        if row.get("role") != "tool" or ARCHIVE_POINTER_SEP not in str(row.get("content")):
            continue
        (path,) = _pointer_paths(row["content"])
        assert path.parent == home / "cache" / "pruned"
        assert path.read_bytes() == originals[row["tool_call_id"]].encode("utf-8")
        archived += 1
    return archived


def test_threshold_compaction_archives_dropped_results_and_the_summarizer_reads_the_pointers(home, summarizer):
    msgs = _transcript()
    originals = _originals(msgs)
    c = _compressor()
    _working, pruned, start, end = c._plan_compaction_window(msgs)
    assert pruned and start < end

    compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    files = _archived(home)
    assert files
    (prompt,) = summarizer
    named = set(_pointer_paths(prompt))
    assert named and named <= set(files)
    for path in named:
        assert path.read_bytes() in {body.encode("utf-8") for body in originals.values()}
    # Stubs kept in the tail name their originals too.
    _assert_pointers_name_byte_exact_originals(home, originals, compressed)


def test_prepared_pass_summarizes_the_archived_stubs(home):
    msgs = _transcript()
    originals = _originals(msgs)
    c = _compressor()
    c.prepared_compaction = PreparedCompaction()
    seen: list = []

    def _summary(turns, focus_topic=None, memory_context="", bypass_cooldown=False):
        seen.extend(turns)
        return c._with_summary_prefix("prepared")

    c._generate_summary = _summary
    c.last_prompt_tokens = c.threshold_tokens - 1
    agent = SimpleNamespace(context_compressor=c, compression_enabled=True, api_mode="chat_completions")
    thread = pc.maybe_prepare(agent, msgs, origin="tool-batch")
    assert thread is not None
    thread.join(5.0)

    assert c.prepared_compaction._entry is not None
    assert _assert_pointers_name_byte_exact_originals(home, originals, seen)
    assert msgs == _transcript()  # the live transcript is untouched


def test_manual_compress_now_archives_what_it_drops(home, summarizer):
    from agent.conversation_compression_manual import compress_now, parse_compress_args
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("sid", "cli", model="test/model")
        db.append_messages_batch("sid", _transcript()[1:])
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                            quiet_mode=True, session_db=db, session_id="sid", skip_context_files=True,
                            skip_memory=True)
        agent._compression_feasibility_checked = True
        history = db.get_messages_as_conversation("sid")
        originals = _originals(history)

        result = compress_now(agent, history, parse_compress_args(""), system_message="")

        assert result.status == "compressed"
        assert summarizer and _pointer_paths(summarizer[-1])
        for path in _pointer_paths(summarizer[-1]):
            assert path.read_bytes() in {body.encode("utf-8") for body in originals.values()}
    finally:
        db.close()


def test_a_failed_archive_write_keeps_that_result_whole_and_compaction_still_commits(home, summarizer):
    msgs = _transcript()
    originals = _originals(msgs)
    real = __import__("tools.tool_result_storage", fromlist=["x"])._write_to_spillover
    failed: list[str] = []

    def flaky(content, filename, subdir="cache/spillover", exclusive=False):
        if subdir == "cache/pruned" and not failed:
            failed.append(content)
            return None
        return real(content, filename, subdir, exclusive)

    c = _compressor()
    with patch("tools.tool_result_storage._write_to_spillover", side_effect=flaky):
        working, _, _, _ = c._plan_compaction_window(msgs)
        kept_id = next(cid for cid, body in originals.items() if body == failed[0])
        assert next(m for m in working if m.get("tool_call_id") == kept_id)["content"] == failed[0]
        assert _assert_pointers_name_byte_exact_originals(home, originals, working)
        failed.clear()
        compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    (prompt,) = summarizer
    assert _pointer_paths(prompt)
    for path in _pointer_paths(prompt):
        assert path.read_bytes() != failed[0].encode("utf-8")


def test_a_large_result_the_prune_left_whole_is_archived_before_the_summary_replaces_it(home, summarizer):
    """Live on Nous: a 44 KB read sat inside the summary window, the prune (tail-budget pass) left it
    whole, and the summary replaced it with no copy on disk. The window's own results are saved first."""
    big = json.dumps({"content": "1|" + "Q" * 44_000})
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read it"}, *_round(0, big),
            {"role": "assistant", "content": "read"}, {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "again"},
            {"role": "assistant", "content": "again"}, {"role": "user", "content": "now"}]
    c = _compressor()
    c.protect_first_n, c.protect_last_n, c.tail_token_budget = 1, 2, 10_000  # the live run's tail budget
    assert c._prune_boundary(msgs, c.protect_last_n, c.tail_token_budget) <= 3  # the prune spares the read

    compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    (prompt,) = summarizer
    (path,) = _pointer_paths(prompt)
    assert path.read_bytes() == big.encode("utf-8")
    assert "Q" * 5_000 not in prompt


def test_lean_tail_demotion_archives_the_demoted_originals(home):
    from agent.context_compressor_tail import demote_stale_tail_tools

    msgs = _transcript(n=12, chars=3_000)
    originals = _originals(msgs)
    demoted, count = demote_stale_tail_tools(msgs, 2, "s1")

    assert count
    assert _assert_pointers_name_byte_exact_originals(home, originals, demoted) == count
    again, _ = demote_stale_tail_tools(demoted, 2, "s1")  # an archived stub is never archived twice
    assert again == demoted and len(_archived(home)) == count


@pytest.mark.parametrize("backend, archives", [("local", True), ("docker", False), ("ssh", False)])
def test_only_a_backend_that_reads_host_files_gets_archive_pointers(home, monkeypatch, summarizer, backend, archives):
    """The same compaction archives on the local backend and keeps plain stubs where the model cannot
    ``read_file`` a host path; either way the summary input is demoted."""
    monkeypatch.setenv("TERMINAL_ENV", backend)
    msgs = _transcript()
    c = _compressor()

    compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    (prompt,) = summarizer
    assert bool(_archived(home)) is archives
    assert (ARCHIVE_POINTER_SEP in prompt) is archives
    assert "x" * 5_000 not in prompt
