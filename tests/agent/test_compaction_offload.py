"""Every compaction summarizes the ORIGINAL tool results first, then archives what it dropped.

Threshold ``compress()``, the prepared pass (``compression.prepare_ahead``) and manual ``/compress``
(``compress_now``, also the cache-expiry and gateway-hygiene owners) all plan through the write-free
``_plan_compaction_window``; the summarizer reads the window's original results (``summary_view``), and only
once a summary exists is each tool result the compaction removed or shortened written byte-exact to
``$HERMES_HOME/cache/pruned/`` with its stub naming the file (and the summary row listing it). A failed or
aborted summary writes nothing; a failed write keeps that result whole and the compaction still commits; a
sandboxed backend (which cannot ``read_file`` host paths) keeps plain stubs and writes nothing.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import prepared_compaction as pc
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from tools.tool_result_storage import ARCHIVE_POINTER_SEP
from agent.prepared_compaction import PreparedCompaction
from agent.conversation_compression_manual import CompressRequest, compress_now

_SUMMARY = "## Goal\nRead the files.\n## Progress\nRead them."


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return tmp_path


@pytest.fixture
def summarizer():
    """The summary LLM boundary; records every prompt it is sent."""
    class _Prompts(list):
        archived_at_call: list

    prompts = _Prompts()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = _SUMMARY

    def _call(**kw):
        prompts.append("\n".join(str(m.get("content")) for m in kw["messages"]))
        _call.archived_at_call.append(len(_archived_dir()))
        return response

    _call.archived_at_call = []
    prompts.archived_at_call = _call.archived_at_call

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


def _archived_dir() -> list[Path]:
    return _archived(Path(os.environ["HERMES_HOME"]))


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


def test_the_summarizer_reads_original_results_and_archives_land_only_after_the_summary(home, summarizer):
    """Summarize first, archive after: a distinctive in-window body reaches the summary prompt verbatim, no
    archive exists while the summarizer runs, and every dropped result is on disk byte-exact once it commits."""
    msgs = _transcript()
    msgs[5]["content"] = "DISTINCTIVE-ORIGINAL-7731 " + msgs[5]["content"]
    originals = _originals(msgs)
    c = _compressor()
    _working, pruned, start, end = c._plan_compaction_window(msgs)
    assert pruned and start < end <= len(msgs) and start <= 5 < end
    assert not _archived(home)  # planning is write-free

    compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    (prompt,) = summarizer
    assert "DISTINCTIVE-ORIGINAL-7731" in prompt
    assert ARCHIVE_POINTER_SEP not in prompt  # the summarizer never reads pointers
    assert summarizer.archived_at_call == [0]  # nothing was written before the summary existed
    files = _archived(home)
    assert files and {f.read_bytes() for f in files} <= {b.encode("utf-8") for b in originals.values()}
    committed = "\n".join(str(m.get("content")) for m in compressed)
    assert all(f"cache/pruned/{f.name}" in committed for f in files)  # every archive stays reachable
    _assert_pointers_name_byte_exact_originals(home, originals, compressed)


def test_an_aborted_summary_writes_no_archive(home):
    msgs = _transcript()
    c = _compressor()
    c.abort_on_summary_failure = True
    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("summary model down")):
        out = c.compress(msgs, current_tokens=c.threshold_tokens + 1)
    assert not any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in out)
    assert not _archived(home)


def test_prepared_pass_summarizes_the_original_results_and_writes_nothing(home):
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
    tool_rows = [m for m in seen if m.get("role") == "tool"]
    assert tool_rows and all(m["content"] == originals[m["tool_call_id"]] for m in tool_rows)
    assert not _archived(home)  # the splicing compaction archives, after its summary is adopted
    assert msgs == _transcript()  # the live transcript is untouched


def test_manual_compress_now_summarizes_originals_then_archives_what_it_drops(home, summarizer):
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
        assert summarizer and ARCHIVE_POINTER_SEP not in summarizer[-1]
        assert summarizer.archived_at_call[-1] == 0
        files = _archived(home)
        assert files and {f.read_bytes() for f in files} <= {b.encode("utf-8") for b in originals.values()}
    finally:
        db.close()


@pytest.mark.parametrize("trigger", [None, "cache_expiry"])
@pytest.mark.parametrize("flag", [None, "noop", "aborted", "refused_would_grow"])
def test_manual_and_automatic_compression_preserve_commit_and_force_contracts(trigger, flag):
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "row " * 20} for i in range(6)]
    summary = {"headline": "Compressed", **({flag: True} if flag else {})}
    compress = MagicMock(return_value=(history[-2:], None))
    agent = SimpleNamespace(_compress_context=compress, compression_summary=summary,
                            _compression_skipped_due_to_lock=False)

    result = compress_now(agent, history, CompressRequest(), trigger=trigger)

    assert result.status == "compressed"
    assert result.committed is (flag is None)
    assert compress.call_args.kwargs["force"] is (trigger is None)
    assert compress.call_args.kwargs.get("trigger") == trigger


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
        compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    assert failed and all(f.read_bytes() != failed[0].encode("utf-8") for f in _archived(home))
    assert _assert_pointers_name_byte_exact_originals(home, originals, compressed) or _archived(home)


def test_a_large_result_the_prune_left_whole_is_archived_after_the_summary_replaces_it(home, summarizer):
    """Live on Nous: a 44 KB read sat inside the summary window, the prune (tail-budget pass) left it whole,
    and the summary replaced it with no copy on disk. The summarizer reads it; the archive follows."""
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
    assert "Q" * 1_000 in prompt and ARCHIVE_POINTER_SEP not in prompt
    (path,) = _archived(home)
    assert path.read_bytes() == big.encode("utf-8")


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
    ``read_file`` a host path; either way the summarizer reads the original results."""
    monkeypatch.setenv("TERMINAL_ENV", backend)
    msgs = _transcript()
    c = _compressor()

    compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)

    assert any(m.get(COMPRESSED_SUMMARY_METADATA_KEY) for m in compressed)
    (prompt,) = summarizer
    assert bool(_archived(home)) is archives
    assert ARCHIVE_POINTER_SEP not in prompt


def test_compaction_window_archives_structured_tool_text_instead_of_losing_it(home):
    """A text-block tool result has no image to strip; the handoff still replaces its row."""
    body = [{"type": "text", "text": "Important result: " + "Q" * 4_000}]
    msgs = _round("structured", "placeholder")
    msgs[1]["content"] = body
    msgs[1]["api_content"] = body

    projected, count, _written = _compressor()._offload_window_results(msgs, 0, len(msgs))

    assert count == 1
    stub = projected[1]["content"]
    assert isinstance(stub, str) and ARCHIVE_POINTER_SEP in stub
    (path,) = _pointer_paths(stub)
    assert path.read_bytes() == json.dumps(body, ensure_ascii=False).encode("utf-8")
    assert "api_content" not in projected[1]  # no stale full body can bypass the stub
    assert msgs[1]["content"] == body  # the source transcript stays untouched


@pytest.mark.parametrize("body", ["receipt: command completed", "[Duplicate tool output: receipt]"])
def test_compaction_archives_small_string_results(home, summarizer, body):
    msgs = _transcript()
    msgs[5]["content"] = body
    c = _compressor()
    _plan, _n, start, end = c._plan_compaction_window(msgs)
    assert start <= 5 < end
    compressed = c.compress(msgs, current_tokens=c.threshold_tokens + 1)
    files = _archived(home)
    assert any(p.read_bytes() == body.encode("utf-8") for p in files)
    assert all(f"cache/pruned/{p.name}" in str(compressed) for p in files)
