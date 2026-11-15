"""Review round on the cold-cache prune (#156081): the guards fail closed and a step is never split.

Real ``ContextCompressor``, real ``read_file`` and the real backend resolver; nothing patched but the
context window and (for the write check) the disk.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return tmp_path


def _compressor(protect_last_n=4):
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        return ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=protect_last_n)


def _step(i, width, chars=12_000):
    ids = [f"s{i}_c{j}" for j in range(width)]
    rows = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": cid})}}
        for cid in ids]}]
    rows += [{"role": "tool", "tool_call_id": cid, "content": f"{cid}\n" + chr(65 + i) * chars} for cid in ids]
    return rows


def _earlier_turn(steps, width):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read everything"}]
    for i in range(steps):
        msgs += _step(i, width)
    return msgs


@pytest.mark.parametrize("turn_start", [-1, 3])  # -1: no user row survived; 3: a tool row
def test_a_turn_index_that_names_no_user_row_skips_the_round(home, turn_start):
    msgs = _earlier_turn(6, 1) + [{"role": "user", "content": "go on"}] + _earlier_turn(6, 1)[2:]
    pruned, count = _compressor().prune_tool_results_only(
        msgs, current_tokens=200_000, cold_cache=True, turn_start=turn_start)

    assert (pruned, count) == (msgs, 0)
    assert not (home / "cache" / "pruned").exists()


def test_parallel_results_of_one_step_are_pruned_all_or_none(home):
    """8 steps x 7 parallel calls: a count-only tail cut step 7 into 2 stubbed / 5 whole."""
    msgs = _earlier_turn(8, 7)
    pruned, count = _compressor(protect_last_n=12).prune_tool_results_only(msgs, current_tokens=200_000, cold_cache=True)

    assert count
    steps = [i for i, m in enumerate(msgs) if m.get("tool_calls")]
    states = []
    for start in steps:
        group = range(start + 1, start + 8)
        changed = {pruned[k]["content"] != msgs[k]["content"] for k in group}
        assert len(changed) == 1, f"step at {start} split"
        states.append(changed.pop())
    assert states[0] and not states[-1]


def test_a_single_line_txt_archive_reads_back_completely(home):
    """A 300,000-char one-line result: read_file must page all of it, not stop at the per-line cap."""
    from tools.file_tools import read_file_tool
    from tools.tool_result_storage import write_pruned_original

    body = "".join(f"<{i:06d}>" for i in range(37_500))
    assert len(body) == 300_000 and "\n" not in body
    host, stub = write_pruned_original(body, "s1", "terminal", "call_1")
    assert Path(host).read_bytes() == body.encode()

    seen, offset = [], 1
    for _ in range(400):
        page = json.loads(read_file_tool(os.path.expanduser(stub), offset=offset, limit=500))
        seen += [line.split("|", 1)[1] for line in page["content"].split("\n") if line]
        nxt = page.get("next_offset") or (offset + 500 if page.get("truncated") else None)
        if not nxt:
            break
        offset = nxt
    assert "".join(seen) == body


def test_an_archive_whose_bytes_differ_on_disk_is_discarded(home):
    """Same size, different bytes: a size check passes it, the byte comparison does not."""
    from tools import tool_result_storage as trs

    real_read = Path.read_bytes

    def corrupt(path):
        data = real_read(path)
        return data[:-1] + b"?" if "cache/pruned" in str(path) else data

    with patch.object(Path, "read_bytes", corrupt):
        assert trs.write_pruned_original("exact body " * 50, "s1", "terminal", "call_1") is None
    assert not any((home / "cache" / "pruned").iterdir())


@pytest.mark.parametrize("backend", ["my_plugin_backend", "singularity", "managed_modal"])
def test_any_backend_other_than_local_keeps_the_result_whole(home, monkeypatch, backend):
    """Fail closed on a backend the guard does not know: its read_file may not see host files."""
    monkeypatch.setenv("TERMINAL_ENV", backend)
    msgs = _earlier_turn(8, 1)
    pruned, count = _compressor().prune_tool_results_only(msgs, current_tokens=200_000, cold_cache=True)

    assert (pruned, count) == (msgs, 0)
    assert not (home / "cache" / "pruned").exists() or not any((home / "cache" / "pruned").iterdir())


def test_a_resumed_session_reads_the_stored_cache_clock(home):
    """A resumed agent (gateway/TUI replays carry no assistant timestamps) reads the shared clock the store
    saved: cold only past its estimated end; a committed prune bumps the context generation, so the next
    reading is unknown and the prune stands down until a fresh response re-anchors the clock."""
    import time
    from types import SimpleNamespace

    from agent.prompt_cache_clock import CacheEvent, _route_key, event_to_json
    from agent.turn_proactive_prune import prompt_cache_is_cold
    from hermes_state import SessionDB

    route = ("openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6")
    db = SessionDB(db_path=home / "state.db")

    def resumed():
        return SimpleNamespace(provider=route[0], base_url=route[1], model=route[2], api_mode="chat_completions",
                               session_id="s1", _session_db=db, context_compressor=SimpleNamespace(compression_count=0))

    def save(estimated_until):
        key = _route_key(*route)
        event = CacheEvent(route=key, destination=key, route_class="openrouter_claude", requested_ttl="5m",
                           sent_at=0.0, read_tokens=0, write_tokens=100, tier="5m", confidence="declared",
                           reason="test", coverage={"5m": estimated_until}, guaranteed_until=estimated_until,
                           estimated_until=estimated_until, session_id="s1",
                           generation=db.prompt_cache_generation("s1"))
        db.queue_token_counts("s1", input_tokens=1, cache_write_tokens=100, api_call_count=1,
                              prompt_cache_clock=event_to_json(event))
        assert db.flush_token_counts()

    try:
        db.create_session("s1", "cli")
        db.append_message("s1", role="user", content="q")
        db.append_message("s1", role="assistant", content="a")
        save(time.time() + 300)
        assert prompt_cache_is_cold(resumed()) is False
        save(time.time() - 1)
        assert prompt_cache_is_cold(resumed()) is True
        db.archive_and_compact("s1", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "[pruned]"}])
        assert prompt_cache_is_cold(resumed()) is False  # the rewrite moved the generation: unknown, not cold
    finally:
        db.close()
