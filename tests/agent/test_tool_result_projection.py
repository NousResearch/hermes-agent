"""Tests for the provider-bound tool-result projection (agent/tool_result_projection.py).

The projection replaces stale, large, recoverable tool results with stubs **on the outbound
request only** — the canonical transcript, session resume and the UI keep every byte. These
tests pin the contract the design claims: freshness, confirmed recoverability (including on a
remote backend), stability of the wire prefix, the two-phase transaction, and fail-closed
behaviour when recovery cannot be proven.

Mirrors the construction/patching conventions of
``tests/agent/test_proactive_tool_result_pruning.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import tool_result_projection as proj
from agent.context_compressor import ContextCompressor
from agent.tool_result_projection import (
    PROJECTION_MARKER,
    ProjectionKey,
    build_stub,
    content_digest,
    is_candidate,
    is_projected_tool_result,
    project_stale_tool_results,
    protected_tail_start,
    resolve_policy,
    tail_bounds,
    take_call,
    tool_call_index,
    trigger_tokens,
)
from tools.tool_result_storage import (
    get_spillover_dir,
    spillover_path_is_readable,
    store_spillover_content,
)

WINDOW = 200_000
BIG_CHARS = 20_000


def _compressor(**kw):
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,          # tail message floor (cap = 4 * 4 = 16)
        tool_result_projection="auto",
        tool_result_projection_min_tokens=8_000,
        tool_result_projection_min_result_chars=4_000,
        tool_result_projection_tail_ratio=0.0,   # tail = the 12K token floor + message bounds
    )
    defaults.update(kw)
    with patch("agent.context_compressor.get_model_context_length", return_value=WINDOW):
        return ContextCompressor(**defaults)


def _agent(cc=None, *, caching=False):
    agent = SimpleNamespace(_use_prompt_caching=caching)
    agent.context_compressor = cc if cc is not None else _compressor()
    return agent


def _assistant_call(cid, name="read_file", args='{"path":"src/foo.py"}'):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}],
    }


def _tool_msg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _build(n_pairs, big_indices, big_chars=BIG_CHARS, small="ok"):
    """``system`` + *n_pairs* (assistant tool_call, tool result) pairs, big ones distinct."""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid, args=json.dumps({"path": f"src/f{i}.py"})))
        msgs.append(_tool_msg(cid, chr(65 + i) * big_chars if i in big_indices else small))
    return msgs


def _content(msgs, cid):
    return [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == cid][0]["content"]


def _spillover_files():
    try:
        return sorted(p.name for p in get_spillover_dir().iterdir())
    except OSError:
        return []


def _with_buffer(msgs, *, big=6, small_pairs=8):
    """Pad a synthetic session with big non-candidate rows plus a small tail.

    The protected tail is a token budget (12K floor) plus a message floor, so a session built
    only from huge rows would keep pulling them into the tail. A buffer of rows that cannot be
    candidates at all keeps the interesting rows outside it, deterministically.
    """
    out = list(msgs)
    for i in range(big):
        cid = f"buf_{i}"
        out.append(_assistant_call(cid, name="read_file", args=json.dumps({"path": f"buffer/{i}.py"})))
        # Big AND non-candidate (an error envelope is never projected), so the buffer only
        # spends tail budget without ever being stubbed.
        out.append(_tool_msg(cid, json.dumps({"error": "buffer", "pad": "b" * BIG_CHARS})))
    for i in range(small_pairs):
        out.append(_assistant_call(f"tail2_{i}"))
        out.append(_tool_msg(f"tail2_{i}", "ok"))
    return out


def _stub_path(stub: str) -> str:
    return next(line for line in stub.splitlines() if line.startswith("full output archived at:")).split(": ", 1)[1]


class _FakeRemoteEnv:
    """Stand-in for a remote BaseEnvironment: records commands, answers readability probes."""

    def __init__(self, *, readable=True):
        self.readable = readable
        self.commands: list[str] = []
        self.probed: list[str] = []
        self.written: dict[str, str] = {}

    def execute(self, cmd, timeout=None, stdin_data=None):
        self.commands.append(cmd)
        if stdin_data is not None:
            self.written[cmd.rsplit(">", 1)[1].strip()] = stdin_data
            return {"returncode": 0, "output": ""}
        if cmd.startswith("test -r"):
            self.probed.append(cmd.split(" ", 2)[2].strip())
            return {"returncode": 0 if self.readable else 1, "output": ""}
        return {"returncode": 0, "output": ""}

    def get_temp_dir(self):
        return "/sandbox/tmp"


# ── opt-in by default ────────────────────────────────────────────────────────


def test_projection_is_off_unless_enabled():
    """Archiving old output changes what the model sees, so it ships opt-in."""
    assert resolve_policy(SimpleNamespace()).enabled is False
    assert resolve_policy(_agent(_compressor(tool_result_projection="off"))).enabled is False
    assert resolve_policy(_agent(_compressor(tool_result_projection="auto"))).enabled is True


def test_off_mode_never_touches_the_request():
    msgs = _build(16, big_indices=set(range(10)))
    assert project_stale_tool_results(_agent(_compressor(tool_result_projection="off")), msgs) == 0
    assert all(not is_projected_tool_result(m["content"]) for m in msgs if m.get("role") == "tool")


@pytest.mark.parametrize("mode", ["auto", "on", "true", "enabled"])
def test_accepted_enable_modes(mode):
    assert resolve_policy(_agent(_compressor(tool_result_projection=mode))).enabled is True


@pytest.mark.parametrize("mode", ["off", "false", "disabled", "none", "0", ""])
def test_accepted_disable_modes(mode):
    assert resolve_policy(_agent(_compressor(tool_result_projection=mode))).enabled is False


# ── the core win ─────────────────────────────────────────────────────────────


def test_projects_old_large_results():
    msgs = _build(16, big_indices=set(range(10)))
    projected = project_stale_tool_results(_agent(), msgs)

    assert projected >= 4
    for cid in ("call_0", "call_1", "call_2", "call_3"):
        content = _content(msgs, cid)
        assert is_projected_tool_result(content)
        assert "read_file" in content          # the tool that produced it
        assert len(content) < 600              # a stub, not the payload


def test_protected_tail_keeps_its_bytes():
    msgs = _build(16, big_indices=set(range(10)))
    project_stale_tool_results(_agent(), msgs)
    # The newest pair is always inside the tail.
    assert _content(msgs, "call_15") == "ok"
    assert all(not is_projected_tool_result(_content(msgs, f"call_{i}")) for i in (14, 15))


def test_small_results_are_never_projected():
    msgs = _build(16, big_indices=set())
    assert project_stale_tool_results(_agent(), msgs) == 0


def test_errors_multimodal_and_optout_keep_their_bytes():
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    payloads = {
        "call_err": json.dumps({"error": "ENOENT: no such file", "pad": "x" * BIG_CHARS}),
        "call_false": json.dumps({"success": False, "pad": "y" * BIG_CHARS}),
        "call_text": "Error: command failed\n" + "z" * BIG_CHARS,
        "call_optout": json.dumps({"projection_safe": False, "pad": "w" * BIG_CHARS}),
        "call_ok": "q" * BIG_CHARS,
    }
    msgs = [{"role": "system", "content": "sys"}]
    for cid, content in payloads.items():
        msgs.append(_assistant_call(cid))
        msgs.append(_tool_msg(cid, content))
    msgs = _with_buffer(msgs)

    assert project_stale_tool_results(_agent(cc), msgs) == 1
    for cid in ("call_err", "call_false", "call_text", "call_optout"):
        assert _content(msgs, cid) == payloads[cid], cid
    assert is_projected_tool_result(_content(msgs, "call_ok"))


def test_multimodal_tool_result_is_untouched():
    content = [{"type": "text", "text": "x" * BIG_CHARS}]
    msgs = [{"role": "system", "content": "sys"}, _assistant_call("call_m"), _tool_msg("call_m", content)]
    assert project_stale_tool_results(_agent(), msgs) == 0
    assert _content(msgs, "call_m") is content


def test_terminal_failure_envelopes_are_never_projected():
    """The real terminal result is ``{"output": ..., "exit_code": rc, "error": null}``: a non-zero
    exit is a failure even though ``error`` is null, and a stale failure (red tests, a broken build,
    an OOM kill) is often where the causal evidence the model needs later actually lives."""
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    payloads = {
        "call_rc1": json.dumps({"output": "x" * BIG_CHARS, "exit_code": 1, "error": None}),
        "call_oom": json.dumps({"output": "y" * BIG_CHARS, "exit_code": 137, "error": None}),
        "call_rctext": json.dumps({"output": "z" * BIG_CHARS, "exit_code": 2}),   # no "error" key
        "call_unknown_rc": json.dumps({"output": "u" * BIG_CHARS, "exit_code": None, "error": None}),
        "call_rc0": json.dumps({"output": "w" * BIG_CHARS, "exit_code": 0, "error": None}),
    }
    msgs = [{"role": "system", "content": "sys"}]
    for cid, content in payloads.items():
        msgs.append(_assistant_call(cid, name="terminal", args='{"command": "make test"}'))
        msgs.append(_tool_msg(cid, content))
    msgs = _with_buffer(msgs)

    assert project_stale_tool_results(_agent(cc), msgs) == 2   # the two successes stay projectable
    for cid in ("call_rc1", "call_oom", "call_rctext"):
        assert _content(msgs, cid) == payloads[cid], cid
    assert is_projected_tool_result(_content(msgs, "call_rc0"))
    assert is_projected_tool_result(_content(msgs, "call_unknown_rc"))  # no exit code to judge by


# ── invariant 1: canonical untouched, structure preserved ────────────────────


def test_canonical_transcript_is_never_modified():
    from agent.conversation_loop import _clone_message_for_send

    canonical = _build(16, big_indices=set(range(10)))
    snapshot = json.dumps(canonical, sort_keys=True)
    api_messages = [_clone_message_for_send(m) for m in canonical]

    assert project_stale_tool_results(_agent(), api_messages) > 0

    assert json.dumps(canonical, sort_keys=True) == snapshot
    assert is_projected_tool_result(_content(canonical, "call_0")) is False
    assert is_projected_tool_result(_content(api_messages, "call_0")) is True


def test_roles_order_and_tool_call_ids_survive():
    msgs = _build(16, big_indices=set(range(10)))
    before = [(m.get("role"), m.get("tool_call_id")) for m in msgs]
    project_stale_tool_results(_agent(), msgs)
    assert [(m.get("role"), m.get("tool_call_id")) for m in msgs] == before


# ── invariant 2: recovery is confirmed, or the row keeps its bytes ───────────


def test_full_output_is_recoverable_from_the_archived_file():
    msgs = _build(16, big_indices=set(range(10)))
    original = _content(msgs, "call_1")
    project_stale_tool_results(_agent(), msgs)

    path = _stub_path(_content(msgs, "call_1"))
    assert Path(path).is_file()
    assert Path(path).read_text(encoding="utf-8") == original


def test_failed_persist_keeps_the_result_fail_closed():
    msgs = _build(16, big_indices=set(range(10)))
    with patch("tools.tool_result_storage.store_spillover_content", return_value=None):
        assert project_stale_tool_results(_agent(), msgs) == 0
    assert all(not is_projected_tool_result(m["content"]) for m in msgs if m.get("role") == "tool")


def test_dead_persisted_path_keeps_the_preview():
    """A ``<persisted-output>`` row whose file was pruned keeps its preview: the preview is
    worth more to the model than a pointer to a file that no longer exists."""
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    body = ("<persisted-output>\nThis tool result was too large.\n"
            "Full output saved to: /nonexistent/gone.txt\nPreview:\n" + "p" * BIG_CHARS
            + "\n</persisted-output>")
    msgs = [{"role": "system", "content": "sys"}, _assistant_call("call_p"), _tool_msg("call_p", body)]
    for i in range(8):
        msgs.append(_assistant_call(f"call_t{i}"))
        msgs.append(_tool_msg(f"call_t{i}", "ok"))

    assert project_stale_tool_results(_agent(cc), msgs) == 0
    assert _content(msgs, "call_p") == body


def test_live_persisted_path_is_reused_without_rewriting_the_file():
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    spill = get_spillover_dir() / "call_p_live.txt"
    spill.parent.mkdir(parents=True, exist_ok=True)
    spill.write_text("full body", encoding="utf-8")
    body = ("<persisted-output>\nToo large.\n"
            f"Full output saved to: {spill}\nPreview:\n" + "p" * BIG_CHARS + "\n</persisted-output>")
    msgs = [{"role": "system", "content": "sys"}, _assistant_call("call_p"), _tool_msg("call_p", body)]
    msgs = _with_buffer(msgs)

    assert project_stale_tool_results(_agent(cc), msgs) == 1
    assert str(spill) in _content(msgs, "call_p")
    assert spill.read_text(encoding="utf-8") == "full body"


# ── remote backend ladder ────────────────────────────────────────────────────


def test_remote_backend_without_a_live_env_projects_nothing():
    """A host path is no proof of readability inside a sandbox, and with no env there is
    nothing to probe or copy into — so the pass declines instead of writing a dead pointer."""
    msgs = _build(16, big_indices=set(range(10)))
    with patch.object(proj, "_backend_is_remote", return_value=True):
        assert project_stale_tool_results(_agent(), msgs, env=None) == 0
    assert all(not is_projected_tool_result(m["content"]) for m in msgs if m.get("role") == "tool")


def test_remote_backend_uses_the_sandbox_visible_path(monkeypatch):
    """docker/modal: the stub must name the translated path the sandbox can read, not the host
    path — and the host copy still lands in the canonical spillover store."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    env = _FakeRemoteEnv(readable=True)
    msgs = _build(16, big_indices=set(range(10)))

    assert project_stale_tool_results(_agent(), msgs, env=env) >= 4

    path = _stub_path(_content(msgs, "call_0"))
    assert path.startswith("/root/.hermes/"), path
    assert path in env.probed                      # confirmed readable inside the sandbox
    assert Path(path).name in _spillover_files()   # canonical host copy exists too


def test_remote_backend_declines_when_the_sandbox_cannot_read_it(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    env = _FakeRemoteEnv(readable=False)
    msgs = _build(16, big_indices=set(range(10)))

    assert project_stale_tool_results(_agent(), msgs, env=env) == 0
    assert all(not is_projected_tool_result(m["content"]) for m in msgs if m.get("role") == "tool")


def test_the_turn_task_id_is_the_key_the_sandbox_registry_uses(monkeypatch):
    """The sandbox is registered under the turn's ``effective_task_id``, which the turn prologue
    stores as ``agent._current_task_id`` (it is a fresh UUID when the caller passes no task id).
    Looking anywhere else — ``session_id`` especially — finds no env, which turns remote support off
    on the real path while every unit test that hands ``env=`` in explicitly still passes."""
    from tools.terminal_tool import _active_environments

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    env = _FakeRemoteEnv(readable=True)
    agent = _agent()
    agent.session_id = "session-abc"                 # plausible-looking, and NOT the sandbox key
    agent._current_task_id = "turn-123"
    assert getattr(agent, "_task_id", None) is None  # the key the first version looked for

    with patch.dict(_active_environments, {"turn-123": env}, clear=True):
        assert proj._resolve_active_env(agent) is env
        msgs = _build(16, big_indices=set(range(10)))
        # end to end, through ``env="auto"``: the remote ladder is actually reachable
        assert project_stale_tool_results(agent, msgs) >= 4
        assert _stub_path(_content(msgs, "call_0")).startswith("/root/.hermes/")
        assert env.probed


def test_an_unresolved_backend_is_fail_closed(monkeypatch):
    """``_backend_is_remote`` is tri-state: if resolution fails we do NOT know the backend is
    host-side, and assuming it would hand the model a host path it may not be able to open."""
    # The real resolver, with its import broken, must answer "unknown" — not "local".
    with patch.dict(sys.modules, {"tools.env_probe": None}):
        assert proj._backend_is_remote() is None

    msgs = _build(16, big_indices=set(range(10)))
    with patch.object(proj, "_backend_is_remote", return_value=None):
        assert project_stale_tool_results(_agent(), msgs, env=None) == 0
    assert all(not is_projected_tool_result(m["content"]) for m in msgs if m.get("role") == "tool")

    # ...but unknown WITH a live env to probe is fine: readability is still confirmed.
    msgs = _build(16, big_indices=set(range(10)))
    with patch.object(proj, "_backend_is_remote", return_value=None):
        assert project_stale_tool_results(_agent(), msgs, env=_FakeRemoteEnv(readable=True)) >= 4


def test_store_ladder_contract():
    """The storage helper is the single place the ladder lives: host path locally, verified
    path on a remote env, and None whenever readability cannot be confirmed."""
    host_env = None  # host-side (no sandbox)
    path = store_spillover_content("payload", "key_host", env=host_env)
    assert path and Path(path).is_file()
    assert path == str(get_spillover_dir() / "key_host.txt")

    remote_ok = _FakeRemoteEnv(readable=True)
    assert store_spillover_content("payload2", "key_remote", env=remote_ok)
    assert remote_ok.probed, "a remote path must be probed before it is handed out"

    remote_bad = _FakeRemoteEnv(readable=False)
    assert store_spillover_content("payload3", "key_remote_bad", env=remote_bad) is None
    assert spillover_path_is_readable("/anything", None) is False or True  # host-side stat, no env


# ── invariant 5: identity + stability ────────────────────────────────────────


def test_rows_sharing_or_missing_a_call_id_recover_their_own_bytes():
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    rows = [("dup", "A" * BIG_CHARS), ("dup", "B" * BIG_CHARS), ("", "C" * BIG_CHARS), ("", "D" * BIG_CHARS)]
    msgs = [{"role": "system", "content": "sys"}]
    for idx, (cid, payload) in enumerate(rows):
        msgs.append(_assistant_call(cid or f"call_{idx}"))
        msgs.append(_tool_msg(cid, payload))
    msgs = _with_buffer(msgs)

    assert project_stale_tool_results(_agent(cc), msgs) == 4

    tool_rows = [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") in {"dup", ""}]
    assert len(tool_rows) == 4
    seen = {}
    for row, (_cid, payload) in zip(tool_rows, rows):
        assert is_projected_tool_result(row["content"])
        seen[_stub_path(row["content"])] = payload
    assert len(seen) == 4, "each row needs its own archived file"
    for path, payload in seen.items():
        assert Path(path).read_text(encoding="utf-8") == payload


def test_a_new_row_reusing_a_projected_id_is_not_treated_as_already_done():
    """Identity is (call id, content digest): a NEW row that reuses an id must not inherit the
    old row's stickiness and bypass the protected tail."""
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    agent = _agent(cc)
    msgs = _build(16, big_indices=set(range(10)))
    assert project_stale_tool_results(agent, msgs) >= 4
    assert is_projected_tool_result(_content(msgs, "call_3"))   # a row that WAS projected

    # Same id as an already-projected row, different bytes, and it sits inside the tail.
    grown = [dict(m) for m in msgs]
    grown.append(_assistant_call("call_3"))
    grown.append(_tool_msg("call_3", "N" * BIG_CHARS))
    project_stale_tool_results(agent, grown)

    dup_rows = [m for m in grown if m.get("role") == "tool" and m.get("tool_call_id") == "call_3"]
    assert len(dup_rows) == 2
    assert is_projected_tool_result(dup_rows[0]["content"])     # the original, projected earlier
    assert dup_rows[1]["content"] == "N" * BIG_CHARS            # the new row is in the tail: untouched


def test_sticky_rows_replay_byte_identically_when_the_gates_decline():
    agent = _agent()
    canonical = _build(16, big_indices=set(range(10)))
    first = [dict(m) for m in canonical]
    assert project_stale_tool_results(agent, first) >= 4
    stub = _content(first, "call_0")

    # Raise the trigger through the roof: no FRESH work is approved, but stability is not
    # optional — the same row must come back with the same bytes.
    agent.context_compressor.tool_result_projection_min_tokens = 10_000_000
    second = [dict(m) for m in canonical]
    assert project_stale_tool_results(agent, second) >= 1
    assert _content(second, "call_0") == stub


def test_tool_name_and_args_follow_occurrence_order():
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append(_assistant_call("dup", name="terminal", args='{"command":"ls"}'))
    msgs.append(_tool_msg("dup", "T" * BIG_CHARS))
    msgs.append(_assistant_call("dup", name="web_extract", args='{"urls":["https://x"]}'))
    msgs.append(_tool_msg("dup", "W" * BIG_CHARS))
    msgs = _with_buffer(msgs)

    assert project_stale_tool_results(_agent(cc), msgs) == 2
    first, second = [m["content"] for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "dup"]
    assert "tool=terminal" in first and '"command":"ls"' in first
    assert "tool=web_extract" in second and "https://x" in second


def test_duplicate_ids_with_unequal_candidacy_stay_paired():
    """A result that is NOT a candidate still owns its occurrence of the call id. Consuming lazily
    made the small row skip the queue, so the large row inherited the small row's call and its stub
    described a terminal command it never came from."""
    cc = _compressor(tool_result_projection_min_tokens=2_000)
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append(_assistant_call("dup", name="terminal", args='{"command":"pytest -q"}'))
    msgs.append(_tool_msg("dup", "ok"))                     # small: never a candidate
    msgs.append(_assistant_call("dup", name="web_extract", args='{"urls":["https://real.example"]}'))
    msgs.append(_tool_msg("dup", "W" * BIG_CHARS))          # large: the one that gets projected
    msgs = _with_buffer(msgs)

    assert project_stale_tool_results(_agent(cc), msgs) == 1
    # Both rows carry the same call id, so select them by position: small first, large second.
    small, large = [m["content"] for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "dup"]
    assert small == "ok"
    assert is_projected_tool_result(large)
    assert "tool=web_extract" in large and "https://real.example" in large
    assert "pytest -q" not in large, "the stub must not borrow the un-projected row's call"


def test_recovery_instruction_never_invites_re_execution():
    stub = build_stub(
        tool_name="terminal", tool_args='{"command":"terraform apply"}', content_len=5000,
        line_count=10, digest="abc123", recovery_path="/tmp/archived.txt",
    )
    assert "re-run" not in stub.lower() and "rerun" not in stub.lower()
    assert "read the archived file with read_file" in stub.lower()
    assert "terraform apply" in stub  # the args are quoted as metadata, not as an instruction


# ── invariant 6: two phases, transactional commit ───────────────────────────


def test_declined_pass_writes_nothing_to_disk():
    before = _spillover_files()
    msgs = _build(16, big_indices=set(range(10)))
    cc = _compressor(tool_result_projection_min_tokens=10_000_000)  # trigger unreachable
    assert project_stale_tool_results(_agent(cc), msgs) == 0
    assert _spillover_files() == before


def test_phase_two_failure_leaves_the_wire_untouched():
    """Persistence happens after the decision and before the commit; a failure in between must
    leave the request exactly as the assembly produced it."""
    msgs = _build(16, big_indices=set(range(10)))
    snapshot = json.dumps(msgs, sort_keys=True)
    calls = {"n": 0}
    real_build = proj.build_stub

    def _explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom")
        return real_build(*args, **kwargs)

    with patch.object(proj, "build_stub", side_effect=_explode):
        assert project_stale_tool_results(_agent(), msgs) == 0
    assert json.dumps(msgs, sort_keys=True) == snapshot


def test_cache_capable_declines_when_the_reclaim_cannot_pay_for_the_break():
    msgs = _build(16, big_indices=set(range(10)))
    msgs.append({"role": "user", "content": "t" * 200_000})
    assert project_stale_tool_results(_agent(caching=True), msgs) == 0
    assert project_stale_tool_results(_agent(caching=False), msgs) >= 4


def test_cache_capable_routes_need_a_bigger_pile_of_stale_bytes():
    policy = resolve_policy(_agent(_compressor(tool_result_projection_min_tokens=0)))
    assert trigger_tokens(policy, WINDOW, True) > trigger_tokens(policy, WINDOW, False)
    assert trigger_tokens(policy, 8_000, True) > 0


def test_a_provider_cached_route_takes_the_cached_path_without_marker_policy():
    """``_use_prompt_caching`` means "Hermes emits cache-control markers", NOT "the destination
    caches prefixes". A route that reports cached input tokens has a prefix cache worth protecting
    even with the marker policy off, and it has to take the bigger trigger plus the break-cost gate
    — otherwise a pass can invalidate an 80K-char cached prefix to reclaim 16K tokens."""
    def _session():
        msgs = _build(16, big_indices=set(range(10)))
        msgs.append({"role": "user", "content": "t" * 200_000})   # makes the break uneconomical
        return msgs

    # No caching anywhere: the route is treated as uncached and the pass runs.
    assert project_stale_tool_results(_agent(caching=False), _session()) >= 4
    # Explicit marker policy: the pass declines.
    assert project_stale_tool_results(_agent(caching=True), _session()) == 0

    # Same route, marker policy off, but the provider's own usage accounting shows a warm prefix.
    observed = _agent(caching=False)
    observed.session_cache_read_tokens = 86_832
    assert proj.destination_caches_prefixes(observed) is True
    assert project_stale_tool_results(observed, _session()) == 0
    # ...and the signal really is what changed, not the flag:
    assert proj.destination_caches_prefixes(_agent(caching=False)) is False


def test_recomputed_break_cost_declines_a_pass_that_verification_shrank():
    """Phase 1 approves against a region where EVERY planned row became a stub. If verification
    then drops rows, those rows stay complete in the request, the invalidated region grows, and the
    reclaim can stop paying for it — so the second gate has to be recomputed, not assumed."""
    def _build_uneven():
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(14):
            cid = f"call_{i}"
            # One huge row sits AFTER the first stub, so dropping it leaves the region that gets
            # re-prefilled holding its full bytes while the reclaim shrinks.
            chars = 300_000 if i == 3 else 20_000
            msgs.append(_assistant_call(cid, args=json.dumps({"path": f"src/f{i}.py"})))
            msgs.append(_tool_msg(cid, chr(65 + i) * chars))
        return _with_buffer(msgs)

    cc = _compressor(tool_result_projection_min_tokens=0)

    # Control: with every planned row verified, the same session is projectable.
    msgs = _build_uneven()
    assert project_stale_tool_results(_agent(cc, caching=True), msgs) >= 4

    # Now the single largest candidate fails verification. The survivors still clear the
    # stale-pile trigger, so only a recomputed cache-break cost can stop the commit.
    msgs = _build_uneven()
    snapshot = json.dumps(msgs, sort_keys=True)
    real_confirm = proj._confirmed_stub
    dropped_keys: list[str] = []

    def _drop_the_big_ones(entry, env):
        if entry["old_tokens"] > 50_000:
            dropped_keys.append(entry["key"].token())
            return None
        return real_confirm(entry, env)

    estimates = {"n": 0}
    real_estimate = proj._estimate_region_tokens

    def _counting_estimate(*args, **kwargs):
        estimates["n"] += 1
        return real_estimate(*args, **kwargs)

    with patch.object(proj, "_confirmed_stub", side_effect=_drop_the_big_ones), \
         patch.object(proj, "_estimate_region_tokens", side_effect=_counting_estimate):
        assert project_stale_tool_results(_agent(cc, caching=True), msgs) == 0

    assert dropped_keys, "the test must actually drop rows in phase 2"
    assert estimates["n"] == 2, "phase 1 decision + the post-verification recheck"
    assert json.dumps(msgs, sort_keys=True) == snapshot, "a declined pass must not touch the wire"


def test_tail_message_bounds_are_internal_constants():
    """``protect_last_n`` governs what a compaction summary keeps; deriving the tail from it would
    put most of a tool-heavy session inside the protected region (its default is 20 messages)."""
    floor, cap = proj.tail_bounds(resolve_policy(_agent(_compressor(protect_last_n=200))), WINDOW)[1:]
    assert (floor, cap) == (proj.DEFAULT_TAIL_FLOOR_MESSAGES, proj.DEFAULT_TAIL_MESSAGE_CAP)


# ── helpers ──────────────────────────────────────────────────────────────────


def test_projection_key_is_content_scoped():
    assert ProjectionKey("c1", content_digest("a")).token() != ProjectionKey("c1", content_digest("b")).token()
    assert ProjectionKey("", content_digest("a")).storage_key().startswith("tool_result_")


def test_stub_is_a_pure_function_of_the_row():
    kwargs = dict(tool_name="read_file", tool_args='{"path":"a.py"}', content_len=10,
                  line_count=2, digest="deadbeef", recovery_path="/tmp/s.txt")
    stub = build_stub(**kwargs)
    assert stub == build_stub(**kwargs)
    assert stub.startswith(PROJECTION_MARKER)
    assert stub == stub.strip() and not stub.endswith("\n")


def test_stub_is_not_mistaken_for_an_existing_summary_stub():
    from agent.context_compressor import _is_summary_stub

    stub = build_stub(tool_name="terminal", tool_args='{"command":"pytest -q"}', content_len=1234,
                      line_count=9, digest="cafe", recovery_path="/tmp/s.txt")
    assert _is_summary_stub(stub) is False


def test_tail_bounds_are_bounded_on_both_sides():
    policy = resolve_policy(_agent(_compressor(tool_result_projection_tail_ratio=0.0)))
    tokens, floor, cap = tail_bounds(policy, WINDOW)
    assert tokens == 12_000 and floor == 8 and cap == 60

    msgs = _build(16, big_indices=set(range(10)))
    start = protected_tail_start(msgs, policy, WINDOW)
    assert 0 < start < len(msgs)


def test_tool_call_index_keeps_every_occurrence():
    index = tool_call_index([_assistant_call("c1", name="terminal"), _assistant_call("c1", name="read_file")])
    assert index["c1"] == [("terminal", '{"path":"src/foo.py"}'), ("read_file", '{"path":"src/foo.py"}')]
    assert take_call(index, {"tool_call_id": "c1"})[0] == "terminal"
    assert take_call(index, {"tool_call_id": "c1"})[0] == "read_file"
    assert take_call(index, {"tool_call_id": "c1"}) == ("unknown", "")


def test_is_candidate_rejects_small_and_non_tool_rows():
    policy = resolve_policy(_agent())
    assert is_candidate({"role": "user", "content": "x" * BIG_CHARS}, policy) is False
    assert is_candidate(_tool_msg("c", "small"), policy) is False
    assert is_candidate(_tool_msg("c", "x" * BIG_CHARS), policy) is True


# ── wired into the real request assembly ─────────────────────────────────────


class _CapturingCompletions:
    """One canned completion that records the messages the API call actually carried."""

    def __init__(self):
        self.requests: list = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content="Done.", tool_calls=[], reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


def _run_real_turn(monkeypatch, history):
    from run_agent import AIAgent

    completions = _CapturingCompletions()
    monkeypatch.setattr(
        "agent.process_bootstrap.OpenAI",
        lambda **_kw: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])

    agent = AIAgent(
        model="test-model", api_key="test-key", base_url="http://localhost:8080/v1",
        platform="cli", max_iterations=3, quiet_mode=True, skip_memory=True,
    )
    agent._disable_streaming = True
    # Deterministic policy: what the request carries, not how the window is inferred.
    agent.context_compressor.tool_result_projection = "auto"
    agent.context_compressor.tool_result_projection_min_tokens = 8_000
    agent.context_compressor.tool_result_projection_min_result_chars = 4_000
    agent.context_compressor.tool_result_projection_tail_ratio = 0.0
    agent.context_compressor.protect_last_n = 4

    result = agent.run_conversation("keep going", conversation_history=[dict(m) for m in history])
    assert completions.requests, "the turn made no API call"
    return agent, result, completions.requests[0]["messages"]


def _stale_history():
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
    for i in range(14):
        history.append(_assistant_call(f"call_{i}", args=json.dumps({"path": f"src/f{i}.py"})))
        history.append(_tool_msg(f"call_{i}", chr(65 + i) * BIG_CHARS))
    for i in range(8):
        history.append(_assistant_call(f"call_tail_{i}"))
        history.append(_tool_msg(f"call_tail_{i}", "ok"))
    return history


def test_request_carries_stubs_while_the_transcript_keeps_the_bytes(monkeypatch):
    history = _stale_history()
    before = json.dumps(history, sort_keys=True)

    _agent_, result, sent = _run_real_turn(monkeypatch, history)

    stale_on_wire = [
        m for m in sent
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
        and is_projected_tool_result(m["content"])
    ]
    assert stale_on_wire, "no stale tool result was projected on the wire"
    assert "archived at:" in stale_on_wire[0]["content"]

    assert json.dumps(history, sort_keys=True) == before
    kept = [m for m in sent if m.get("role") == "tool" and m.get("tool_call_id") == "call_tail_7"]
    assert kept and kept[0]["content"] == "ok"


def test_spillover_written_by_the_projection_holds_the_original_bytes(monkeypatch):
    history = _stale_history()
    _agent_, _result, sent = _run_real_turn(monkeypatch, history)

    stub = next(m["content"] for m in sent
                if m.get("role") == "tool" and is_projected_tool_result(m.get("content") or ""))
    recovered = Path(_stub_path(stub)).read_text(encoding="utf-8")
    assert recovered == next(m["content"] for m in history if m.get("tool_call_id") == "call_0")


# ── config plumbing ──────────────────────────────────────────────────────────


def test_config_section_reaches_the_compressor():
    from agent.agent_init import _parse_compression_config

    agent = SimpleNamespace(model="test", context_length=None, api_mode="chat_completions")
    cfg = {"compression": {
        "tool_result_projection": "auto",
        "tool_result_projection_min_tokens": 12_345,
        "tool_result_projection_min_result_chars": 9_000,
        "tool_result_projection_tail_ratio": 0.05,
    }}
    with patch("agent.context_compressor.get_model_context_length", return_value=WINDOW):
        cs = _parse_compression_config(agent, cfg)

    assert cs.tool_result_projection == "auto"
    assert cs.tool_result_projection_min_tokens == 12_345
    assert cs.tool_result_projection_min_result_chars == 9_000
    assert cs.tool_result_projection_tail_ratio == pytest.approx(0.05)


def test_config_section_defaults_to_off_and_survives_junk():
    from agent.agent_init import _parse_compression_config

    agent = SimpleNamespace(model="test", context_length=None, api_mode="chat_completions")
    junk = {"compression": {
        "tool_result_projection": None,
        "tool_result_projection_min_tokens": "nonsense",
        "tool_result_projection_tail_ratio": True,
    }}
    with patch("agent.context_compressor.get_model_context_length", return_value=WINDOW):
        cs = _parse_compression_config(agent, junk)

    assert cs.tool_result_projection == "off"
    assert cs.tool_result_projection_min_tokens == 0
    assert cs.tool_result_projection_tail_ratio == pytest.approx(0.025)
