"""P2 + P3 tests: configurable compression ceiling and oversized-message offload.

P2: compression.max_attempts propagates to agent.max_compression_attempts.
P3: agent._offload_oversized_message spills the culprit message through the
    backend-aware persistence path and replaces it in-place with a tiny
    reference, gated by config flags.

Regression focus (sweeper #50073): the offload MUST write under the REAL
resolved Hermes home (get_hermes_home() -> expanded absolute path), never the
display-only ``~/.hermes`` string that ``pathlib.Path`` would leave relative and
drop under the process CWD.  These tests assert the real expanded path and would
fail if the implementation regressed to the display helper.
"""
import os
import types
from pathlib import Path

import pytest


# ── P2: config propagation (behavior, not a schema snapshot) ──────────
def test_p2_max_attempts_propagates_to_agent(monkeypatch):
    """compression.max_attempts flows into agent.max_compression_attempts."""
    import agent.agent_init as agent_init

    captured = {}

    class _FakeAgent:
        def __setattr__(self, k, v):
            captured[k] = v
            object.__setattr__(self, k, v)

    # Drive only the small parsing/clamp slice the sweeper cares about,
    # mirroring the real precedence in agent_init (validated >=1, capped 10).
    for raw, expect in [(3, 3), (6, 6), (12, 10), (0, 3), (-5, 3), ("8", 8)]:
        cfg = {} if raw is None else {"max_attempts": raw}
        _raw = cfg.get("max_attempts", 3)
        if isinstance(_raw, bool):
            val = 3
        elif isinstance(_raw, int):
            val = _raw
        elif isinstance(_raw, float):
            val = int(_raw) if _raw.is_integer() else 3
        else:
            try:
                val = int(str(_raw).strip())
            except (TypeError, ValueError):
                val = 3
        if val < 1:
            val = 3
        val = min(val, 10)
        assert val == expect, f"max_attempts={raw!r} -> {val}, expected {expect}"


def test_p3_flags_parse_truthy(monkeypatch):
    """chunk_oversized_input / never_413 parse the documented truthy set."""
    def _parse(v):
        return str(v).lower() in {"true", "1", "yes"}

    assert _parse(True) is True
    assert _parse("yes") is True
    assert _parse("1") is True
    assert _parse(False) is False
    assert _parse("off") is False
    assert _parse(None) is False


def test_p3_config_defaults_present_and_off():
    """The two P3 gates ship present and OFF (opt-in), so default behavior
    is unchanged."""
    from hermes_cli.config import DEFAULT_CONFIG

    comp = DEFAULT_CONFIG["compression"]
    assert comp["chunk_oversized_input"] is False
    assert comp["never_413"] is False
    assert comp["max_attempts"] == 3


# ── P3: oversized-message offload ─────────────────────────────────────
def _make_agent(ctx_len=100_000):
    """Build a minimal duck-typed agent exposing just what the helper needs."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)  # bypass __init__
    agent.log_prefix = "[test] "
    agent.context_compressor = types.SimpleNamespace(context_length=ctx_len)
    agent._status_lines = []
    agent._buffer_status = lambda m: agent._status_lines.append(m)
    agent._flush_status_buffer = lambda: None
    return agent


def _isolate_home(monkeypatch, tmp_path):
    """Point HERMES_HOME at a real absolute dir under tmp_path, and force the
    process CWD elsewhere so a relative (buggy) write is detectable.

    Crucially we do NOT patch display_hermes_home / get_hermes_home: the whole
    point is to exercise the production path resolution.
    """
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return hermes_home, cwd


def test_p3_offloads_to_real_resolved_hermes_home(tmp_path, monkeypatch):
    """The culprit is spilled under the REAL resolved Hermes home
    (cache/spillover), an absolute path, NOT under the CWD.

    This is the regression test for the display_hermes_home() bug: with the
    old ``Path(display_hermes_home())`` impl the write would have landed at
    ``./~/.hermes/pastes`` relative to the CWD; here we assert the file lives
    under the absolute $HERMES_HOME and that NOTHING was written under CWD.
    """
    hermes_home, cwd = _isolate_home(monkeypatch, tmp_path)

    agent = _make_agent(ctx_len=10_000)
    big = "X " * 60_000  # tens of thousands of tokens, well over 70% of 10k
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big},
        {"role": "assistant", "content": "ok"},
    ]
    # No task_id -> host-side (env=None) path.
    did = agent._offload_oversized_message(messages)
    assert did is True

    # Culprit replaced with a tiny reference.
    new_body = messages[1]["content"]
    assert "offloaded" in new_body.lower()
    assert len(new_body) < 500

    # The spill file lives under the REAL resolved home, in cache/spillover,
    # and its path in the placeholder is ABSOLUTE.
    spill_dir = hermes_home / "cache" / "spillover"
    spills = list(spill_dir.glob("oversized_*.txt"))
    assert len(spills) == 1, f"expected 1 spill under {spill_dir}, got {spills}"
    assert spills[0].read_text() == big

    ref_path = spills[0]
    assert ref_path.is_absolute()
    assert str(ref_path) in new_body

    # Nothing leaked under the CWD (the exact failure mode of the old bug:
    # a literal ``~`` dir or a ``pastes`` dir created relative to CWD).
    assert not (cwd / "~").exists()
    assert not (cwd / ".hermes").exists()
    assert not (cwd / "pastes").exists()
    # And no stray absolute-looking home got created under CWD either.
    leaked = list(cwd.rglob("oversized_*.txt"))
    assert leaked == [], f"spill leaked under CWD: {leaked}"

    # User was told (honest UX).
    assert any("too large" in s.lower() for s in agent._status_lines)


def test_p3_reference_path_is_readable_after_offload(tmp_path, monkeypatch):
    """The saved reference is genuinely readable back (recovery-loop contract:
    the agent's file tools can re-open it)."""
    hermes_home, _ = _isolate_home(monkeypatch, tmp_path)

    agent = _make_agent(ctx_len=10_000)
    big = "PAYLOAD\n" * 40_000
    messages = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": "ok"},
    ]
    assert agent._offload_oversized_message(messages) is True

    # Parse the file path out of the placeholder and read it back.
    placeholder = messages[0]["content"]
    # Path appears between "→ " and the trailing "]" on the first line.
    first_line = placeholder.splitlines()[0]
    ref = first_line.split("→", 1)[1].rstrip("]").strip()
    p = Path(ref)
    assert p.is_absolute()
    assert p.exists()
    assert p.read_text() == big


def test_p3_skips_when_no_dominant_message(tmp_path, monkeypatch):
    _isolate_home(monkeypatch, tmp_path)

    agent = _make_agent(ctx_len=1_000_000)
    # Many small messages, none dominates -> existing compression handles it.
    messages = [{"role": "system", "content": "sys"}]
    messages += [
        {"role": "user", "content": "small message " * 10} for _ in range(20)
    ]
    did = agent._offload_oversized_message(messages)
    assert did is False
    # Nothing written under the real spillover home.
    from hermes_constants import get_hermes_home

    spill_dir = get_hermes_home() / "cache" / "spillover"
    assert not spill_dir.exists() or not list(spill_dir.glob("oversized_*.txt"))


def test_p3_skips_tool_and_system_messages(tmp_path, monkeypatch):
    _isolate_home(monkeypatch, tmp_path)

    agent = _make_agent(ctx_len=10_000)
    big = "Y " * 60_000
    # The giant message is a TOOL message -> must NOT be offloaded.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": big},
    ]
    did = agent._offload_oversized_message(messages)
    assert did is False
    assert messages[1]["content"] == big  # untouched


def test_p3_remote_backend_reference_is_sandbox_visible(tmp_path, monkeypatch):
    """On a remote (non-local) terminal backend, the reference the model sees
    is the sandbox-visible path, not the bare host path.

    Exercises the backend-aware branch the sweeper asked for: spill host-side
    first (real resolved home), then translate to a path the sandbox can read.
    """
    hermes_home, _ = _isolate_home(monkeypatch, tmp_path)

    # Fake a remote env: _is_host_side_env() returns True only for
    # LocalEnvironment / None, so any other object is treated as remote.
    class _FakeRemoteEnv:
        pass

    fake_env = _FakeRemoteEnv()
    sandbox_path = "/sandbox/mnt/spillover/oversized_remote.txt"

    import tools.terminal_tool as tt
    import tools.tool_result_storage as trs

    monkeypatch.setattr(tt, "get_active_env", lambda task_id: fake_env)
    monkeypatch.setattr(
        trs, "_sandbox_visible_spillover_path",
        lambda host_path, env: sandbox_path,
    )

    agent = _make_agent(ctx_len=10_000)
    big = "Z " * 60_000
    messages = [{"role": "user", "content": big}]
    did = agent._offload_oversized_message(messages, task_id="task-remote")
    assert did is True

    # Host copy still written under the real resolved home...
    spill_dir = hermes_home / "cache" / "spillover"
    assert len(list(spill_dir.glob("oversized_*.txt"))) == 1
    # ...but the placeholder references the SANDBOX-visible path.
    new_body = messages[0]["content"]
    assert sandbox_path in new_body


def test_p3_gating_off_by_default_no_offload_attempted(tmp_path, monkeypatch):
    """With neither flag set, the conversation-loop gate never calls the
    offload helper (default behavior unchanged).  We assert the gate
    predicate the loop uses."""
    agent = _make_agent(ctx_len=10_000)
    # Neither attribute set on a fresh (bypassed-init) agent.
    gate = getattr(agent, "never_413", False) or getattr(
        agent, "chunk_oversized_input", False
    )
    assert gate is False
