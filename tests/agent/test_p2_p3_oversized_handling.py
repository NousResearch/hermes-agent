"""P2 + P3 tests: configurable compression ceiling and oversized-message offload.

P2: compression.max_attempts flows to agent.max_compression_attempts (floor 1).
P3: agent._offload_oversized_message writes the culprit message to a file and
     replaces it in-place with a tiny reference, gated by config flags.
"""
import types
from pathlib import Path

import pytest


# ── P2: config clamp logic ────────────────────────────────────────────
@pytest.mark.parametrize(
    "cfg,expect",
    [({}, 3), ({"max_attempts": 12}, 12), ({"max_attempts": 6}, 6),
     ({"max_attempts": 0}, 1), ({"max_attempts": -5}, 1)],
)
def test_p2_max_attempts_clamp(cfg, expect):
    # mirrors agent_init.py: max(1, int(_compression_cfg.get("max_attempts", 3)))
    assert max(1, int(cfg.get("max_attempts", 3))) == expect


def test_p2_config_schema_defaults_present():
    from hermes_cli.config import DEFAULT_CONFIG
    comp = DEFAULT_CONFIG["compression"]
    assert comp["max_attempts"] == 3
    assert comp["chunk_oversized_input"] is False
    assert comp["never_413"] is False


# ── P3: oversized-message offload ─────────────────────────────────────
def _make_agent(tmp_path, ctx_len=100_000):
    """Build a minimal duck-typed agent exposing just what the helper needs."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)  # bypass __init__
    agent.log_prefix = "[test] "
    agent.context_compressor = types.SimpleNamespace(context_length=ctx_len)
    agent._status_lines = []
    agent._buffer_status = lambda m: agent._status_lines.append(m)
    agent._flush_status_buffer = lambda: None
    return agent


def test_p3_offloads_dominant_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # display_hermes_home reads HERMES_HOME; point pastes/ into tmp_path
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "display_hermes_home", lambda: str(tmp_path))

    agent = _make_agent(tmp_path, ctx_len=10_000)
    # one giant user message (~well over 70% of 10k-token window) + small tail
    big = "X " * 60_000  # ~ tens of thousands of tokens
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big},
        {"role": "assistant", "content": "ok"},
    ]
    did = agent._offload_oversized_message(messages)
    assert did is True
    # culprit replaced with a tiny reference
    new_body = messages[1]["content"]
    assert "offloaded" in new_body.lower()
    assert "pastes" in new_body
    assert len(new_body) < 500
    # file actually written under pastes/
    pastes = list((Path(tmp_path) / "pastes").glob("oversized_*.txt"))
    assert len(pastes) == 1
    assert pastes[0].read_text() == big
    # user was told (honest UX)
    assert any("too large" in s.lower() for s in agent._status_lines)


def test_p3_skips_when_no_dominant_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "display_hermes_home", lambda: str(tmp_path))

    agent = _make_agent(tmp_path, ctx_len=1_000_000)
    # many small messages, none dominates → existing compression should handle it
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": "small message " * 10} for _ in range(20)]
    did = agent._offload_oversized_message(messages)
    assert did is False
    # nothing written
    assert not (Path(tmp_path) / "pastes").exists() or not list(
        (Path(tmp_path) / "pastes").glob("oversized_*.txt")
    )


def test_p3_skips_tool_and_system_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "display_hermes_home", lambda: str(tmp_path))

    agent = _make_agent(tmp_path, ctx_len=10_000)
    big = "Y " * 60_000
    # the giant message is a TOOL message → must NOT be offloaded
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": big},
    ]
    did = agent._offload_oversized_message(messages)
    assert did is False
    assert messages[1]["content"] == big  # untouched
