"""run_claude_code_turn — prompt combination and the warm-process registry.

Drives the runtime with minimal agent stand-ins and the fake `claude` CLI so
the two live gateway bugs stay fixed: the gateway's ephemeral system prompt
must reach the child, and a session must keep ONE warm process across the
per-request AIAgent instances api_server builds.
"""

from __future__ import annotations

import json
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import claude_code_runtime as rt

_FAKE = Path(__file__).parent / "transports" / "fake_claude_cli.py"


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-setup-token")
    wrapper = tmp_path / "claude"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"exec {json.dumps(sys.executable)} {json.dumps(str(_FAKE))} \"$@\"\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(rt, "_claude_code_config", lambda: {"binary": str(wrapper), "expose_hermes_tools": False})
    # Fresh registry per test.
    with rt._REGISTRY_LOCK:
        rt._REGISTRY.clear()
    yield home
    for key in list(rt._REGISTRY):
        rt.evict_session(key)


def _agent(session_id: str, *, cached="BASE-PROMPT", ephemeral=None) -> SimpleNamespace:
    """The attributes run_claude_code_turn touches on a real AIAgent."""
    a = SimpleNamespace(
        session_id=session_id,
        _cached_system_prompt=cached,
        ephemeral_system_prompt=ephemeral,
        model="sonnet",
        api_mode="claude_code",
        provider="claude-code-cli",
        base_url="claude-code://local",
        api_key="x",
        _interrupt_requested=False,
        _interrupt_message=None,
        _skill_nudge_interval=0,
        _iters_since_skill=0,
        valid_tool_names=set(),
        _session_db=None,
        session_api_calls=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_cost_status=None, session_cost_source=None,
        context_compressor=None,
        show_commentary=True,
    )
    a.clear_interrupt = lambda: None
    a._sync_external_memory_for_turn = lambda **kw: None
    a._spawn_background_review = lambda **kw: None
    return a


def _turn(agent, text="hello"):
    messages = [{"role": "user", "content": text}]
    return rt.run_claude_code_turn(
        agent, user_message=text, original_user_message=text,
        messages=messages, effective_task_id="t",
    )


def _prompt_file_text(session) -> str:
    return Path(session._system_prompt_path).read_text()


class TestEphemeralPrompt:
    def test_ephemeral_prompt_reaches_the_child(self):
        agent = _agent("s1", ephemeral="MARKER-EPH")
        result = _turn(agent)
        assert result["completed"] is True
        text = _prompt_file_text(agent._claude_code_session)
        assert text == "BASE-PROMPT\n\nMARKER-EPH"  # same join as conversation_loop
        assert agent._claude_code_session.system_prompt == rt.combined_system_prompt(agent)

    def test_changed_ephemeral_prompt_respawns_with_new_content(self):
        first = _agent("s2", ephemeral="MARKER-ONE")
        _turn(first)
        session = first._claude_code_session
        pid_one = session.pid
        second = _agent("s2", ephemeral="MARKER-TWO")
        _turn(second)
        assert second._claude_code_session is session  # same registry entry
        assert session.pid != pid_one  # respawned
        assert _prompt_file_text(session).endswith("MARKER-TWO")


class TestRegistry:
    def test_two_agent_instances_share_one_process(self):
        a1 = _agent("shared", ephemeral="E")
        r1 = _turn(a1, "one")
        pid = a1._claude_code_session.pid
        a2 = _agent("shared", ephemeral="E")
        r2 = _turn(a2, "two")
        assert r1["completed"] and r2["completed"]
        assert a2._claude_code_session is a1._claude_code_session
        assert a2._claude_code_session.pid == pid  # one spawn, same pid
        assert rt.registered_session_count() == 1

    def test_different_sessions_get_different_processes(self):
        a1, a2 = _agent("A"), _agent("B")
        _turn(a1)
        _turn(a2)
        assert a1._claude_code_session is not a2._claude_code_session
        assert a1._claude_code_session.pid != a2._claude_code_session.pid
        assert rt.registered_session_count() == 2

    def test_idle_timeout_evicts(self):
        a = _agent("idle")
        _turn(a)
        session = a._claude_code_session
        assert session.is_alive()
        assert rt.sweep_idle_sessions(0.0, now=time.monotonic() + 1) == 1
        assert rt.registered_session_count() == 0
        assert not session.is_alive()
        # Next turn for that session rebuilds transparently.
        b = _agent("idle")
        assert _turn(b)["completed"] is True
        assert b._claude_code_session is not session

    def test_retire_evicts(self):
        a = _agent("crash")
        result = _turn(a, "please CRASH")
        assert result["completed"] is False
        assert rt.registered_session_count() == 0
        assert a._claude_code_session is None

    def test_hooks_rebound_to_the_current_agent(self):
        a1 = _agent("hooks")
        seen1, seen2 = [], []
        a1._fire_stream_delta = seen1.append
        _turn(a1, "first")
        a2 = _agent("hooks")
        a2._fire_stream_delta = seen2.append
        _turn(a2, "second")
        assert "".join(seen1) == "echo: first"
        assert "".join(seen2) == "echo: second"


class TestRegistryHardening:
    def test_retire_happens_under_the_turn_lock(self, monkeypatch, tmp_path):
        """A waiter on a retired session must get a fresh process, never the
        one being closed underneath it."""
        import threading

        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "turn_timeout": 1.0, "silence_timeout": 0.4})
        a = _agent("retire")
        _turn(a, "warm")
        first_pid = a._claude_code_session.pid
        results = {}

        def waiter():
            b = _agent("retire")
            results["b"] = _turn(b, "after")
            results["pid"] = b._claude_code_session.pid

        t = threading.Thread(target=waiter)
        # A's turn hangs -> silence timeout -> should_retire; B is queued on the lock.
        import time as _t
        threading.Timer(0.1, t.start).start()
        ra = _turn(a, "HANG")
        t.join(timeout=30)
        assert ra["completed"] is False and "no output" in (ra["error"] or "")
        assert results["b"]["completed"] is True
        assert "exited unexpectedly" not in (results["b"]["error"] or "")
        assert results["pid"] != first_pid

    def test_max_sessions_lru_eviction(self, monkeypatch, caplog):
        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "max_sessions": 2})
        a1, a2, a3 = _agent("L1"), _agent("L2"), _agent("L3")
        _turn(a1); _turn(a2)
        s1 = a1._claude_code_session
        with caplog.at_level("INFO", logger="agent.claude_code_runtime"):
            _turn(a3)
        assert rt.registered_session_count() == 2
        assert not s1.is_alive()
        assert "evicting LRU session L1" in caplog.text
        with rt._REGISTRY_LOCK:
            assert set(rt._REGISTRY) == {"L2", "L3"}

    def test_respawn_rate_guard_warns_once(self, caplog):
        with caplog.at_level("WARNING", logger="agent.claude_code_runtime"):
            for i in range(6):
                _turn(_agent("dyn", ephemeral=f"per-request-{i}"))
        warnings = [r for r in caplog.records if "changes every request" in r.getMessage()]
        assert len(warnings) == 1

    def test_shutdown_registry_closes_children_and_temp_files(self):
        a = _agent("exit")
        _turn(a)
        s = a._claude_code_session
        prompt = s._system_prompt_path
        assert prompt and Path(prompt).exists()
        rt._shutdown_registry()
        assert rt.registered_session_count() == 0
        assert not s.is_alive()
        assert not Path(prompt).exists()

    def test_prune_stale_temp_files(self, tmp_path):
        import os, time as _t
        cfg = tmp_path / "cc"
        cfg.mkdir()
        old = cfg / "system-prompt-old.md"; old.write_text("x")
        old_mcp = cfg / "hermes-claude-mcp-old.json"; old_mcp.write_text("{}")
        fresh = cfg / "system-prompt-new.md"; fresh.write_text("y")
        stale = _t.time() - 2 * 24 * 3600
        os.utime(old, (stale, stale)); os.utime(old_mcp, (stale, stale))
        assert rt.prune_stale_temp_files(str(cfg)) == 2
        assert fresh.exists() and not old.exists() and not old_mcp.exists()
