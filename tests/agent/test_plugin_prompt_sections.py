from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap

from agent.system_prompt import build_system_prompt, invalidate_system_prompt
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from run_agent import AIAgent


def _real_agent(*, session_id: str = "plugin-section-test") -> AIAgent:
    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        provider="openrouter",
        platform="cli",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id=session_id,
    )


def _install_test_section(manager: PluginManager, content) -> None:
    manager._discovered = True
    ctx = PluginContext(
        PluginManifest(name="example-plugin", key="example-plugin", source="user"),
        manager,
    )
    ctx.register_system_prompt_section(
        "example.rules",
        content,
        position="after_memory",
        max_chars=1000,
    )


def test_real_aiagent_builds_section_once_and_keeps_it_out_of_static_prefix(monkeypatch):
    # Pin the workspace snapshot: build_coding_workspace_block shells out to
    # live `git status`/`git log` on every build, and a git call failing or
    # timing out under xdist contention makes the two builds differ in the
    # Branch/Recent-commits lines — a flake unrelated to what this test
    # asserts (plugin sections). Byte-stability of the REAL workspace block
    # is coding_context's contract, covered by its own tests.
    monkeypatch.setattr(
        "agent.coding_context.build_coding_workspace_block",
        lambda cwd=None: "Workspace (snapshot at session start):\n- Root: /pinned",
    )
    calls = []

    def section(session_info):
        calls.append(dict(session_info))
        return f"rules render {len(calls)}"

    manager = PluginManager()
    _install_test_section(manager, section)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    agent = _real_agent()

    first = build_system_prompt(agent)
    invalidate_system_prompt(agent)
    rebuilt = build_system_prompt(agent)

    assert first == rebuilt
    assert len(calls) == 1
    assert calls[0]["session_id"] == agent.session_id
    assert "## Plugin Context: example.rules" in first
    assert "rules render 1" in first
    assert first.index("## Plugin Context: example.rules") < first.index("Conversation started:")
    assert "example.rules" not in agent._cached_system_prompt_static


def test_fresh_process_resume_restores_identical_full_prompt_without_callback(tmp_path):
    """The existing persisted full prompt is the only resume state required."""
    db_path = tmp_path / "state.db"
    calls_path = tmp_path / "calls.txt"
    child = textwrap.dedent(
        """
        import base64
        import json
        import os
        from pathlib import Path

        from agent.conversation_loop import _restore_or_build_system_prompt
        from agent.system_prompt import build_system_prompt, invalidate_system_prompt
        from hermes_cli import plugins
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=Path(os.environ["TEST_DB"]))
        session_id = "resume-plugin-section"
        db.ensure_session(session_id, source="cli", model="test/model")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="openrouter",
            platform="cli",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id=session_id,
            session_db=db,
        )

        manager = PluginManager()
        manager._discovered = True
        ctx = PluginContext(
            PluginManifest(name="example-plugin", key="example-plugin", source="user"),
            manager,
        )
        calls_path = Path(os.environ["TEST_CALLS"])
        def render(_session_info):
            count = int(calls_path.read_text() or "0") if calls_path.exists() else 0
            calls_path.write_text(str(count + 1))
            return "original bytes" if os.environ["TEST_PHASE"] == "first" else "CHANGED"
        ctx.register_system_prompt_section("example.rules", render, position="after_memory")
        plugins._plugin_manager = manager

        history = [] if os.environ["TEST_PHASE"] == "first" else [
            {"role": "user", "content": "already persisted"}
        ]
        _restore_or_build_system_prompt(agent, None, history)
        restored = agent._cached_system_prompt
        rebuilt_equal = None
        if os.environ["TEST_PHASE"] != "first":
            invalidate_system_prompt(agent)
            rebuilt = build_system_prompt(agent)
            rebuilt_equal = rebuilt == restored
            agent._cached_system_prompt = rebuilt
        print(json.dumps({
            "prompt_b64": base64.b64encode(
                agent._cached_system_prompt.encode("utf-8")
            ).decode("ascii"),
            "calls": int(calls_path.read_text()),
            "rebuilt_equal": rebuilt_equal,
        }))
        db.close()
        """
    )

    outputs = []
    for phase in ("first", "resume"):
        env = os.environ.copy()
        env.update(
            HERMES_HOME=str(tmp_path / "hermes-home"),
            TEST_DB=str(db_path),
            TEST_CALLS=str(calls_path),
            TEST_PHASE=phase,
        )
        proc = subprocess.run(
            [sys.executable, "-c", child],
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
            check=True,
        )
        outputs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    first_prompt = base64.b64decode(outputs[0]["prompt_b64"])
    resumed_prompt = base64.b64decode(outputs[1]["prompt_b64"])
    if first_prompt != resumed_prompt:
        diff_at = next(
            i for i, (left, right) in enumerate(zip(first_prompt, resumed_prompt))
            if left != right
        )
        raise AssertionError(
            f"prompt bytes differ at {diff_at}: "
            f"first={first_prompt[diff_at - 100:diff_at + 100]!r} "
            f"resumed={resumed_prompt[diff_at - 100:diff_at + 100]!r}"
        )
    assert b"original bytes" in first_prompt
    assert b"CHANGED" not in resumed_prompt
    assert outputs[0]["calls"] == outputs[1]["calls"] == 1
    assert outputs[1]["rebuilt_equal"] is True
