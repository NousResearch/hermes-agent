from __future__ import annotations

import copy
import json
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agent import skill_utils
from tools import skills_tool


class _GrantAwareCLI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = "session-123"
        self.system_prompt = "base prompt"
        self.preloaded_skills = []
        self.conversation_history = [
            {"role": "user", "content": "prior-user-canary"},
            {"role": "assistant", "content": "prior-assistant-canary"},
        ]
        self.prompt_during_run = ""
        self.history_during_run = []

    def run(self):
        assert skill_utils.is_skill_read_granted("alpha") is True
        assert skill_utils.is_skill_read_granted("beta") is False
        self.prompt_during_run = self.system_prompt
        self.history_during_run = copy.deepcopy(self.conversation_history)


def _prepare_home(tmp_path, monkeypatch, disabled: str = "[alpha, beta]"):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(
        f"skills:\n  disabled: {disabled}\n",
        encoding="utf-8",
    )
    skill_dir = home / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: test\n---\n\nalpha\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    skill_utils._raw_config_cache_clear()
    return home, config_path


def _audit_events(home):
    return [
        json.loads(line)
        for line in (home / "logs" / "skill-grants.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_main_scopes_disabled_skill_grant_to_cli_session(tmp_path, monkeypatch):
    import cli as cli_mod

    home, config_path = _prepare_home(tmp_path, monkeypatch)
    original_config = config_path.read_bytes()
    original_disabled = skill_utils.get_disabled_skill_names()
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "build")
    skills_dir = home / "skills"
    alpha_dir = skills_dir / "alpha"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    (alpha_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: integration skill\n---\n\nalpha body\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _GrantAwareCLI(**kwargs)
        return created["cli"]

    def fake_build(skills, task_id=None):
        assert skills == ["alpha"]
        assert task_id == "session-123"
        assert skill_utils.is_skill_read_granted("alpha") is True
        loaded = json.loads(skills_tool.skill_view("alpha", preprocess=False))
        assert loaded["success"] is True
        assert "alpha body" in loaded["content"]
        return "alpha prompt", ["alpha"], []

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(cli_mod, "build_preloaded_skills_prompt", fake_build)

    cli_mod.main(skills="alpha", toolsets="safe")

    assert skill_utils.is_skill_read_granted("alpha") is False
    assert created["cli"].prompt_during_run == "base prompt\n\nalpha prompt"
    assert created["cli"].system_prompt == created["cli"].prompt_during_run
    assert created["cli"].history_during_run == created["cli"].conversation_history == [
        {"role": "user", "content": "prior-user-canary"},
        {"role": "assistant", "content": "prior-assistant-canary"},
    ]
    assert config_path.read_bytes() == original_config
    assert skill_utils.get_disabled_skill_names() == original_disabled
    events = _audit_events(home)
    assert events[0]["source"] == "cli"
    assert events[0]["session_id"] == "session-123"
    assert events[0]["task_id"] is None
    assert events[-1]["terminal_status"] == "completed"
    serialized = json.dumps(events)
    assert "prior-user-canary" not in serialized
    assert "prior-assistant-canary" not in serialized


def test_interactive_threads_preserve_selected_skill_grant(
    tmp_path, monkeypatch
):
    import signal
    import sys

    import cli as cli_mod
    import hermes_cli.plugins

    home, _ = _prepare_home(tmp_path, monkeypatch)
    beta_dir = home / "skills" / "beta"
    beta_dir.mkdir(parents=True)
    (beta_dir / "SKILL.md").write_text(
        "---\nname: beta\ndescription: sibling\n---\n\nbeta body\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "_hermes_home", home)
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda: None)
    monkeypatch.setattr(cli_mod, "_detect_light_mode", lambda: False)
    monkeypatch.setattr(cli_mod, "patch_stdout", lambda: nullcontext())
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.model_switch",
        SimpleNamespace(prewarm_picker_cache_async=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.curator",
        SimpleNamespace(maybe_run_curator=lambda **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.onboarding",
        SimpleNamespace(
            OPENCLAW_RESIDUE_FLAG="test-residue",
            detect_openclaw_residue=lambda: False,
            is_seen=lambda *_args, **_kwargs: True,
            mark_seen=lambda *_args, **_kwargs: None,
            openclaw_residue_hint_cli=lambda: "",
        ),
    )
    plugin_manager = SimpleNamespace(_cli_ref=None)
    monkeypatch.setattr(
        hermes_cli.plugins,
        "get_plugin_manager",
        lambda: plugin_manager,
    )

    inner_started = threading.Event()
    observations = {}
    cli_holder = {}

    class ToolCallingAgent:
        max_iterations = 1
        model = "unit-test-model"
        platform = "cli"

        def __init__(self):
            self.session_id = ""

        def run_conversation(self, **_kwargs):
            observations["agent_thread"] = threading.get_ident()
            observations["selected"] = json.loads(
                skills_tool.skill_view("alpha", preprocess=False)
            )
            observations["sibling"] = json.loads(
                skills_tool.skill_view("beta", preprocess=False)
            )
            inner_started.set()
            return {
                "final_response": "",
                "messages": [],
                "api_calls": 1,
                "completed": True,
                "failed": False,
            }

        def interrupt(self, *_args, **_kwargs):
            return None

    class InteractiveCLI(cli_mod.HermesCLI):
        def _claim_active_session(self, *_args, **_kwargs):
            return True

        def show_banner(self):
            return None

        def _show_security_advisories(self):
            return None

        def _console_print(self, *_args, **_kwargs):
            return None

        def _install_tool_callbacks(self):
            return None

        def _ensure_tirith_security(self):
            return None

        def _check_config_mcp_changes(self):
            return None

        def _print_user_message_preview(self, *_args, **_kwargs):
            observations["process_thread"] = threading.get_ident()

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, _message):
            return {
                "signature": "threaded-test-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **_kwargs):
            return True

        def _reset_stream_state(self):
            return None

        def _flush_credit_notices(self):
            return None

        def _flush_stream(self):
            return None

        def _pet_start_anim(self):
            return None

        def _pet_stop_anim(self):
            return None

        def _pet_react_turn_end(self):
            return None

        def _persist_active_session_before_close(self):
            return None

        def _print_exit_summary(self):
            return None

        def _release_active_session(self):
            return None

    agent = ToolCallingAgent()
    cli = InteractiveCLI(
        model="unit-test-model",
        toolsets=["safe"],
        provider="unit-test",
        api_key="unit-test-placeholder",
        base_url="http://127.0.0.1",
        max_turns=1,
        compact=True,
    )
    cli.agent = agent
    agent.session_id = cli.session_id
    cli._active_agent_route_signature = "threaded-test-route"
    cli._session_db = None
    cli_holder["cli"] = cli

    class TestApplication:
        def __init__(self, **_kwargs):
            self.is_running = True
            self.loop = None
            self.renderer = SimpleNamespace(
                cpr_not_supported_callback=lambda: None
            )
            self._on_resize = lambda: None

        def invalidate(self):
            return None

        def exit(self):
            self.is_running = False

        def run(self):
            cli._pending_input.put("inspect disabled skills")
            assert inner_started.wait(timeout=5)
            deadline = time.monotonic() + 5
            while cli._agent_running and time.monotonic() < deadline:
                time.sleep(0.01)
            assert cli._agent_running is False
            cli._should_exit = True
            self.is_running = False

    monkeypatch.setattr(cli_mod, "Application", TestApplication)
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **_kwargs: cli_holder["cli"])

    main_thread = threading.get_ident()
    cli_mod.main(
        skills=["alpha"],
        toolsets="safe",
        model="unit-test-model",
        provider="unit-test",
        api_key="unit-test-placeholder",
        base_url="http://127.0.0.1",
        max_turns=1,
        compact=True,
    )

    assert observations["selected"]["success"] is True
    assert "\nalpha\n" in observations["selected"]["content"]
    assert observations["sibling"]["success"] is False
    assert "beta body" not in json.dumps(observations["sibling"])
    assert observations["process_thread"] != main_thread
    assert observations["agent_thread"] not in {
        main_thread,
        observations["process_thread"],
    }
    assert skill_utils.current_skill_read_grant() is None
    assert skill_utils.is_skill_read_granted("alpha") is False
    assert skill_utils.is_skill_read_granted("beta") is False


def test_main_records_kanban_task_scope_for_disabled_skill(tmp_path, monkeypatch):
    import cli as cli_mod

    home, _ = _prepare_home(tmp_path, monkeypatch, "[alpha]")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    monkeypatch.setenv("HERMES_PROFILE", "build")
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kwargs: _GrantAwareCLI(**kwargs))
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("alpha prompt", ["alpha"], []),
    )

    cli_mod.main(skills=["alpha"], toolsets="safe")

    events = _audit_events(home)
    assert events[0]["source"] == "kanban"
    assert events[0]["task_id"] == "task-123"
    assert events[0]["profile"] == "build"
    assert events[-1]["terminal_status"] == "completed"


def test_preload_exception_closes_grant_immediately(tmp_path, monkeypatch):
    import cli as cli_mod

    home, _ = _prepare_home(tmp_path, monkeypatch, "[alpha]")
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kwargs: _GrantAwareCLI(**kwargs))

    def fail_preload(*_args, **_kwargs):
        assert skill_utils.is_skill_read_granted("alpha") is True
        raise RuntimeError("preload exploded")

    monkeypatch.setattr(cli_mod, "build_preloaded_skills_prompt", fail_preload)
    with pytest.raises(RuntimeError, match="preload exploded"):
        cli_mod.main(skills=["alpha"], toolsets="safe")

    assert skill_utils.is_skill_read_granted("alpha") is False
    assert _audit_events(home)[-1]["terminal_status"] == "failed"


def test_default_profile_attribution_is_not_dot_hermes(tmp_path, monkeypatch):
    import cli as cli_mod

    _prepare_home(tmp_path, monkeypatch, "[alpha]")
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    assert cli_mod._active_skill_grant_profile() == "default"


def test_profile_attribution_uses_profile_home_without_env_hint(tmp_path, monkeypatch):
    import cli as cli_mod

    home = tmp_path / ".hermes" / "profiles" / "build"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [alpha]\n", encoding="utf-8"
    )
    skill_dir = home / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: test\n---\n\nalpha\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    skill_utils._raw_config_cache_clear()
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kwargs: _GrantAwareCLI(**kwargs))
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda *_args, **_kwargs: ("alpha prompt", ["alpha"], []),
    )

    cli_mod.main(skills=["alpha"], toolsets="safe")

    assert _audit_events(home)[0]["profile"] == "build"


def test_interactive_cancellation_return_is_audited_as_cancelled(
    tmp_path, monkeypatch
):
    import cli as cli_mod

    home, _ = _prepare_home(tmp_path, monkeypatch, "[alpha]")

    class CancelledCLI(_GrantAwareCLI):
        def run(self):
            assert skill_utils.is_skill_read_granted("alpha") is True
            self._last_run_cancelled = True

    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kwargs: CancelledCLI(**kwargs))
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda *_args, **_kwargs: ("alpha prompt", ["alpha"], []),
    )

    cli_mod.main(skills=["alpha"], toolsets="safe")

    assert _audit_events(home)[-1]["terminal_status"] == "cancelled"


def test_system_exit_130_is_audited_as_cancelled(tmp_path, monkeypatch):
    import cli as cli_mod

    home, _ = _prepare_home(tmp_path, monkeypatch, "[alpha]")

    @cli_mod._skill_grant_lifecycle
    def cancelled_invocation():
        skill_utils.issue_skill_read_grant(
            ["alpha"],
            session_id="quiet-session",
            task_id="task-quiet",
            profile="build",
            requester="build",
            source="kanban",
        )
        raise SystemExit(130)

    with pytest.raises(SystemExit) as exc_info:
        cancelled_invocation()

    assert exc_info.value.code == 130
    assert skill_utils.is_skill_read_granted("alpha") is False
    assert _audit_events(home)[-1]["terminal_status"] == "cancelled"


def test_quiet_single_query_exit_zero_closes_grant_as_completed(
    tmp_path, monkeypatch
):
    import cli as cli_mod

    home, _ = _prepare_home(tmp_path, monkeypatch, "[alpha]")

    class QuietCLI(_GrantAwareCLI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.provider = "test-provider"
            self.model = "test-model"
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id=self.session_id,
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=self._run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return surface == "cli" and stderr is True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, effective_query):
            assert effective_query == "hello"
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **_kwargs):
            return True

        def _run_conversation(self, *, user_message, conversation_history):
            assert user_message == "hello"
            assert conversation_history == self.conversation_history
            assert skill_utils.is_skill_read_granted("alpha") is True
            return {"final_response": "done", "failed": False}

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", QuietCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda *_args, **_kwargs: ("alpha prompt", ["alpha"], []),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(
            query="hello",
            quiet=True,
            skills=["alpha"],
            toolsets="safe",
        )

    assert exc_info.value.code == 0
    assert skill_utils.current_skill_read_grant() is None
    assert skill_utils.is_skill_read_granted("alpha") is False
    closed = [event for event in _audit_events(home) if event["event"] == "closed"]
    assert [event["terminal_status"] for event in closed] == ["completed"]
