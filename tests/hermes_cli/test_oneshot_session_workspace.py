"""CLI one-shot workspace binding for per-session Docker isolation."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import hermes_cli.main as cli_main
import hermes_cli.oneshot as oneshot
from hermes_cli._parser import build_top_level_parser


def _parse(argv):
    parser, _subparsers, _chat_parser = build_top_level_parser()
    return parser.parse_args(argv)


def test_top_level_oneshot_accepts_max_turns():
    args = _parse(["--max-turns", "17", "-z", "run the reporter"])
    assert args.oneshot == "run the reporter"
    assert args.max_turns == 17

    before_chat = _parse(["--max-turns", "23", "chat", "-q", "prompt"])
    assert before_chat.max_turns == 23
    after_chat = _parse(["chat", "--max-turns", "19", "-q", "prompt"])
    assert after_chat.max_turns == 19


def test_chat_dispatch_preserves_max_turns_from_both_flag_positions(monkeypatch):
    import sys
    import types

    import hermes_cli.free_tier_bootstrap as free_tier_bootstrap
    import hermes_cli.observability.shared_metrics_consent as consent
    import hermes_cli.observability.shared_metrics_process as process_metrics

    fake_cli = types.ModuleType("cli")
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    for name in (
        "_apply_safe_mode", "_apply_user_config_bypass", "_guard_noninteractive_user_config",
        "_start_chat_background_prefetch", "_pin_kanban_board_env", "_warn_retired_xai_models",
        "_confirm_startup_expensive_model_override", "_read_query_file",
    ):
        monkeypatch.setattr(cli_main, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_main, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(cli_main, "_resolve_chat_session_args", lambda _args, use_tui: None)
    monkeypatch.setattr(cli_main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(free_tier_bootstrap, "run_bootstrap", lambda announce=False: None)
    monkeypatch.setattr(consent, "offer_consent_before_chat", lambda _args: None)
    monkeypatch.setattr(process_metrics, "begin_process", lambda _surface: None)

    for argv, expected in (
        (["--max-turns", "23", "chat", "-q", "prompt"], 23),
        (["chat", "--max-turns", "19", "-q", "prompt"], 19),
    ):
        parser, _subparsers, chat_parser = build_top_level_parser()
        chat_parser.set_defaults(func=cli_main.cmd_chat)
        args = parser.parse_args(argv)
        captured = {}
        setattr(fake_cli, "main", lambda **kwargs: captured.update(kwargs))

        cli_main.cmd_chat(args)

        assert captured["max_turns"] == expected


def test_oneshot_dispatch_passes_explicit_workspace_and_turn_limit(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr(cli_main, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(
        cli_main, "_resolve_chat_session_args",
        lambda args, use_tui: cli_main._apply_in_dir(args),
    )
    monkeypatch.setattr(
        cli_main, "_run_and_exit_oneshot",
        lambda prompt, **kwargs: captured.update(prompt=prompt, **kwargs),
    )
    args = SimpleNamespace(
        oneshot="run the reporter", in_dir=str(tmp_path), model=None, provider=None,
        toolsets="file,terminal", skills=None, usage_file="usage.json", resume=None,
        reasoning=None, max_turns=17, continue_last=None, no_restore_cwd=False,
        worktree=False,
    )

    cli_main._run_oneshot_from_args(args)

    assert captured["workspace_cwd"] == str(tmp_path.resolve())
    assert captured["max_turns"] == 17
    assert captured["usage_file"] == "usage.json"


def test_explicit_in_keeps_workspace_when_resuming_session(monkeypatch, tmp_path):
    workspace = tmp_path / "explicit-workspace"
    previous_session_cwd = tmp_path / "saved-session-cwd"
    workspace.mkdir()
    previous_session_cwd.mkdir()
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(cli_main, "_resolve_session_by_name_or_id", lambda session_id: session_id)
    monkeypatch.setattr(cli_main, "_resolve_continue_arg", lambda args, use_tui: None)
    monkeypatch.setattr(cli_main, "_import_foreign_resume", lambda _args: None)
    monkeypatch.setattr(
        cli_main, "_session_db",
        mock.Mock(side_effect=AssertionError("explicit --in must suppress saved-cwd restoration")),
    )
    monkeypatch.setattr(
        cli_main, "_run_and_exit_oneshot",
        lambda prompt, **kwargs: captured.update(prompt=prompt, **kwargs),
    )
    args = SimpleNamespace(
        oneshot="resume inside explicit workspace", in_dir=str(workspace), model=None, provider=None,
        toolsets="file,terminal", skills=None, usage_file=None, resume="session-123",
        reasoning=None, max_turns=17, continue_last=None, no_restore_cwd=False,
        worktree=False,
    )

    cli_main._run_oneshot_from_args(args)

    assert args.no_restore_cwd is True
    assert Path.cwd() == workspace.resolve()
    assert captured["workspace_cwd"] == str(workspace.resolve())


def test_resume_without_in_restores_saved_cwd_but_does_not_request_mount(monkeypatch, tmp_path):
    saved_cwd = tmp_path / "saved-session-cwd"
    saved_cwd.mkdir()
    captured = {}
    database = mock.MagicMock()
    database.get_session.return_value = {"cwd": str(saved_cwd)}
    db_context = mock.MagicMock()
    db_context.__enter__.return_value = database
    db_context.__exit__.return_value = False
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(cli_main, "_resolve_session_by_name_or_id", lambda session_id: session_id)
    monkeypatch.setattr(cli_main, "_resolve_continue_arg", lambda args, use_tui: None)
    monkeypatch.setattr(cli_main, "_import_foreign_resume", lambda _args: None)
    monkeypatch.setattr(cli_main, "_session_db", lambda: db_context)
    monkeypatch.setattr(
        cli_main, "_run_and_exit_oneshot",
        lambda prompt, **kwargs: captured.update(prompt=prompt, **kwargs),
    )
    args = SimpleNamespace(
        oneshot="resume session", in_dir=None, model=None, provider=None,
        toolsets="file,terminal", skills=None, usage_file=None, resume="session-123",
        reasoning=None, max_turns=17, continue_last=None, no_restore_cwd=False,
        worktree=False,
    )

    cli_main._run_oneshot_from_args(args)

    assert Path.cwd() == saved_cwd.resolve()
    assert captured["workspace_cwd"] is None


def test_run_oneshot_forwards_workspace_and_turn_limit(monkeypatch, tmp_path):
    with mock.patch.object(
        oneshot, "_run_agent",
        return_value=("done", {"final_response": "done", "completed": True, "failed": False}),
    ) as run_agent:
        result = oneshot.run_oneshot(
            "prompt", workspace_cwd=str(tmp_path), max_turns=17,
        )

    assert result == 0
    assert run_agent.call_args.kwargs["workspace_cwd"] == str(tmp_path)
    assert run_agent.call_args.kwargs["max_turns"] == 17


def test_run_oneshot_without_explicit_in_does_not_infer_process_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(
        oneshot, "_run_agent",
        return_value=("done", {"final_response": "done", "completed": True, "failed": False}),
    ) as run_agent:
        result = oneshot.run_oneshot("prompt")

    assert result == 0
    assert run_agent.call_args.kwargs["workspace_cwd"] is None


def test_one_shot_turn_mounts_only_its_session_workspace(monkeypatch, tmp_path):
    from tools import terminal_tool

    workspace = tmp_path / "run-worktree"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})

    register = getattr(oneshot, "_register_oneshot_workspace", None)
    clear = getattr(oneshot, "_clear_oneshot_workspace", None)
    assert callable(register) and callable(clear), "one-shot workspace lifecycle helpers are required"

    task_id = register("oneshot-session", str(workspace))
    config = {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": str(tmp_path),
    }
    assert isinstance(task_id, str)
    assert task_id == "oneshot-session"
    assert terminal_tool._resolve_task_host_cwd(config, task_id) == str(workspace.resolve())

    clear(task_id)
    assert terminal_tool._resolve_task_host_cwd(config, task_id) is None


def test_run_agent_applies_max_turns_and_session_workspace(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import hermes_constants
    import hermes_cli.config as config_module
    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.runtime_provider as runtime_provider
    import run_agent
    from tools import terminal_tool

    workspace = tmp_path / "run-worktree"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    config = {
        "env_type": "docker",
        "docker_mount_cwd_to_workspace": True,
        "host_cwd": str(tmp_path),
    }
    captured = {}
    control = {"raise": False}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = "oneshot-session"

        def run_conversation(self, prompt, conversation_history=None, task_id=None):
            assert task_id == self.session_id
            assert conversation_history is None  # preserve the one-shot empty-history contract
            assert terminal_tool._resolve_task_host_cwd(config, task_id) == str(workspace.resolve())
            if control["raise"]:
                raise RuntimeError("simulated turn failure")
            return {"final_response": "ok", "completed": True, "failed": False}

    monkeypatch.setattr(config_module, "load_config", lambda: {})
    monkeypatch.setattr(
        oneshot, "_resolve_model_and_provider",
        lambda *args, **kwargs: SimpleNamespace(
            model="model", provider=None, api_mode=None, base_url=None, api_key=None,
        ),
    )
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot, "_load_resume_target", lambda *args: (None, [], None))
    monkeypatch.setattr(oneshot, "_apply_stored_session_runtime", lambda choice, *a, **k: choice)
    monkeypatch.setattr(
        runtime_provider, "resolve_runtime_with_fallback", lambda *a, **k: ({"provider": None}, None),
    )
    monkeypatch.setattr(hermes_constants, "resolve_reasoning_config", lambda *a: None)
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **kwargs: None)
    monkeypatch.setattr(oneshot, "_normalize_toolsets", lambda _toolsets: ["file"])
    monkeypatch.setattr(oneshot, "_build_preloaded_skills_prompt", lambda _skills: None)
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _cfg: None)
    monkeypatch.setattr(oneshot, "_close_agent", lambda *_args: None)
    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    response, result = oneshot._run_agent(
        "prompt", toolsets=["file"], use_config_toolsets=False,
        workspace_cwd=str(workspace), max_turns=17,
    )

    assert response == "ok"
    assert result["completed"] is True
    assert captured["max_iterations"] == 17
    assert terminal_tool._resolve_task_host_cwd(config, "oneshot-session") is None

    control["raise"] = True
    try:
        oneshot._run_agent(
            "prompt", toolsets=["file"], use_config_toolsets=False,
            workspace_cwd=str(workspace), max_turns=17,
        )
    except RuntimeError as exc:
        assert str(exc) == "simulated turn failure"
    else:
        raise AssertionError("simulated run_conversation failure did not propagate")
    assert terminal_tool._resolve_task_host_cwd(config, "oneshot-session") is None
