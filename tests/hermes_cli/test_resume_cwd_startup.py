"""Classic CLI startup must honor cwd overrides through history/agent initialization."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def launch_resumed_cli(monkeypatch, tmp_path):
    import cli
    from hermes_cli import main
    from hermes_cli._parser import build_top_level_parser
    from hermes_state_registry import acquire

    saved = tmp_path / "A"
    target = tmp_path / "B"
    saved.mkdir()
    target.mkdir()
    monkeypatch.chdir(target)
    monkeypatch.setenv("TERMINAL_CWD", str(target))
    db = acquire()
    db.create_session("saved-session", source="cli", cwd=str(saved), model="saved-model")
    db.append_message("saved-session", "user", content="Remember this workspace.")
    db.append_message("saved-session", "assistant", content="Remembered.")

    # Replace external services, not argument forwarding, construction, DB reads,
    # resume loading, or either of the real cwd operations under test.
    monkeypatch.setattr(main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", lambda **kw: None)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_consent.offer_consent_before_chat",
        lambda args: None,
    )
    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(cli.HermesCLI, "_install_tool_callbacks", lambda self: None)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **kw: None,
    )
    monkeypatch.setattr("agent.credits_tracker.seed_credits_at_session_start", lambda agent: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "custom", "api_key": "test-key",
            "base_url": "http://example.invalid/v1", "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(cli, "_active_agent_ref", None)
    agent_factory = MagicMock()
    monkeypatch.setattr("run_agent.AIAgent", agent_factory)

    def launch(flags, *, preload=False, direct=False):
        initialized = []

        def initialize(cli_obj, *args, **kwargs):
            if preload:
                assert cli_obj._preload_resumed_session()
            assert cli_obj._init_agent()
            initialized.append(cli_obj)

        # Stop at the first turn boundary, after the real classic CLI constructor
        # and agent initialization, without making an LLM request or opening a REPL.
        monkeypatch.setattr(cli, "_run_single_query_mode", initialize)
        parser, _, _ = build_top_level_parser()
        args = parser.parse_args([
            "chat", "--cli", "--resume", "saved-session", "--ignore-rules",
            "-q", "Continue", *flags,
        ])
        if direct:
            cli.main(
                resume=args.resume, query=args.query, ignore_rules=True,
                no_restore_cwd=args.no_restore_cwd,
            )
        else:
            main.cmd_chat(args)
        cli_obj, = initialized
        assert cli_obj.session_id == "saved-session"
        assert [m["content"] for m in cli_obj.conversation_history] == [
            "Remember this workspace.", "Remembered.",
        ]
        assert agent_factory.call_args.kwargs["model"] == "saved-model"
        return cli_obj

    yield launch, saved, target
    db.close()


@pytest.mark.parametrize("preload", [False, True], ids=["late", "early"])
@pytest.mark.parametrize("override", ["in", "no-restore"])
def test_startup_resume_preserves_explicit_cwd(launch_resumed_cli, override, preload):
    launch, _, target = launch_resumed_cli
    flags = ["--in", str(target)] if override == "in" else ["--no-restore-cwd"]

    launch(flags, preload=preload)

    assert (Path.cwd().resolve(), Path(os.environ["TERMINAL_CWD"]).resolve()) == (
        target.resolve(), target.resolve(),
    )


@pytest.mark.parametrize("preload", [False, True], ids=["late", "early"])
@pytest.mark.parametrize("direct", [False, True], ids=["cmd-chat", "cli-main"])
def test_startup_no_restore_cwd_retargets_inherited_terminal_cwd(
    launch_resumed_cli, preload, direct, monkeypatch,
):
    launch, saved, target = launch_resumed_cli
    monkeypatch.setenv("TERMINAL_CWD", str(saved))

    launch(["--no-restore-cwd"], preload=preload, direct=direct)

    assert (Path.cwd().resolve(), Path(os.environ["TERMINAL_CWD"]).resolve()) == (
        target.resolve(), target.resolve(),
    )


@pytest.mark.parametrize("preload", [False, True], ids=["late", "early"])
def test_startup_no_restore_cwd_keeps_terminal_cwd_unset(launch_resumed_cli, preload, monkeypatch):
    launch, _, target = launch_resumed_cli
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    launch(["--no-restore-cwd"], preload=preload)

    assert Path.cwd().resolve() == target.resolve()
    assert "TERMINAL_CWD" not in os.environ


@pytest.mark.parametrize("preload", [False, True], ids=["late", "early"])
def test_default_cli_resume_restores_saved_cwd(launch_resumed_cli, preload):
    launch, saved, _ = launch_resumed_cli
    # The direct classic CLI entry must also restore cwd, without argparse's
    # earlier chdir masking whether the history loader still does its job.
    launch([], preload=preload, direct=True)

    assert (Path.cwd().resolve(), Path(os.environ["TERMINAL_CWD"]).resolve()) == (
        saved.resolve(), saved.resolve(),
    )


@pytest.mark.parametrize("preload", [False, True], ids=["late", "early"])
def test_default_resume_already_in_saved_cwd_retargets_tools(launch_resumed_cli, monkeypatch, preload):
    from agent.runtime_cwd import resolve_agent_cwd

    launch, saved, target = launch_resumed_cli
    monkeypatch.chdir(saved)
    monkeypatch.setenv("TERMINAL_CWD", str(target))

    launch([], preload=preload, direct=True)

    assert Path.cwd().resolve() == saved.resolve()
    assert Path(os.environ["TERMINAL_CWD"]).resolve() == saved.resolve()
    assert resolve_agent_cwd().resolve() == saved.resolve()


@pytest.mark.parametrize("preload", [False, True], ids=["late", "early"])
@pytest.mark.parametrize("terminal_cwd", [None, "", "   "], ids=["absent", "empty", "whitespace"])
def test_default_resume_already_in_saved_cwd_preserves_unconfigured_tools(
    launch_resumed_cli, monkeypatch, preload, terminal_cwd,
):
    from agent.runtime_cwd import resolve_context_cwd

    launch, saved, _ = launch_resumed_cli
    monkeypatch.chdir(saved)
    if terminal_cwd is None:
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
    else:
        monkeypatch.setenv("TERMINAL_CWD", terminal_cwd)

    launch([], preload=preload, direct=True)

    assert Path.cwd().resolve() == saved.resolve()
    assert os.environ.get("TERMINAL_CWD") == terminal_cwd
    assert resolve_context_cwd() is None


@pytest.mark.parametrize("preload", [False, True], ids=["late", "early"])
@pytest.mark.parametrize("direct", [False, True], ids=["cmd-chat", "cli-main"])
@pytest.mark.parametrize("terminal_cwd", ["", "   "], ids=["empty", "whitespace"])
def test_startup_no_restore_preserves_blank_terminal_cwd(
    launch_resumed_cli, monkeypatch, preload, direct, terminal_cwd,
):
    from agent.runtime_cwd import resolve_context_cwd

    launch, _, target = launch_resumed_cli
    monkeypatch.setenv("TERMINAL_CWD", terminal_cwd)

    launch(["--no-restore-cwd"], preload=preload, direct=direct)

    assert Path.cwd().resolve() == target.resolve()
    assert os.environ["TERMINAL_CWD"] == terminal_cwd
    assert resolve_context_cwd() is None


@pytest.mark.parametrize("command", ["resume", "sessions"])
@pytest.mark.parametrize("override", ["in", "no-restore"])
def test_mid_chat_resume_restores_cwd_after_launch_override(launch_resumed_cli, override, command):
    launch, saved, target = launch_resumed_cli
    flags = ["--in", str(target)] if override == "in" else ["--no-restore-cwd"]
    cli_obj = launch(flags)
    cli_obj._session_db.create_session("other-session", source="cli", cwd=str(saved))
    cli_obj._session_db.append_message("other-session", "user", content="Other history.")
    cli_obj.resume_display = "minimal"

    cli_obj.process_command(f"/{command} other-session")

    assert cli_obj.session_id == "other-session"
    assert cli_obj.conversation_history[0]["content"] == "Other history."
    assert (Path.cwd().resolve(), Path(os.environ["TERMINAL_CWD"]).resolve()) == (
        saved.resolve(), saved.resolve(),
    )


@pytest.mark.parametrize("command", ["resume", "sessions"])
def test_mid_chat_resume_same_directory_retargets_tools(launch_resumed_cli, monkeypatch, command):
    from agent.runtime_cwd import resolve_agent_cwd

    launch, saved, target = launch_resumed_cli
    cli_obj = launch(["--no-restore-cwd"])
    cli_obj._session_db.create_session("other-session", source="cli", cwd=str(target))
    cli_obj._session_db.append_message("other-session", "user", content="Same-directory history.")
    cli_obj.resume_display = "minimal"
    monkeypatch.setenv("TERMINAL_CWD", str(saved))

    cli_obj.process_command(f"/{command} other-session")

    assert cli_obj.session_id == "other-session"
    assert cli_obj.conversation_history[0]["content"] == "Same-directory history."
    assert Path.cwd().resolve() == target.resolve()
    assert Path(os.environ["TERMINAL_CWD"]).resolve() == target.resolve()
    assert resolve_agent_cwd().resolve() == target.resolve()
