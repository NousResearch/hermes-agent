"""Focused contracts for the /objective Executive-v2 CLI surface."""

from types import SimpleNamespace
from unittest.mock import patch


def test_objective_dispatch_routes_to_executive_v2_dry_run():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "objective-dispatch-test"

    command = "/objective --dry-run Ship the release"

    with (
        patch("hermes_cli.plugins.fire_pre_command_hook") as pre_command,
        patch.object(
            cli,
            "_handle_executive_v2_dryrun",
        ) as dry_run,
    ):
        result = cli.process_command(command)

    assert result is True
    dry_run.assert_called_once_with(command)
    pre_command.assert_called_once()


def test_objective_handler_is_preview_only_and_never_persists():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    cli = CLICommandsMixin()
    cli._agent = None
    cli.agent = SimpleNamespace(
        session_id="objective-dry-run-session"
    )

    state = object()

    with (
        patch(
            "hermes_cli.config.read_raw_config",
            return_value={
                "agent": {
                    "executive_v2_enabled": True,
                }
            },
        ),
        patch(
            "agent.executive.flag.resolve_v2_enabled",
            return_value=True,
        ) as resolve_enabled,
        patch(
            "agent.executive.objective_engine.ObjectiveEngine"
        ) as engine_cls,
        patch(
            "agent.executive.dryrun.render_dry_run",
            return_value=(
                "preview\n"
                "/objective persist forbidden\n"
                "/objective cancel forbidden"
            ),
        ),
        patch("cli._cprint") as cprint,
    ):
        engine = engine_cls.return_value
        engine.run_pipeline.return_value = "objective-id"
        engine.get_state.return_value = state

        cli._handle_executive_v2_dryrun(
            "/objective --dry-run Ship the release"
        )

    resolve_enabled.assert_called_once_with(
        cli.agent,
        config_value=True,
    )

    engine_cls.assert_called_once_with(
        user_id="objective-dry-run-session",
        enabled=True,
    )

    engine.run_pipeline.assert_called_once_with(
        "Ship the release",
        persist_to_state_meta=False,
    )

    engine.get_state.assert_called_once_with(
        "objective-id"
    )

    rendered = "\n".join(
        str(call.args[0])
        for call in cprint.call_args_list
        if call.args
    )

    assert "/objective persist" not in rendered
    assert "/objective cancel" not in rendered
    assert "persist/cancel are not supported" in rendered


def test_objective_unset_config_preserves_legacy_enable_bridge():
    from agent.executive.flag import CONFIG_UNSET
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    cli = CLICommandsMixin()
    cli._agent = None
    cli.agent = SimpleNamespace(
        session_id="objective-bridge-session"
    )

    with (
        patch(
            "hermes_cli.config.read_raw_config",
            return_value={},
        ),
        patch(
            "agent.executive.flag.resolve_v2_enabled",
            return_value=True,
        ) as resolve_enabled,
        patch(
            "agent.executive.objective_engine.ObjectiveEngine"
        ) as engine_cls,
        patch(
            "agent.executive.dryrun.render_dry_run",
            return_value="preview",
        ),
        patch("cli._cprint"),
    ):
        engine = engine_cls.return_value
        engine.run_pipeline.return_value = "oid"
        engine.get_state.return_value = object()

        cli._handle_executive_v2_dryrun(
            "/objective --dry-run Bridge contract"
        )

    resolve_enabled.assert_called_once_with(
        cli.agent,
        config_value=CONFIG_UNSET,
    )

    engine.run_pipeline.assert_called_once_with(
        "Bridge contract",
        persist_to_state_meta=False,
    )
