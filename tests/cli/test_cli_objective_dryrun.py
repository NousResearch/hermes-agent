"""CLI tests for Executive v2 /objective dry-run wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_cli.commands import resolve_command


class FakeObjectiveEngine:
    instances: list["FakeObjectiveEngine"] = []

    def __init__(self, *, user_id: str, enabled: bool) -> None:
        self.user_id = user_id
        self.enabled = enabled
        self.run_pipeline = MagicMock(return_value="oid-1")
        self.get_state = MagicMock(return_value=SimpleNamespace(objective_id="oid-1"))
        self.persist = MagicMock()
        self.archive = MagicMock()
        self.list_persisted = MagicMock(return_value=[])
        self.apply = MagicMock()
        FakeObjectiveEngine.instances.append(self)


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = SimpleNamespace(session_id="session-1")
    cli_obj._agent = cli_obj.agent
    cli_obj.conversation_history = []
    cli_obj.session_id = "session-1"
    cli_obj._pending_input = MagicMock()
    cli_obj._pending_resume_sessions = None
    return cli_obj


def _run_objective(command: str):
    FakeObjectiveEngine.instances.clear()
    cli_obj = _make_cli()
    printed: list[str] = []
    dryrun_text = (
        "DRY RUN RENDERED\n"
        "│ /objective persist <objective_id>  to save this\n"
        "│ /objective cancel                  to discard"
    )
    with patch("cli._cprint", side_effect=lambda text: printed.append(str(text))), \
         patch("agent.executive.flag.resolve_v2_enabled", return_value=True), \
         patch("agent.executive.objective_engine.ObjectiveEngine", FakeObjectiveEngine), \
         patch("agent.executive.dryrun.render_dry_run", return_value=dryrun_text) as render:
        result = cli_obj.process_command(command)
    engine = FakeObjectiveEngine.instances[0] if FakeObjectiveEngine.instances else None
    return result, engine, render, printed


def test_objective_registered_and_wired_to_cli_handler():
    cmd = resolve_command("objective")
    assert cmd is not None
    assert cmd.args_hint == "[--dry-run] <objective>"

    cli_obj = _make_cli()
    with patch.object(cli_obj, "_handle_executive_v2_dryrun") as handler:
        assert cli_obj.process_command("/objective implement API") is True
    handler.assert_called_once_with("/objective implement API")


def test_objective_preserves_multi_word_goal_without_flag():
    _, engine, render, printed = _run_objective("/objective implement API")

    engine.run_pipeline.assert_called_once_with(
        "implement API",
        persist_to_state_meta=False,
    )
    engine.get_state.assert_called_once_with("oid-1")
    render.assert_called_once_with(engine.get_state.return_value)
    rendered_output = "\n".join(printed)
    assert "DRY RUN RENDERED" in rendered_output
    assert "/objective persist" not in rendered_output
    assert "/objective cancel" not in rendered_output
    assert "persist/cancel are not supported" in rendered_output
    engine.persist.assert_not_called()
    engine.archive.assert_not_called()
    engine.apply.assert_not_called()


def test_objective_dry_run_flag_preserves_multi_word_goal():
    _, engine, render, _ = _run_objective("/objective --dry-run implement API")

    engine.run_pipeline.assert_called_once_with(
        "implement API",
        persist_to_state_meta=False,
    )
    render.assert_called_once()
    engine.persist.assert_not_called()
    engine.archive.assert_not_called()
    engine.apply.assert_not_called()


def test_objective_preserves_quoted_goal_text():
    _, engine, _, _ = _run_objective('/objective "implement API"')

    engine.run_pipeline.assert_called_once_with(
        "implement API",
        persist_to_state_meta=False,
    )


def test_objective_preserves_spanish_goal_text():
    _, engine, _, _ = _run_objective("/objective diseñar modo objetivo dry-run")

    engine.run_pipeline.assert_called_once_with(
        "diseñar modo objetivo dry-run",
        persist_to_state_meta=False,
    )


def test_objective_without_goal_shows_usage_and_does_not_run_pipeline():
    _, engine, render, printed = _run_objective("/objective")

    assert engine is None
    render.assert_not_called()
    assert any("Usage: /objective [--dry-run] <objective>" in line for line in printed)


def test_objective_dry_run_does_not_expose_persist_or_cancel_paths():
    _, engine, render, printed = _run_objective("/objective persist oid-1")

    engine.run_pipeline.assert_called_once_with(
        "persist oid-1",
        persist_to_state_meta=False,
    )
    engine.persist.assert_not_called()
    engine.archive.assert_not_called()
    engine.apply.assert_not_called()
    render.assert_called_once()
    assert "persisted to state_meta" not in "\n".join(printed)


def test_objective_dry_run_does_not_start_runtime_workers_or_side_effect_systems():
    _, engine, _, _ = _run_objective("/objective implement API")

    forbidden_methods = [
        "apply",
        "spawn_workers",
        "run_runtime",
        "run_goal_runner",
        "run_kanban",
        "write_gbrain",
        "write_obsidian",
        "write_notebooklm",
    ]
    for name in forbidden_methods:
        method = getattr(engine, name, MagicMock())
        assert method.call_count == 0, name
