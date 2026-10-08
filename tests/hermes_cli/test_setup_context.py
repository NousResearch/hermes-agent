"""Shared-file setup preserves profile config, navigation and the Blank Slate contract."""

from argparse import ArgumentParser
from pathlib import Path

import pytest

from hermes_cli import config as cfg, setup, setup_quick


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    from hermes_cli import managed_scope
    managed_scope.invalidate_managed_cache()
    for relative in (".codex/AGENTS.md", ".claude/CLAUDE.md"):
        path = tmp_path / relative
        path.parent.mkdir()
        path.write_text("Trusted shared instructions", encoding="utf-8")
    return tmp_path


def test_context_section_parses_and_saves_additively(home, monkeypatch):
    from hermes_cli.subcommands.setup import build_setup_parser
    parser = ArgumentParser()
    build_setup_parser(parser.add_subparsers(), cmd_setup=lambda args: None)
    args = parser.parse_args(["setup", "context"])
    assert args.section == "context"
    cfg.set_config_value("context.engine", "custom")
    cfg.set_config_value("context.external_files", "~/My Rules.md")
    monkeypatch.setattr(setup, "is_interactive_stdin", lambda: True)
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **kw: True)
    setup.run_setup_wizard(args)
    context = cfg.load_config_readonly()["context"]
    assert context["engine"] == "custom"
    assert context["external_files"] == [
        "~/My Rules.md", "~/.codex/AGENTS.md", "~/.claude/CLAUDE.md",
    ]


def test_decline_and_cancel_leave_config_untouched(home, monkeypatch):
    config = {"context": {"engine": "custom", "external_files": ["~/custom.md"]}}
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **kw: False)
    setup.setup_external_context_files(config)
    assert config["context"]["external_files"] == ["~/custom.md"]
    assert not cfg.get_config_path().exists()
    answers = iter([True, setup._SetupCancelled()])

    def choose(*args, **kwargs):
        result = next(answers)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(setup, "prompt_yes_no", choose)
    with pytest.raises(setup._SetupCancelled):
        setup.setup_external_context_files(config)
    assert config["context"]["external_files"] == ["~/custom.md"]
    assert not cfg.get_config_path().exists()


def test_alias_is_not_duplicated_and_reconfigure_can_disable_it(home, monkeypatch):
    alias = home / "alias.md"
    alias.hardlink_to(home / ".codex/AGENTS.md")
    config = {"context": {"external_files": [str(alias), "~/custom.md"]}}
    defaults = []

    def keep(question, default=False):
        defaults.append(default)
        return default

    monkeypatch.setattr(setup, "prompt_yes_no", keep)
    setup.setup_external_context_files(config)
    assert defaults == [True, False]
    assert config["context"]["external_files"] == [str(alias), "~/custom.md"]
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *a, **kw: False)
    setup.setup_external_context_files(config)
    assert cfg.load_config_readonly()["context"]["external_files"] == ["~/custom.md"]


def test_back_from_later_section_reopens_saved_choices(home, monkeypatch):
    config = {}
    answers = iter([True, True, False])
    shown = []

    def choose(question, default=False):
        start = setup._handle_setup_menu_navigation(setup.MenuNavigationEvent.BEGIN)
        if start.should_replay:
            answer = start.replay_value
        else:
            shown.append(question)
            answer = next(answers)
        setup._handle_setup_menu_navigation(setup.MenuNavigationEvent.RESOLVE, answer)
        return answer

    attempts = 0

    def next_section():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            setup._handle_setup_menu_navigation(setup.MenuNavigationEvent.BEGIN)
            setup._handle_setup_menu_navigation(setup.MenuNavigationEvent.BACK)

    monkeypatch.setattr(setup, "prompt_yes_no", choose)
    with setup._setup_navigation_scope():
        setup._run_setup_steps([
            ("External Context", lambda: setup.setup_external_context_files(config)),
            ("Next", next_section),
        ])
    assert len(shown) == 3
    assert "Claude" in shown[-1]
    assert cfg.load_config_readonly()["context"]["external_files"] == ["~/.codex/AGENTS.md"]


def test_full_and_quick_setup_offer_context_but_blank_slate_does_not(home, monkeypatch):
    offered = []
    for name in ("setup_model_provider", "setup_terminal_backend", "setup_gateway", "setup_tools",
                 "save_config", "_print_setup_summary"):
        monkeypatch.setattr(setup, name, lambda *a, **kw: None)
    monkeypatch.setattr(setup, "setup_external_context_files", lambda config: offered.append(True))
    setup._run_full_setup({}, home, is_existing=True, migration_ran=False)
    assert offered == [True]
    offered.clear()
    monkeypatch.setattr(setup, "prompt_choice", lambda *a, **kw: 0)
    monkeypatch.setattr(setup_quick, "_run_nous_flow", lambda *a, **kw: True)
    monkeypatch.setattr(setup_quick, "_reload_config_into", lambda *a, **kw: None)
    setup_quick._run_first_time_quick_setup({}, home, is_existing=False)
    assert offered == [True]
    offered.clear()
    monkeypatch.setattr(setup_quick, "_set_bundled_skills_opt_out", lambda *a, **kw: None)
    setup_quick._run_blank_slate_setup({}, home, is_existing=False)
    assert offered == []


def test_candidate_aliases_are_offered_once_without_reading_contents(home, monkeypatch):
    claude = home / ".claude/CLAUDE.md"
    claude.unlink()
    claude.symlink_to(home / ".codex/AGENTS.md")
    config = {}
    offered = []

    def choose(question, default=False):
        offered.append((question, default))
        return True

    def no_read(*args, **kwargs):
        raise AssertionError("Setup must not read instruction file contents")

    with monkeypatch.context() as detection:
        detection.setattr(setup, "prompt_yes_no", choose)
        detection.setattr(setup, "save_config", lambda config: None)
        detection.setattr(Path, "read_text", no_read)
        setup.setup_external_context_files(config)
    assert len(offered) == 1
    assert offered[0][1] is False
    assert config["context"]["external_files"] == ["~/.codex/AGENTS.md"]
