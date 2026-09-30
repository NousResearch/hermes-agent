"""`hermes tools --summary` is a non-interactive report and must not be blocked
by the interactive-TTY guard (#128758)."""

import types

import pytest


class TestToolsSummaryTtyGuard:
    def test_summary_bypasses_tty_guard(self, monkeypatch):
        """--summary runs without a terminal and never calls _require_tty."""

        def _boom(command_name):
            raise AssertionError(
                f"_require_tty({command_name!r}) must not run for --summary"
            )

        monkeypatch.setattr("hermes_cli.main._require_tty", _boom)
        monkeypatch.setattr(
            "hermes_cli.tools_config.load_config",
            lambda: {"platform_toolsets": {"cli": ["web"]}},
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_enabled_platforms", lambda: {"cli"}
        )
        seen = {}
        monkeypatch.setattr(
            "hermes_cli.tools_config._print_tools_summary",
            lambda cfg, plats: seen.update(cfg=cfg, plats=plats),
        )

        from hermes_cli.main_agent_cmds import cmd_tools

        cmd_tools(types.SimpleNamespace(tools_action=None, summary=True))

        assert seen["plats"] == {"cli"}
        assert seen["cfg"] == {"platform_toolsets": {"cli": ["web"]}}

    def test_interactive_path_still_requires_tty(self, monkeypatch):
        """Bare `hermes tools` (the curses picker) keeps the guard."""
        monkeypatch.setattr(
            "hermes_cli.main._require_tty",
            lambda command_name: (_ for _ in ()).throw(SystemExit(1)),
        )

        from hermes_cli.main_agent_cmds import cmd_tools

        with pytest.raises(SystemExit):
            cmd_tools(types.SimpleNamespace(tools_action=None, summary=False))
