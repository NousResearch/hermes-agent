from pathlib import Path
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj._app = MagicMock()
    return cli_obj


def _make_native_skill(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "networking" / "hermes-network"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("name: hermes-network\n")
    return skill_dir


def _native_launcher(skill_dir: Path):
    return {
        "/hermes-network": {
            "name": "hermes-network",
            "description": "Chat-only network skill",
            "skill_dir": str(skill_dir),
            "native_launcher": {
                "executable": ".venv/bin/hermes-network",
                "setup_command": "python3 scripts/install_runtime.py",
                "setup_sentinel": ".venv/bin/hermes-network",
                "terminal": "tmux",
                "terminal_name": "hermes-network",
                "default_args": ["app"],
                "default_subcommand": "app",
                "interactive_subcommands": ["app", "host", "join", "chat"],
                "global_options_with_values": ["--data-dir"],
                "global_flag_options": ["--help", "-h"],
                "hold_on_failure": True,
            },
        }
    }


class TestCLIHermesNetworkCommand:
    def test_default_launch_bootstraps_and_opens_tmux(self, tmp_path):
        cli_obj = _make_cli()
        skill_dir = _make_native_skill(tmp_path)

        with (
            patch("cli._skill_commands", _native_launcher(skill_dir)),
            patch("cli.shutil.which", return_value="/usr/bin/tmux"),
            patch.dict("cli.os.environ", {"TMUX": "1"}, clear=False),
            patch("subprocess.run") as mock_run,
        ):
            result = cli_obj.process_command("/hermes-network")

        assert result is True
        cli_obj._app.run_system_command.assert_not_called()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[:3] == ["/usr/bin/tmux", "new-window", "-n"]
        assert args[3] == "hermes-network"
        command = args[4]
        assert "python3 scripts/install_runtime.py" in command
        assert ".venv/bin/hermes-network app" in command
        assert "Bootstrapping native skill runtime" in command

    def test_noninteractive_subcommand_runs_via_bash_inline(self, tmp_path):
        cli_obj = _make_cli()
        skill_dir = _make_native_skill(tmp_path)

        with (
            patch("cli._skill_commands", _native_launcher(skill_dir)),
            patch("cli.shutil.which", return_value="/usr/bin/tmux"),
            patch("subprocess.run") as mock_run,
        ):
            cli_obj.process_command("/hermes-network demo-local-mesh")

        cli_obj._app.run_system_command.assert_not_called()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[:2] == ["bash", "-lc"]
        command = args[2]
        assert "python3 scripts/install_runtime.py" in command
        assert ".venv/bin/hermes-network demo-local-mesh" in command

    def test_missing_skill_dir_shows_error(self, tmp_path):
        cli_obj = _make_cli()
        missing_dir = tmp_path / "skills" / "networking" / "hermes-network"

        with patch("cli._skill_commands", _native_launcher(missing_dir)):
            cli_obj.process_command("/hermes-network")

        cli_obj.console.print.assert_called()
        printed = "\n".join(str(call.args[0]) for call in cli_obj.console.print.call_args_list)
        assert "not installed correctly" in printed.lower()
