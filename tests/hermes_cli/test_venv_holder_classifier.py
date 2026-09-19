"""Cross-platform unit tests for the Hermes command-line classifier (#90778)."""

import pytest

from hermes_cli.update_cmd_windows import _hermes_holder_subcommand


class TestHolderSubcommand:
    @pytest.mark.parametrize(
        ("cmdline", "expected"),
        [
            (r"C:\x\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1", "serve"),
            (r"C:\x\venv\Scripts\python.exe -m hermes_cli.main dashboard", "dashboard"),
            (r"python.exe -m hermes_cli.main gateway run", "gateway"),
            # profile selector skipped; its VALUE must not become the subcommand
            (r"python -m hermes_cli.main --profile serve gateway run", "gateway"),
            (r"python -m hermes_cli.main -p work serve", "serve"),
            # 90778: flags containing subcommand words are not subcommands
            (r"python -m hermes_cli.main kanban --preserve-cache", "kanban"),
            # 91869 review: EVERY top-level value flag must be skipped —
            # a flag VALUE equal to a subcommand must not become the label
            (r"python -m hermes_cli.main --reasoning high serve", "serve"),
            (r"python -m hermes_cli.main -m dashboard serve", "serve"),
            (r"python -m hermes_cli.main -t browser,files gateway run", "gateway"),
            (r"python -m hermes_cli.main --model=dashboard serve", "serve"),
            # -c consumes ONE value token; later bare tokens are (harmless,
            # unhinted) subcommand candidates — pin that shape honestly
            (r"python -m hermes_cli.main -c mysession serve", "serve"),
            (r"C:\bin\hermes.exe dashboard", "dashboard"),
            (r"/usr/local/bin/hermes serve", "serve"),
            # no hermes entry at all
            (r"python -c import time; time.sleep(3)", None),
            # entry but no subcommand
            (r"python -m hermes_cli.main", None),
        ],
    )
    def test_parses_subcommand(self, cmdline, expected):
        assert _hermes_holder_subcommand(cmdline) == expected
