"""Cross-platform unit tests for the venv-holder message classifier (#90778)."""

import pytest

from hermes_cli.update_cmd import (
    _format_venv_python_holders_message,
    _hermes_holder_subcommand,
)


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


class TestHolderMessage:
    def _msg(self, cmdline):
        return _format_venv_python_holders_message([(4242, "python.exe", cmdline)])

    def test_dashboard_not_labeled_desktop_backend(self):
        message = self._msg(r"C:\v\Scripts\python.exe -m hermes_cli.main dashboard")
        assert "close the desktop app" not in message.lower()
        assert "hermes dashboard" in message

    def test_preserve_cache_not_labeled_serve(self):
        message = self._msg(r"python -m hermes_cli.main kanban --preserve-cache")
        holder_line = next(l for l in message.splitlines() if "PID 4242" in l)
        # the holder LINE gets no serve/desktop hint (generic footer text
        # legitimately mentions the desktop app)
        assert "←" not in holder_line

    def test_serve_gets_backend_hint(self):
        message = self._msg(r"python -m hermes_cli.main serve --host 127.0.0.1 --port 0")
        assert "Hermes backend" in message

    def test_gateway_hint(self):
        message = self._msg(r"python -m hermes_cli.main gateway run")
        assert "← gateway" in message

    def test_unknown_argv_gets_no_hint(self):
        message = self._msg(r"python -c import this")
        assert "←" not in message
