"""Regression test for slash_worker PATH construction (#83845).

When the gateway is launched by the Desktop/Dashboard app it can inherit a
minimal PATH (/usr/local/sbin:/usr/local/bin:/usr/bin:/bin:/sbin:/usr/sbin)
that omits the Hermes venv bin dir and ~/.local/bin. The spawned
tui_gateway.slash_worker then cannot resolve Hermes-managed CLIs such as
browser-use/uvx via shutil.which, breaking browser_exec.

`tui_gateway.server._prepend_tool_paths` prepends those two directories to
the worker env PATH while preserving the inherited PATH.
"""

import os
import sys
from pathlib import Path

from tui_gateway import server as tui_server


class TestPrependToolPaths:
    def test_prepends_managed_venv_and_user_bin(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        env = {"PATH": "/usr/bin"}
        result = tui_server._prepend_tool_paths(env)

        parts = result["PATH"].split(os.pathsep)
        # managed bin first (managed-first policy), then venv bin, then
        # user-local bin, then the original PATH preserved
        assert parts[0] == str(tmp_path / "hh" / "bin")
        assert parts[1] == str(Path(sys.executable).parent)
        assert str(Path.home() / ".local" / "bin") in parts
        assert parts[-1] == "/usr/bin"

    def test_preserves_existing_path_when_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        env = {}
        result = tui_server._prepend_tool_paths(env)

        parts = result["PATH"].split(os.pathsep)
        assert parts[0] == str(tmp_path / "hh" / "bin")
        assert str(Path(sys.executable).parent) in parts
        assert str(Path.home() / ".local" / "bin") in parts

    def test_managed_bin_leads_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
        env = {"PATH": "/bin"}
        result = tui_server._prepend_tool_paths(env)
        assert result["PATH"].split(os.pathsep)[0] == str(tmp_path / "hh" / "bin")
