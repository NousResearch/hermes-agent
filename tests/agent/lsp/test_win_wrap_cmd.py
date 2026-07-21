"""Unit tests for LSPClient._win_wrap_cmd (issue #49470).

npm/npx-installed LSP servers often place a Unix shell script (no
.exe/.cmd/.bat extension, '#!/bin/sh' shebang) as the launcher -- e.g.
pyright-langserver under ~/.hermes/lsp/bin/. On Windows, CreateProcess
cannot execute these directly and fails with WinError 193. _win_wrap_cmd()
detects the shebang and wraps the command with 'bash -c' when bash is
available on PATH.
"""
from unittest.mock import patch

from agent.lsp.client import LSPClient


def _write_shebang_script(tmp_path, name="pyright-langserver"):
    script = tmp_path / name
    script.write_bytes(b"#!/bin/sh\nexec node \"$(dirname \"$0\")/pyright.js\" \"$@\"\n")
    return script


class TestWinWrapCmdShebangDetection:
    def test_shebang_script_wrapped_with_bash_when_available(self, tmp_path):
        script = _write_shebang_script(tmp_path)
        with patch("shutil.which", return_value="/usr/bin/bash"):
            result = LSPClient._win_wrap_cmd([str(script), "--stdio"])
        assert result[0] == "/usr/bin/bash"
        assert result[1] == "-c"
        assert str(script) in result[2]
        assert "--stdio" in result[2]

    def test_shebang_script_falls_through_unchanged_when_bash_absent(self, tmp_path, caplog):
        script = _write_shebang_script(tmp_path)
        with patch("shutil.which", return_value=None):
            with caplog.at_level("WARNING", logger="agent.lsp.client"):
                result = LSPClient._win_wrap_cmd([str(script), "--stdio"])
        assert result == [str(script), "--stdio"]
        assert any("bash" in r.getMessage().lower() for r in caplog.records)

    def test_non_shebang_extensionless_file_left_unchanged(self, tmp_path):
        """A genuine native binary without a recognized extension must not
        be wrapped -- only files that actually start with '#!' are."""
        native = tmp_path / "some-native-tool"
        native.write_bytes(b"\x7fELF\x02\x01\x01\x00")  # ELF magic, not a shebang
        with patch("shutil.which", return_value="/usr/bin/bash"):
            result = LSPClient._win_wrap_cmd([str(native), "--stdio"])
        assert result == [str(native), "--stdio"]

    def test_nonexistent_file_left_unchanged(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        result = LSPClient._win_wrap_cmd([str(missing), "--stdio"])
        assert result == [str(missing), "--stdio"]


class TestWinWrapCmdExistingBehaviorPreserved:
    def test_cmd_extension_still_wrapped_with_cmd_exe(self):
        result = LSPClient._win_wrap_cmd(["C:\\tools\\server.cmd", "--stdio"])
        assert result == ["cmd.exe", "/c", "C:\\tools\\server.cmd", "--stdio"]

    def test_bat_extension_still_wrapped_with_cmd_exe(self):
        result = LSPClient._win_wrap_cmd(["C:\\tools\\server.bat"])
        assert result == ["cmd.exe", "/c", "C:\\tools\\server.bat"]

    def test_exe_extension_passed_through_unchanged(self):
        result = LSPClient._win_wrap_cmd(["C:\\tools\\server.exe", "--stdio"])
        assert result == ["C:\\tools\\server.exe", "--stdio"]

    def test_ps1_extension_passed_through_unchanged(self):
        result = LSPClient._win_wrap_cmd(["C:\\tools\\server.ps1"])
        assert result == ["C:\\tools\\server.ps1"]
