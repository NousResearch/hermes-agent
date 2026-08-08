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

    def test_env_node_shebang_spawns_via_node_not_bash(self, tmp_path):
        """Regression (review of #68494): a shebang naming a non-shell
        interpreter (Node, via the common '#!/usr/bin/env node' form) must
        NOT be routed through bash -c -- bash would try to execute the
        JS source as shell script under the wrong interpreter and fail.
        It must spawn via the actual named interpreter, resolved from PATH."""
        script = tmp_path / "some-js-launcher"
        script.write_bytes(b"#!/usr/bin/env node\nconsole.log('hi');\n")
        with patch(
            "shutil.which",
            side_effect=lambda name: (
                "/usr/bin/node" if name == "node" else "/usr/bin/bash"
            ),
        ):
            result = LSPClient._win_wrap_cmd([str(script), "--stdio"])
        assert result[0] == "/usr/bin/node"
        assert result[1] == str(script)
        assert result[2] == "--stdio"
        assert "bash" not in result

    def test_direct_python_shebang_spawns_via_python_not_bash(self, tmp_path):
        """Same regression, direct-path shebang form (no /usr/bin/env
        indirection): '#!/usr/bin/python3'."""
        script = tmp_path / "some-py-launcher"
        script.write_bytes(b"#!/usr/bin/python3\nprint('hi')\n")
        with patch(
            "shutil.which",
            side_effect=lambda name: (
                "/usr/bin/python3" if name == "python3" else "/usr/bin/bash"
            ),
        ):
            result = LSPClient._win_wrap_cmd([str(script), "--stdio"])
        assert result[0] == "/usr/bin/python3"
        assert result[1] == str(script)
        assert "bash" not in result

    def test_non_shell_interpreter_not_on_path_falls_through_with_warning(
        self, tmp_path, caplog
    ):
        """If the named non-shell interpreter isn't resolvable, fall
        through unchanged (matching the existing no-bash-found behavior)
        rather than silently misrouting through bash."""
        script = tmp_path / "some-js-launcher"
        script.write_bytes(b"#!/usr/bin/env node\nconsole.log('hi');\n")
        with patch("shutil.which", return_value=None):
            with caplog.at_level("WARNING", logger="agent.lsp.client"):
                result = LSPClient._win_wrap_cmd([str(script), "--stdio"])
        assert result == [str(script), "--stdio"]
        assert any("node" in r.getMessage().lower() for r in caplog.records)

    def test_shell_shebang_variants_still_use_bash(self, tmp_path):
        """Sanity: this fix must not regress the original supported cases --
        sh, bash, zsh, dash shebangs (direct or via /usr/bin/env) must still
        route through bash -c as before."""
        for i, shebang in enumerate(
            (b"#!/bin/sh\n", b"#!/bin/bash\n", b"#!/usr/bin/env bash\n", b"#!/usr/bin/env sh\n")
        ):
            script = tmp_path / f"shell-script-{i}"
            script.write_bytes(shebang + b"exec node real.js \"$@\"\n")
            with patch("shutil.which", return_value="/usr/bin/bash"):
                result = LSPClient._win_wrap_cmd([str(script), "--stdio"])
            assert result[0] == "/usr/bin/bash", (shebang, result)
            assert result[1] == "-c", (shebang, result)


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
