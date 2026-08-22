"""Tests for follow-up fixes to the LSP integration (PR after #24168).

Covers:

1. ``typescript-language-server`` install recipe pulls in ``typescript``
   alongside the server, so the npm install command targets both.
2. ``hermes lsp status`` surfaces a ``Backend warnings`` section when
   bash-language-server is installed but ``shellcheck`` is missing.
3. ``_check_lint`` returns ``skipped`` (not ``error``) when the linter
   command exists on PATH but couldn't actually run — e.g. ``npx tsc``
   without the typescript SDK installed.  This is what unblocks the
   LSP semantic tier on TypeScript files when the user doesn't also
   have a project-level ``tsc``.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from agent.lsp.install import INSTALL_RECIPES


# ---------------------------------------------------------------------------
# Fix 1: typescript install recipe carries the typescript SDK
# ---------------------------------------------------------------------------




def test_install_npm_passes_extras_to_npm_command(tmp_path, monkeypatch):
    """Verify the npm subprocess is invoked with both pkg AND extras."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Pretend npm succeeded but binary doesn't exist — install code
        # will return None, which is fine for this test.
        return MagicMock(returncode=0, stderr="")

    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod, "find_node_executable", lambda c: "/usr/bin/npm" if c == "npm" else None)

    extras = install_mod.INSTALL_RECIPES["typescript-language-server"]["extra_pkgs"]
    sdk_target = next(
        target for target in extras if target.partition("@")[0] == "typescript"
    )
    # The server's current major still loads lib/tsserver.js, which TypeScript
    # 7 removed. Keep the recipe compatible without freezing the whole list.
    assert sdk_target.partition("@")[2].split(".", 1)[0] == "6"
    install_mod._install_npm(
        "typescript-language-server",
        "typescript-language-server",
        extra_pkgs=extras,
    )

    cmd = captured["cmd"]
    assert "typescript-language-server" in cmd
    # Both must come AFTER the npm flags, in install-target position
    install_idx = cmd.index("install")
    assert cmd.index("typescript-language-server") > install_idx
    assert cmd.index(sdk_target) > install_idx


def test_install_npm_works_without_extras(tmp_path, monkeypatch):
    """Backwards compat: pyright-style recipes (no extras) still install."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stderr="")

    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod, "find_node_executable", lambda c: "/usr/bin/npm" if c == "npm" else None)

    install_mod._install_npm("pyright", "pyright-langserver")

    cmd = captured["cmd"]
    assert "pyright" in cmd
    # Should not blow up when extra_pkgs is omitted/None
    install_targets = [c for c in cmd if not c.startswith("-") and c not in {
        "install", "--prefix", str(install_mod.hermes_lsp_bin_dir().parent),
        "/usr/bin/npm",
    }]
    assert install_targets == ["pyright"]




def test_existing_binary_prefers_windows_wrapper_over_posix_shim(tmp_path, monkeypatch):
    """A stale npm POSIX shim must not shadow its native Windows wrapper."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    staged = install_mod.hermes_lsp_bin_dir()
    posix_shim = staged / "pyright-langserver"
    posix_shim.write_text("#!/bin/sh\nexit 0\n")
    posix_shim.chmod(0o755)
    wrapper = staged / "pyright-langserver.cmd"
    wrapper.write_text("@echo off\n")
    wrapper.chmod(0o755)

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("pyright-langserver") == str(wrapper)


def test_existing_binary_prefers_canonical_npm_wrapper(tmp_path, monkeypatch):
    """The npm .cmd must run in node_modules/.bin so its relative paths work."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    staged = install_mod.hermes_lsp_bin_dir()
    (staged / "pyright-langserver").write_text("#!/bin/sh\nexit 0\n")
    (staged / "pyright-langserver.cmd").write_text("@echo off\n")
    npm_bin = staged.parent / "node_modules" / ".bin"
    npm_bin.mkdir(parents=True)
    canonical = npm_bin / "pyright-langserver.cmd"
    canonical.write_text("@echo off\n")
    canonical.chmod(0o755)

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("pyright-langserver") == str(canonical)


def test_existing_binary_prefers_hermes_wrapper_over_canonical_npm(
    tmp_path, monkeypatch
):
    """A generated wrapper must survive process restart and stay preferred."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    staged = install_mod.hermes_lsp_bin_dir()
    generated = staged / "pyright-langserver.hermes.cmd"
    generated.write_text("@echo off\n")
    npm_bin = staged.parent / "node_modules" / ".bin"
    npm_bin.mkdir(parents=True)
    canonical = npm_bin / "pyright-langserver.cmd"
    canonical.write_text("@echo off\n")

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("pyright-langserver") == str(generated)


def test_auto_install_upgrades_preexisting_canonical_npm_wrapper(
    tmp_path, monkeypatch
):
    """Existing managed packages gain a safe wrapper without npm reinstall."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    staging = install_mod.hermes_lsp_bin_dir().parent
    npm_bin = staging / "node_modules" / ".bin"
    npm_bin.mkdir(parents=True)
    (npm_bin / "pyright-langserver.cmd").write_text("@echo off\n")
    package_dir = staging / "node_modules" / "pyright"
    package_dir.mkdir()
    (package_dir / "package.json").write_text(
        json.dumps({"bin": {"pyright-langserver": "langserver.index.js"}}),
        encoding="utf-8",
    )
    (package_dir / "langserver.index.js").write_text("// fixture\n")

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(
        install_mod,
        "find_node_executable",
        lambda name: r"C:\Program Files\nodejs\node.exe" if name == "node" else None,
    )
    monkeypatch.setattr(
        install_mod,
        "_install_npm",
        lambda *_args, **_kwargs: pytest.fail("existing package was reinstalled"),
    )

    resolved = install_mod._do_install("pyright")

    expected = staging / "bin" / "pyright-langserver.hermes.cmd"
    assert resolved == str(expected)
    assert expected.exists()


@pytest.mark.windows_only
def test_preexisting_canonical_tree_repairs_and_spawns_in_hostile_home(
    tmp_path, monkeypatch
):
    """The normal upgrade path repairs and runs a canonical-only install."""
    from hermes_cli._subprocess_compat import windows_batch_proxy_command

    home = tmp_path / "home&b%UNEXPANDED%!UNEXPANDED!(x)^caret"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("UNEXPANDED", "wrong")

    from agent.lsp import install as install_mod

    staging = install_mod.hermes_lsp_bin_dir().parent
    npm_bin = staging / "node_modules" / ".bin"
    npm_bin.mkdir(parents=True)
    (npm_bin / "pyright-langserver.cmd").write_text("@echo off\n")
    package_dir = staging / "node_modules" / "pyright"
    package_dir.mkdir()
    (package_dir / "package.json").write_text(
        json.dumps({"bin": {"pyright-langserver": "capture.py"}}),
        encoding="utf-8",
    )
    (package_dir / "capture.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(
        install_mod,
        "find_node_executable",
        lambda name: sys.executable if name == "node" else None,
    )
    monkeypatch.setattr(
        install_mod,
        "_install_npm",
        lambda *_args, **_kwargs: pytest.fail("existing package was reinstalled"),
    )

    resolved = install_mod._do_install("pyright")

    expected = staging / "bin" / "pyright-langserver.hermes.cmd"
    assert resolved == str(expected)
    result = subprocess.run(
        windows_batch_proxy_command([str(expected), "upgrade-ok"]),
        shell=False,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["upgrade-ok"]


def test_auto_install_ignores_orphaned_generated_and_staged_npm_wrappers(
    tmp_path, monkeypatch
):
    """Orphaned generated or copied wrappers must not suppress repair."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    orphan = install_mod.hermes_lsp_bin_dir() / "pyright-langserver.hermes.cmd"
    orphan.write_text("@echo off\n")
    stale_copy = install_mod.hermes_lsp_bin_dir() / "pyright-langserver.cmd"
    stale_copy.write_text("@echo off\n")
    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod, "find_node_executable", lambda _name: None)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        install_mod,
        "_install_npm",
        lambda *_args, **_kwargs: "reinstalled-wrapper.cmd",
    )

    assert install_mod._do_install("pyright") == "reinstalled-wrapper.cmd"


def test_existing_binary_rejects_posix_only_shim_on_windows(tmp_path, monkeypatch):
    """An extensionless shebang script is not a Win32 executable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    shim = install_mod.hermes_lsp_bin_dir() / "pyright-langserver"
    shim.write_text("#!/bin/sh\nexit 0\n")
    shim.chmod(0o755)

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("pyright-langserver") is None
    assert install_mod.detect_status("pyright") == "missing"


def test_stale_posix_only_install_triggers_npm_repair(tmp_path, monkeypatch):
    """A broken pre-fix install must be rejected and repaired automatically."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    stale_shim = install_mod.hermes_lsp_bin_dir() / "pyright-langserver"
    stale_shim.write_text("#!/bin/sh\nexit 0\n")
    stale_shim.chmod(0o755)
    repaired = (
        install_mod.hermes_lsp_bin_dir().parent
        / "node_modules"
        / ".bin"
        / "pyright-langserver.cmd"
    )
    repair_calls = []

    def fake_install(pkg, bin_name, extra_pkgs=None):
        repair_calls.append((pkg, bin_name, extra_pkgs))
        return str(repaired)

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(install_mod, "_install_npm", fake_install)

    assert install_mod._do_install("pyright") == str(repaired)
    assert repair_calls == [("pyright", "pyright-langserver", [])]


def test_existing_binary_accepts_native_extensionless_pe_on_windows(
    tmp_path, monkeypatch
):
    """A native PE executable remains valid even without a file suffix."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    binary = install_mod.hermes_lsp_bin_dir() / "custom-language-server"
    dos_header = bytearray(64)
    dos_header[:2] = b"MZ"
    dos_header[0x3C:0x40] = (len(dos_header)).to_bytes(4, "little")
    binary.write_bytes(bytes(dos_header) + b"PE\0\0")
    binary.chmod(0o755)

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("custom-language-server") == str(binary)


def test_existing_binary_rejects_mz_prefixed_non_pe_on_windows(tmp_path, monkeypatch):
    """A DOS marker alone must not make an arbitrary blob launchable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    binary = install_mod.hermes_lsp_bin_dir() / "custom-language-server"
    binary.write_bytes(b"MZ\x90\x00not actually a PE executable")
    binary.chmod(0o755)

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("custom-language-server") is None


def test_non_windows_candidates_preserve_extensionless_launcher(monkeypatch):
    """Linux and macOS keep the existing extensionless candidate behavior."""
    from agent.lsp import install as install_mod

    base = install_mod.hermes_lsp_bin_dir() / "pyright-langserver"
    monkeypatch.setattr(install_mod, "_is_windows", lambda: False)

    assert install_mod._native_binary_candidates(base) == [base]


def test_windows_batch_command_uses_explicit_autorun_disabled_shell():
    from hermes_cli._subprocess_compat import windows_batch_command

    command = [r"C:\Hermes\lsp\node_modules\.bin\pyright-langserver.cmd", "--stdio"]
    env = {}

    command_line = windows_batch_command(command, env, prefix="HERMES_LSP_COMMAND")

    assert command_line.lower().startswith('"')
    assert " /d " in command_line.lower()
    assert "/v:off" in command_line.lower()
    assert '"%HERMES_LSP_COMMAND_0%"' in command_line
    assert '"%HERMES_LSP_COMMAND_1%"' in command_line
    assert "^%" not in command_line
    assert command[0] not in command_line
    assert env["HERMES_LSP_COMMAND_0"] == command[0]
    assert env["HERMES_LSP_COMMAND_1"] == command[1]


@pytest.mark.parametrize(
    "argument", ['unsafe\"quote', "unsafe\rline", "unsafe\nline", "unsafe\0nul"]
)
def test_windows_npm_wrapper_rejects_untransportable_arguments(argument):
    from hermes_cli._subprocess_compat import windows_batch_proxy_command

    with pytest.raises(ValueError, match="cannot contain quotes or control"):
        windows_batch_proxy_command([r"C:\Hermes\server.cmd", argument])


@pytest.mark.asyncio
async def test_spawn_routes_windows_batch_launcher_through_native_proxy(
    tmp_path, monkeypatch
):
    from agent.lsp import client as client_mod

    captured = {}

    class FakeProcess:
        stdout = None
        stderr = None

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    async def unexpected_shell(*_args, **_kwargs):
        pytest.fail("Windows batch launcher used an implicit shell")

    monkeypatch.setattr(client_mod.sys, "platform", "win32")
    monkeypatch.setattr(client_mod.asyncio, "create_subprocess_shell", unexpected_shell)
    monkeypatch.setattr(client_mod.asyncio, "create_subprocess_exec", fake_exec)
    wrapper = tmp_path / "a&b" / "server.cmd"
    client = client_mod.LSPClient(
        server_id="test",
        workspace_root=str(tmp_path),
        command=[str(wrapper), "--stdio"],
    )

    await client._spawn()
    assert client._stderr_task is not None
    assert client._reader_task is not None
    await asyncio.gather(client._stderr_task, client._reader_task)

    args = captured["args"]
    assert args[0] == sys.executable
    assert args[1].endswith("_subprocess_compat.py")
    assert args[2] == "--windows-batch-proxy"
    assert list(args[-2:]) == [str(wrapper), "--stdio"]
    assert not any(
        key.startswith("HERMES_LSP_COMMAND_")
        for key in captured["kwargs"]["env"]
    )


async def _exercise_bounded_lsp_cleanup(monkeypatch, expected_events):
    """Drive cleanup with a process that exits only after forced cleanup."""
    from agent.lsp import client as client_mod

    events = []
    finished = asyncio.Event()

    class FakeProcess:
        returncode = None
        _transport = None

        async def wait(self):
            events.append("wait")
            await finished.wait()
            return 0

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")
            finished.set()

    proc = FakeProcess()
    client = client_mod.LSPClient(
        server_id="cleanup-order-test",
        workspace_root=".",
        command=["server"],
    )
    setattr(client, "_proc", proc)
    monkeypatch.setattr(client_mod, "PROCESS_EXIT_GRACE_SECONDS", 0.01)

    if sys.platform == "win32":

        def fake_kill_process_tree(actual_proc):
            assert actual_proc is proc
            events.append("tree-kill")
            finished.set()

        monkeypatch.setattr(client_mod, "kill_process_tree", fake_kill_process_tree)

    await asyncio.wait_for(client._cleanup_process(), timeout=1.0)

    assert events == expected_events


@pytest.mark.windows_only
@pytest.mark.asyncio
async def test_lsp_cleanup_gives_clean_exit_one_bounded_chance_then_tree_kills(
    monkeypatch,
):
    await _exercise_bounded_lsp_cleanup(
        monkeypatch,
        ["wait", "tree-kill", "wait"],
    )


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_lsp_cleanup_gives_clean_exit_one_bounded_chance_then_posix_escalates_linux(
    monkeypatch,
):
    await _exercise_bounded_lsp_cleanup(
        monkeypatch,
        ["wait", "terminate", "wait", "kill", "wait"],
    )


@pytest.mark.macos_only
@pytest.mark.asyncio
async def test_lsp_cleanup_gives_clean_exit_one_bounded_chance_then_posix_escalates_macos(
    monkeypatch,
):
    await _exercise_bounded_lsp_cleanup(
        monkeypatch,
        ["wait", "terminate", "wait", "kill", "wait"],
    )


@pytest.mark.windows_only
@pytest.mark.parametrize(
    "argument",
    ["hello&%UNEXPANDED%!UNEXPANDED!()^caret|pipe<in>out", ""],
)
def test_windows_npm_wrapper_handles_shell_metacharacters(tmp_path, argument):
    """The explicit /D shell preserves metacharacters without an outer shell."""
    from hermes_cli._subprocess_compat import run_windows_batch

    wrapper = (
        tmp_path
        / "a&b%UNEXPANDED%!UNEXPANDED!(x)^caret"
        / "server.cmd"
    )
    wrapper.parent.mkdir()
    wrapper.write_text(
        "@echo off" + chr(13) + chr(10) + "echo READY [%1]" + chr(13) + chr(10)
    )
    env = dict(os.environ)
    env["UNEXPANDED"] = "wrong"
    result = run_windows_batch(
        [str(wrapper), argument],
        env=env,
        prefix="HERMES_LSP_COMMAND",
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f'READY ["{argument}"]']


def test_install_npm_generates_location_independent_windows_wrapper(
    tmp_path, monkeypatch
):
    """Managed npm bins use a Hermes wrapper instead of npm's fragile shim."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    npm_bin = install_mod.hermes_lsp_bin_dir().parent / "node_modules" / ".bin"

    def fake_run(cmd, **kwargs):
        npm_bin.mkdir(parents=True, exist_ok=True)
        (npm_bin / "pyright-langserver").write_text("#!/bin/sh\nexit 0\n")
        (npm_bin / "pyright-langserver.cmd").write_text("@echo off\n")
        package_dir = npm_bin.parent / "pyright"
        package_dir.mkdir()
        (package_dir / "package.json").write_text(
            json.dumps({"bin": {"pyright-langserver": "langserver.index.js"}})
        )
        (package_dir / "langserver.index.js").write_text("// fixture\n")
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(
        install_mod,
        "find_node_executable",
        lambda name: {
            "npm": "C:\\Program Files\\nodejs\\npm.cmd",
            "node": "C:\\Program Files\\nodejs\\node.exe",
        }.get(name),
    )
    monkeypatch.setattr(install_mod, "run_windows_batch", fake_run)

    resolved = install_mod._install_npm("pyright", "pyright-langserver")

    wrapper = install_mod.hermes_lsp_bin_dir() / "pyright-langserver.hermes.cmd"
    assert resolved == str(wrapper)
    content = wrapper.read_text(encoding="ascii")
    assert "setlocal DisableDelayedExpansion" in content
    assert 'set "dp0=%~dp0"' in content
    assert "langserver.index.js" in content
    assert (npm_bin / "pyright-langserver.cmd").exists()


def test_npm_bin_script_rejects_non_object_package_metadata(tmp_path):
    """Malformed-but-valid JSON metadata falls back instead of raising."""
    from agent.lsp import install as install_mod

    package_dir = tmp_path / "node_modules" / "fixture-package"
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text("[]", encoding="utf-8")

    assert (
        install_mod._npm_bin_script(
            tmp_path, "fixture-package", "fixture-language-server"
        )
        is None
    )

@pytest.mark.windows_only
def test_install_npm_handles_metacharacters_in_hermes_home(tmp_path, monkeypatch):
    """npm.cmd must receive the complete --prefix path through cmd.exe."""
    home = tmp_path / "home&b%UNEXPANDED%!UNEXPANDED!(x)^caret"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("UNEXPANDED", "wrong")

    from agent.lsp import install as install_mod

    fake_npm = home / "node" / "npm.cmd"
    fake_npm.parent.mkdir(parents=True)
    fake_npm.write_text(
        os.linesep.join(
            [
                "@echo off",
                'if "%~1"=="--version" (',
                "    echo 1.0.0",
                "    exit /b 0",
                ")",
                'set "prefix=%~3"',
                r'if not exist "%prefix%\node_modules\.bin" mkdir "%prefix%\node_modules\.bin"',
                r'> "%prefix%\node_modules\.bin\pyright-langserver.cmd" echo @echo off',
                r'if not exist "%prefix%\node_modules\fake-package" mkdir "%prefix%\node_modules\fake-package"',
                r'> "%prefix%\node_modules\fake-package\package.json" echo {"bin":{"pyright-langserver":"server.js"}}',
                r'> "%prefix%\node_modules\fake-package\server.js" echo // fixture',
                "exit /b 0",
            ]
        )
        + os.linesep
    )
    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(
        install_mod,
        "find_node_executable",
        lambda name: str(fake_npm)
        if name == "npm"
        else r"C:\Program Files\nodejs\node.exe",
    )

    resolved = install_mod._install_npm("fake-package", "pyright-langserver")

    expected = home / "lsp" / "bin" / "pyright-langserver.hermes.cmd"
    assert resolved == str(expected)
    assert expected.exists()


@pytest.mark.windows_only
def test_generated_npm_wrapper_keeps_managed_node_and_home_relative(
    tmp_path, monkeypatch
):
    """Managed Node and package paths must not bake HERMES_HOME into the shim."""
    home = tmp_path / "home-ünicode&b%UNEXPANDED%!UNEXPANDED!(x)^caret"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.lsp import install as install_mod

    staging = install_mod.hermes_lsp_bin_dir().parent
    package_dir = staging / "node_modules" / "fixture-package"
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text(
        json.dumps({"bin": {"fixture-server": "capture.js"}}),
        encoding="utf-8",
    )
    (package_dir / "capture.js").write_text("// fixture\n", encoding="utf-8")
    managed_node = home / "node" / "node.exe"
    managed_node.parent.mkdir(parents=True)
    managed_node.write_bytes(b"MZ")
    monkeypatch.setattr(
        install_mod,
        "find_node_executable",
        lambda name: str(managed_node) if name == "node" else None,
    )

    wrapper = install_mod._write_windows_node_wrapper(
        staging, "fixture-package", "fixture-server"
    )
    assert wrapper is not None
    content = wrapper.read_text(encoding="ascii")
    assert str(home) not in content
    assert "%dp0%" in content
    assert "node.exe" in content
    assert "capture.js" in content


@pytest.mark.windows_only
@pytest.mark.parametrize(
    "argument",
    [
        "hello&%UNEXPANDED%!UNEXPANDED!()^caret|pipe<in>out",
        "trailing\\",
        "double-trailing\\\\",
    ],
)
def test_generated_npm_wrapper_survives_real_metacharacter_home(
    tmp_path, monkeypatch, argument
):
    """Exercise the generated wrapper through the exact native LSP proxy."""
    home = tmp_path / "home&b%UNEXPANDED%!UNEXPANDED!(x)^caret"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("UNEXPANDED", "wrong")

    from agent.lsp import install as install_mod
    from hermes_cli._subprocess_compat import (
        windows_batch_proxy_command,
        windows_hide_flags,
    )

    staging = install_mod.hermes_lsp_bin_dir().parent
    package_dir = staging / "node_modules" / "fixture-package"
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text(
        json.dumps({"bin": {"fixture-server": "capture.py"}}),
        encoding="utf-8",
    )
    (package_dir / "capture.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        install_mod,
        "find_node_executable",
        lambda name: sys.executable if name == "node" else None,
    )

    wrapper = install_mod._write_windows_node_wrapper(
        staging, "fixture-package", "fixture-server"
    )
    assert wrapper is not None

    env = dict(os.environ)
    proxy_command = windows_batch_proxy_command([str(wrapper), argument, ""])
    result = subprocess.run(
        proxy_command,
        shell=False,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        creationflags=windows_hide_flags(),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [argument, ""]


@pytest.mark.windows_only
def test_run_windows_batch_timeout_kills_descendants(tmp_path):
    """A timed-out batch operation must not leave its real child running."""
    import psutil

    from hermes_cli._subprocess_compat import run_windows_batch

    root = tmp_path / "a&b%UNEXPANDED%!UNEXPANDED!(x)^caret"
    root.mkdir()
    pid_file = root / "child.pid"
    child = root / "child.py"
    child.write_text(
        "import os, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    wrapper = root / "hang.cmd"
    wrapper.write_text(
        "@echo off\r\n"
        "setlocal DisableDelayedExpansion\r\n"
        'set "dp0=%~dp0"\r\n'
        f'"{sys.executable}" "%dp0%child.py" "%dp0%child.pid"\r\n',
        encoding="utf-8",
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_windows_batch(
            [str(wrapper)],
            timeout=1.0,
            stdin=subprocess.DEVNULL,
        )

    assert pid_file.exists()
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3.0
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(child_pid)


@pytest.mark.windows_only
@pytest.mark.asyncio
async def test_lsp_cleanup_kills_batch_proxy_descendants(tmp_path):
    """Forced LSP cleanup owns Python proxy, cmd.exe, and server descendants."""
    import psutil

    from agent.lsp.client import LSPClient

    root = tmp_path / "a&b%UNEXPANDED%!UNEXPANDED!(x)^caret"
    root.mkdir()
    pid_file = root / "child.pid"
    child = root / "child.py"
    child.write_text(
        "import os, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    wrapper = root / "server.cmd"
    wrapper.write_text(
        "@echo off\r\n"
        "setlocal DisableDelayedExpansion\r\n"
        'set "dp0=%~dp0"\r\n'
        f'"{sys.executable}" "%dp0%child.py" "%dp0%child.pid"\r\n',
        encoding="utf-8",
    )
    client = LSPClient(
        server_id="cleanup-test",
        workspace_root=str(tmp_path),
        command=[str(wrapper)],
    )

    await client._spawn()
    assert client._proc is not None
    proxy_pid = client._proc.pid
    deadline = asyncio.get_running_loop().time() + 4.0
    while not pid_file.exists() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert pid_file.exists()
    child_pid = int(pid_file.read_text())
    assert psutil.pid_exists(proxy_pid)
    assert psutil.pid_exists(child_pid)
    descendants = psutil.Process(proxy_pid).children(recursive=True)
    cmd_children = [
        proc
        for proc in descendants
        if (proc.name() or "").lower() in {"cmd", "cmd.exe"}
    ]
    assert len(cmd_children) == 1
    assert "/d" in [part.lower() for part in cmd_children[0].cmdline()]

    await client._cleanup_process()

    deadline = asyncio.get_running_loop().time() + 3.0
    while (
        (psutil.pid_exists(proxy_pid) or psutil.pid_exists(child_pid))
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.05)
    assert not psutil.pid_exists(proxy_pid)
    assert not psutil.pid_exists(child_pid)
    # asyncio's proactor subprocess transport keeps pipe handles open even
    # after the process exits; close it explicitly so the loop teardown does
    # not warn about an unclosed transport from the test's own child.
    transport = getattr(client._proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


def test_install_npm_uses_managed_resolver_off_windows(tmp_path, monkeypatch):
    """Linux/macOS preserve current managed-Node npm resolution."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod, "_is_windows", lambda: False)
    monkeypatch.setattr(
        install_mod,
        "find_node_executable",
        lambda name: "/managed/node/bin/npm" if name == "npm" else None,
    )
    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)

    assert install_mod._install_npm("pyright", "pyright-langserver") is None
    assert captured["cmd"][0] == "/managed/node/bin/npm"

@pytest.mark.windows_only
def test_install_pip_finds_windows_scripts_launcher(tmp_path, monkeypatch):
    """pip console scripts can land in Scripts/ on native Windows.

    ``windows_only``: the ``Scripts/`` layout and the ``.exe`` launcher are
    what pip actually produces on Windows. Faking ``_is_windows()`` on Linux
    made the test assert against a directory tree the test itself created, on
    a host where pip would never lay it out that way.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    def fake_run(cmd, **kwargs):
        scripts_dir = install_mod.hermes_lsp_bin_dir().parent / "python-packages" / "Scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        launcher = scripts_dir / "fake-language-server.exe"
        launcher.write_text("launcher\n")
        launcher.chmod(0o755)
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)

    resolved = install_mod._install_pip("fake-lsp", "fake-language-server")

    assert resolved is not None
    assert resolved.endswith("fake-language-server.exe")
    assert (install_mod.hermes_lsp_bin_dir() / "fake-language-server.exe").exists()


# ---------------------------------------------------------------------------
# Fix 2: ``hermes lsp status`` surfaces shellcheck-missing for bash
# ---------------------------------------------------------------------------






def test_backend_warnings_fires_when_bash_installed_but_shellcheck_missing(tmp_path, monkeypatch):
    """The exact scenario from the bug report."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import cli as lsp_cli

    def which(name):
        if name in {"bash-language-server", "bash-language-server.cmd"}:
            return "C:\\fake\\bash-language-server.cmd"
        return None  # shellcheck missing

    with patch("shutil.which", side_effect=which):
        notes = lsp_cli._backend_warnings()
    assert len(notes) == 1
    assert "shellcheck" in notes[0].lower()
    assert "bash-language-server" in notes[0].lower()


def test_status_output_includes_backend_warnings_section(tmp_path, monkeypatch):
    """End-to-end: status command output includes the warning section."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Pretend bash-language-server is installed but shellcheck is missing
    def which(name):
        if name in {"bash-language-server", "bash-language-server.cmd"}:
            return "C:\\fake\\bash-language-server.cmd"
        return None

    from agent.lsp import cli as lsp_cli

    buf = io.StringIO()
    with patch("shutil.which", side_effect=which), redirect_stdout(buf):
        lsp_cli._cmd_status(emit_json=False)

    output = buf.getvalue()
    assert "Backend warnings" in output
    assert "shellcheck" in output


# ---------------------------------------------------------------------------
# Fix 3: tier-1 lint treats unusable linters as ``skipped``, not ``error``
# ---------------------------------------------------------------------------










def test_check_lint_returns_error_for_real_ts_type_errors(tmp_path):
    """Sanity: real TypeScript errors still go through the error path."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    ts_file = tmp_path / "bad.ts"
    ts_file.write_text("const x: string = 42;\n")

    env = LocalEnvironment()
    fops = ShellFileOperations(env)

    real_tsc_error = (
        "bad.ts:1:7 - error TS2322: Type 'number' is not assignable to type 'string'.\n"
        "1 const x: string = 42;\n"
        "        ~\n"
        "Found 1 error.\n"
    )

    def fake_exec(cmd, **kwargs):
        result = MagicMock()
        result.exit_code = 1
        result.stdout = real_tsc_error
        return result

    with patch.object(fops, "_exec", side_effect=fake_exec), \
         patch.object(fops, "_has_command", return_value=True):
        lint = fops._check_lint(str(ts_file))

    assert lint.skipped is False
    assert lint.success is False
    assert "TS2322" in lint.output


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
