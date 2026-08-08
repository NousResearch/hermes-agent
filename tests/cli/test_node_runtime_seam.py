"""Seam tests for the node_runtime extraction (main.py god-file slice R4-S1).

The three node-runtime leaf helpers (``_is_termux_env``,
``_is_windows_npm_path``, ``_resolve_node_runtime_npm``) moved verbatim from
``hermes_cli/main.py`` into ``hermes_cli/node_runtime.py``; main.py re-exports
them (``# noqa: F401``) so callers and test monkeypatches on
``hermes_cli.main.<name>`` keep resolving unchanged. These tests pin the
seam: object identity through the re-export, patch reachability through
``hermes_cli.main``, the two sanctioned ``_m()`` delegations, and the WSL
npm-rejection re-scan loop (#30271).
"""

import os
from unittest.mock import patch

import pytest

from hermes_cli import main as hm
from hermes_cli import node_runtime


# --- identity: re-export must not shadow the moved objects -------------------

def test_reexport_object_identity_termux_env():
    assert hm._is_termux_env is node_runtime._is_termux_env


def test_reexport_object_identity_windows_npm_path():
    assert hm._is_windows_npm_path is node_runtime._is_windows_npm_path


def test_reexport_object_identity_resolve_node_runtime_npm():
    assert hm._resolve_node_runtime_npm is node_runtime._resolve_node_runtime_npm


# --- patch reachability through hermes_cli.main (existing patch sites) -------

def test_patch_main_name_reaches_in_main_caller(monkeypatch):
    """Monkeypatching ``hermes_cli.main._is_termux_env`` must be observed by
    in-main callers that resolve the name at call time (mirror of
    test_cmd_update.py:165 style) — the re-export keeps the name live in
    main's namespace for `_default_venv_install_target` (stays in main.py)."""
    from hermes_cli import main as hm

    calls = []

    def fake_termux(env=None):
        calls.append(env)
        return True

    monkeypatch.setattr(hm, "_is_termux_env", fake_termux)
    monkeypatch.setattr(
        "hermes_cli.managed_uv.ensure_uv", lambda: "/data/data/com.termux/files/usr/bin/uv"
    )

    install_prefix, env = hm._default_venv_install_target()

    assert install_prefix == ["/data/data/com.termux/files/usr/bin/uv", "pip"]
    assert calls, "in-main caller must reach the patched _is_termux_env"
    assert env is not None
    # termux branch popped PYTHONPATH/PYTHONHOME — proves the patch was honored.
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env


# --- sanctioned _m() delegations (helpers stay in main.py) -------------------

def test_is_termux_env_delegates_to_main_helper(monkeypatch):
    """_is_termux_env body routes through ``_m()._is_termux_startup_environment``
    (helper stays in main.py at R3)."""
    monkeypatch.setattr(hm, "_is_termux_startup_environment", lambda env: True)
    assert node_runtime._is_termux_env({"PREFIX": "/data/data/com.termux/files/usr"}) is True


def test_resolve_node_runtime_npm_delegates_windows_check(monkeypatch):
    """The ``_is_windows()`` gate inside _resolve_node_runtime_npm routes
    through ``_m()`` (stays in main.py; post-#79661 resolves via win_quarantine
    re-export). Native Windows must short-circuit to the platform npm."""
    monkeypatch.setattr(hm, "_is_windows", lambda: True)
    with patch("hermes_constants.find_node_executable", return_value="C:\\nodejs\\npm.cmd") as find:
        assert node_runtime._resolve_node_runtime_npm() == "C:\\nodejs\\npm.cmd"
        find.assert_called_once_with("npm")


# --- aggressive: node resolution + version/WSL matrix ------------------------

def test_is_windows_npm_path_matrix():
    assert node_runtime._is_windows_npm_path("/mnt/c/Program Files/nodejs/npm") is True
    assert node_runtime._is_windows_npm_path("/usr/bin/npm") is False
    assert node_runtime._is_windows_npm_path("C:\\nodejs\\npm.cmd") is True


def test_resolve_node_runtime_npm_linux_native_passthrough(monkeypatch):
    """POSIX host, Linux-native npm on PATH → returned as-is."""
    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    with patch("hermes_constants.find_node_executable", return_value="/usr/bin/npm"):
        assert node_runtime._resolve_node_runtime_npm() == "/usr/bin/npm"


def test_resolve_node_runtime_npm_no_npm_returns_none(monkeypatch):
    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    with patch("hermes_constants.find_node_executable", return_value=None):
        assert node_runtime._resolve_node_runtime_npm() is None


def test_resolve_node_runtime_npm_wsl_rejects_windows_shim(monkeypatch):
    """#30271 regression: POSIX host whose PATH resolves a Windows npm first
    (via WSL interop) must re-scan and return a Linux-native npm."""
    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    with patch("hermes_constants.find_node_executable", return_value="/mnt/c/Program Files/nodejs/npm"):
        # find_node_executable already refused; the re-scan loop owns the path.
        with patch("hermes_cli.node_runtime.shutil.which") as which:
            def fake_which(cmd, path=None):
                assert cmd == "npm"
                if path and path.endswith("/usr/bin"):
                    return "/usr/bin/npm"
                return None

            which.side_effect = fake_which
            monkeypatch.setenv(
                "PATH",
                os.pathsep.join(["/mnt/c/Program Files/nodejs", "/usr/bin", "/bin"]),
            )
            assert node_runtime._resolve_node_runtime_npm() == "/usr/bin/npm"
