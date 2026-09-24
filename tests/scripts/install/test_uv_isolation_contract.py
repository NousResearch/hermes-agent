"""Cross-language conformance for the uv isolation contract (user layout).

The same set of uv write axes is encoded three times: in Python
(``hermes_cli.managed_uv.managed_uv_env``), in ``scripts/install.sh``'s
``uv_isolated_state_env()``, and in ``setup-hermes.sh``'s
``uv_isolated_state_env()``.  install.sh is delivered standalone via
``curl | bash`` and cannot share a sourced file, so the copies are kept in
lockstep by *running* each against the same hostile inherited ``UV_*``
environment and asserting they agree with the Python source of truth.

This is the level that stops the three copies drifting apart.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
SETUP_SH = REPO_ROOT / "setup-hermes.sh"

_SIG = "uv_isolated_state_env() {\n"
# Axes the POSIX installers pin (they leave UV_TOOL_BIN_DIR to the browser-use
# call site, so it is not part of the install-time contract).
_AXIS_ORDER = (
    "UV_CACHE_DIR", "UV_TOOL_DIR", "UV_PYTHON_INSTALL_DIR",
    "UV_PYTHON_INSTALL_BIN", "UV_PYTHON_INSTALL_REGISTRY", "UV_PYTHON_BIN_DIR",
)
# Full runtime axes, including the shim dir managed_uv_env always pins.
_RUNTIME_AXES = (
    "UV_CACHE_DIR", "UV_TOOL_DIR", "UV_TOOL_BIN_DIR", "UV_PYTHON_INSTALL_DIR",
    "UV_PYTHON_INSTALL_BIN", "UV_PYTHON_INSTALL_REGISTRY",
)

def _uv_state_fn(script: Path) -> str:
    text = script.read_text(encoding="utf-8")
    _, marker, rest = text.partition(_SIG)
    assert marker, f"{script} is missing uv_isolated_state_env()"
    body, end, _ = rest.partition("\n}\n")
    assert end, f"{script} has an unterminated uv_isolated_state_env()"
    return marker + body + end


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _shell_axes(script: Path, home: Path) -> str:
    home.mkdir(parents=True, exist_ok=True)
    harness = home / "contract-harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        f"HERMES_HOME='{_bash_path(home)}'\n"
        "export UV_CACHE_DIR=/evil/cache UV_TOOL_DIR=/evil/tools\n"
        "export UV_PYTHON_INSTALL_DIR=/evil/python UV_PYTHON_INSTALL_BIN=1\n"
        "export UV_PYTHON_INSTALL_REGISTRY=1 UV_PYTHON_BIN_DIR=/evil/bin\n"
        + _uv_state_fn(script)
        + "\nuv_isolated_state_env user\n"
        + 'echo "$UV_CACHE_DIR|$UV_TOOL_DIR|$UV_PYTHON_INSTALL_DIR|'
        '$UV_PYTHON_INSTALL_BIN|$UV_PYTHON_INSTALL_REGISTRY|${UV_PYTHON_BIN_DIR:-unset}"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", _bash_path(harness)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _python_axes(home: Path) -> str:
    from hermes_cli.managed_uv import managed_uv_env

    with patch("hermes_cli.managed_uv.get_hermes_home", return_value=home):
        env = managed_uv_env()
    return "|".join(env.get(key, "unset") for key in _AXIS_ORDER)


@pytest.mark.parametrize("script", [INSTALL_SH, SETUP_SH], ids=["install.sh", "setup-hermes.sh"])
@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="POSIX installer contract requires a POSIX host with bash",
)
def test_posix_installer_contract_matches_python(script: Path, tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    assert _shell_axes(script, home) == _python_axes(home)


def _python_runtime_axes(home: Path) -> str:
    from hermes_cli.managed_uv import managed_uv_env

    with patch("hermes_cli.managed_uv.get_hermes_home", return_value=home):
        env = managed_uv_env(base_env={})
    return "|".join(env.get(key, "unset") for key in _RUNTIME_AXES)


def test_stdlib_pocket_matches_runtime_contract(tmp_path, monkeypatch) -> None:
    """``hermes_cli._early_recovery`` cannot import ``managed_uv`` (it runs when
    the checkout is broken), so it hand-rolls the same pins.  Assert the two
    agree on every runtime axis — this is what keeps that pocket from drifting
    out of the contract."""
    from hermes_cli import _early_recovery as er

    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    pocket = er._uv_isolation_env(base_env={})
    assert "|".join(pocket.get(k, "unset") for k in _RUNTIME_AXES) == _python_runtime_axes(home)
