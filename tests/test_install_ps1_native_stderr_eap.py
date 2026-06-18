"""Regression tests for #48352: Windows PowerShell 5.1 native stderr.

PowerShell 5.1 turns stderr from native commands into ``NativeCommandError``
records when ``$ErrorActionPreference = "Stop"``.  ``scripts/install.ps1`` has a
few git/uv calls where stderr can be normal progress output, so those calls must
run with EAP temporarily relaxed and then inspect ``$LASTEXITCODE``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def _assert_relaxed_call(text: str, command_pattern: str) -> None:
    helper_block_pattern = (
        r"Invoke-NativeWithRelaxedErrorAction\s*\{[^}]*"
        + command_pattern
        + r"[^}]*\}"
    )
    inline_pattern = (
        r"\$ErrorActionPreference\s*=\s*\"Continue\"[\s\S]{0,900}?"
        + command_pattern
    )
    assert re.search(helper_block_pattern, text) or re.search(inline_pattern, text), (
        f"install.ps1 must relax ErrorActionPreference around {command_pattern}"
    )


def test_repository_stage_relieves_eap_for_ssh_and_https_git_clone() -> None:
    text = _install_ps1()
    assert "function Invoke-NativeWithRelaxedErrorAction" in text
    _assert_relaxed_call(
        text,
        r"git -c windows\.appendAtomically=false clone --depth 1 --branch \$Branch \$RepoUrlSsh \$InstallDir",
    )
    _assert_relaxed_call(
        text,
        r"git -c windows\.appendAtomically=false clone --depth 1 --branch \$Branch \$RepoUrlHttps \$InstallDir",
    )


def test_uv_venv_and_dependency_installs_relax_eap() -> None:
    text = _install_ps1()
    _assert_relaxed_call(text, r"& \$UvCmd venv venv --python \$PythonVersion")
    _assert_relaxed_call(text, r"& \$UvCmd sync --extra all --locked")
    _assert_relaxed_call(text, r"& \$UvCmd pip install -e \$tier\.Spec")


def test_native_eap_helper_always_restores_previous_preference() -> None:
    text = _install_ps1()
    m = re.search(
        r"function Invoke-NativeWithRelaxedErrorAction \{(?P<body>[\s\S]*?)^\}",
        text,
        re.MULTILINE,
    )
    assert m is not None, "expected a shared helper for NativeCommandError-safe calls"
    body = m.group("body")
    assert "$prevEAP = $ErrorActionPreference" in body
    assert '$ErrorActionPreference = "Continue"' in body
    assert "finally" in body
    assert "$ErrorActionPreference = $prevEAP" in body
