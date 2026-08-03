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


def test_uv_venv_failure_is_not_swallowed_after_eap_relax() -> None:
    """Relaxing EAP must not let a genuine `uv venv` failure pass as success.

    Once EAP is relaxed, a real non-zero `uv venv` exit no longer aborts on its
    own, so install.ps1 must capture $LASTEXITCODE right after the call and fail
    fast — otherwise the `venv` stage falsely reports success (Invoke-Stage emits
    ok=true) when no venv was created. Regression guard for the gap caught while
    reviewing #48372 (the explicit check originally proposed in #48463).
    """
    text = _install_ps1()
    # The uv-venv invocation, then an exit-code capture, then a throw — all
    # within a small window after the relaxed call.
    guard = re.search(
        r"& \$UvCmd venv venv --python \$PythonVersion[\s\S]{0,400}?"
        r"\$LASTEXITCODE[\s\S]{0,200}?"
        r"-ne 0[\s\S]{0,200}?throw",
        text,
    )
    assert guard is not None, (
        "install.ps1 must capture uv venv's exit code and throw on failure after "
        "relaxing ErrorActionPreference, so a genuine venv-creation failure isn't "
        "reported as a successful stage"
    )


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
