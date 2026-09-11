"""Regression: install.ps1 must syntax-check the dashboard backend source.

Issue #59004 reported a fresh Windows desktop install crashing on launch
because ``hermes_cli/web_server.py`` inside the installed checkout still
contained merge-conflict markers. Import-only dependency probes (fastapi /
uvicorn) do not catch that: the packages can be present while the backend
source itself is unparsable.

This test is source-level because Linux CI cannot execute the PowerShell
installer. It locks the contract that install.ps1 runs ``py_compile`` against
``hermes_cli/web_server.py`` and fails the stage when that syntax probe fails.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def test_install_ps1_compiles_web_server_source_after_web_deps_probe() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")

    probe = re.search(
        r'import fastapi, uvicorn[\s\S]{0,1200}?-m py_compile "\$InstallDir\\hermes_cli\\web_server\.py"',
        text,
    )
    assert probe is not None, (
        "install.ps1 must syntax-check hermes_cli/web_server.py after the "
        "dashboard dependency probe so a fresh desktop install fails early on "
        "merge-conflict markers or other SyntaxErrors."
    )


def test_install_ps1_fails_stage_when_web_server_syntax_probe_fails() -> None:
    text = INSTALL_PS1.read_text(encoding="utf-8")

    assert "if ($LASTEXITCODE -eq 0) { $webServerSyntaxOk = $true }" in text
    assert "if (-not $webServerSyntaxOk) {" in text
    assert (
        'throw "dashboard backend source failed syntax check: hermes_cli/web_server.py"'
        in text
    ), (
        "install.ps1 must fail the install stage when hermes_cli/web_server.py "
        "does not compile, instead of writing a broken desktop/backend install."
    )
