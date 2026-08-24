"""Regression: Install-Uv must surface installer errors and have fallbacks.

Issue #69216: Windows installs died with only the generic message
``uv installed but not found at ...\\bin\\uv.exe``.  Two root causes:

1. ``Install-Uv`` piped the astral installer's entire output straight into
   ``Out-Null`` (``2>&1 | Out-Null``), so any real failure -- download error,
   corporate proxy block, AV quarantine, permissions -- was swallowed and
   the user only ever saw the generic post-condition failure (first
   identified in #69366).
2. There was exactly one install source, ``astral.sh``.  Corporate proxies
   commonly block astral.sh while the byte-identical installer published at
   GitHub releases downloads fine (diagnosed by @gakugaku on #69216).

The fix installs a three-rung ladder inside ``Install-Uv``:

- Rung 1: astral.sh installer, output captured via ``Tee-Object``.
- Rung 2: GitHub releases installer mirror, same ``UV_INSTALL_DIR``.
- Rung 3: salvage an existing ``uv.exe`` (``Get-Command uv`` or
  ``%USERPROFILE%\\.local\\bin\\uv.exe``) by copying it into
  ``$HermesHome\\bin\\uv.exe`` so the managed-first invariant holds.

Only after all three rungs fail does it error out -- and then it prints the
tail of the captured installer output so the real cause reaches the user.

install.ps1 only runs on Windows, so these tests lock the contract at the
source-text level (same style as test_install_ps1_uv_powershell_host.py).
"""

import re
from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"

_GITHUB_INSTALLER_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1"
)


@pytest.fixture(scope="module")
def source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def _install_uv_body(source: str) -> str:
    """Extract the text of Install-Uv up to the next top-level function."""
    start = source.index("function Install-Uv")
    tail = source[start + 1 :]
    match = re.search(r"^function ", tail, flags=re.MULTILINE)
    end = start + 1 + (match.start() if match else len(tail))
    return source[start:end]


def test_astral_installer_output_not_swallowed_by_out_null(source: str):
    """Regression pin for the suppression bug (#69366 / #69216).

    The astral invocation must not discard the installer's merged output
    stream; a failed download/AV block has to reach the user.
    """
    forbidden = 'irm https://astral.sh/uv/install.ps1 | iex" 2>&1 | Out-Null'
    assert forbidden not in source, (
        "Install-Uv pipes the astral uv installer's output straight to "
        "Out-Null again -- failures become the generic 'uv installed but "
        "not found' message. Capture the output (e.g. Tee-Object) instead."
    )


def test_astral_installer_output_is_captured(source: str):
    body = _install_uv_body(source)
    astral_lines = [
        ln
        for ln in body.splitlines()
        if "irm https://astral.sh/uv/install.ps1 | iex" in ln
    ]
    assert astral_lines, "astral uv installer invocation not found in Install-Uv"
    for ln in astral_lines:
        assert "Tee-Object" in ln, (
            "astral uv installer output must be captured (Tee-Object) so the "
            f"failure path can show it to the user, got: {ln.strip()!r}"
        )


def test_github_releases_fallback_installer_present(source: str):
    """Rung 2: the GitHub releases mirror of the installer must be tried."""
    body = _install_uv_body(source)
    assert _GITHUB_INSTALLER_URL in body, (
        "Install-Uv must fall back to the GitHub releases uv installer "
        f"({_GITHUB_INSTALLER_URL}) when astral.sh is blocked "
        "(corporate proxies, #69216)."
    )
    fallback_lines = [ln for ln in body.splitlines() if _GITHUB_INSTALLER_URL in ln and "irm " in ln]
    for ln in fallback_lines:
        stripped = ln.strip()
        assert stripped.startswith("& $"), (
            "GitHub fallback installer must be invoked via the resolved "
            f"PowerShell host variable (`& $...`), got: {stripped!r}"
        )
        assert "Tee-Object" in ln, (
            f"GitHub fallback installer output must be captured too: {stripped!r}"
        )


def test_existing_uv_salvage_rung_present(source: str):
    """Rung 3: probe PATH and the astral default dir, copy into managed bin."""
    body = _install_uv_body(source)
    assert "Get-Command uv" in body, (
        "Install-Uv must probe for an existing uv on PATH (Get-Command uv) "
        "before failing."
    )
    assert '".local\\bin\\uv.exe"' in body and "$env:USERPROFILE" in body, (
        "Install-Uv must probe the astral default install location "
        "(%USERPROFILE%\\.local\\bin\\uv.exe)."
    )
    assert "Copy-Item" in body and "$managedUv" in body, (
        "A salvaged uv.exe must be copied into the managed location "
        "($HermesHome\\bin\\uv.exe) so managed-first resolution holds."
    )


def test_failure_path_keeps_manual_install_pointer_and_shows_output(source: str):
    body = _install_uv_body(source)
    assert "https://docs.astral.sh/uv/getting-started/installation/" in body, (
        "the manual-install pointer must survive in the failure path"
    )
    assert "$installerOutput" in body and "Select-Object -Last" in body, (
        "the failure path must print the tail of the captured installer "
        "output so the real error reaches the user"
    )
