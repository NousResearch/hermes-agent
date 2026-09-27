"""The ps1 twins pin bootstrap uv state under the machine root, not the user's.

Their ``UV_CACHE_DIR`` / ``UV_PYTHON_INSTALL_DIR`` must name slots under
``get_default_hermes_root()`` — the cache in its OWN ``cache/uv-bootstrap``
slot, off ``pm.packages.uv_cache_dir()`` (the payload-seeded directory;
bootstrap bytes there first would make that seed skip subtrees yet still mark
itself done), or ``uv python install`` writes managed CPython into a
state tree pm never looks at and re-downloads it. Each file's pin block is
extracted verbatim and executed with its own ``Get-HermesRoot`` — the scripts
cannot be sourced on POSIX pwsh — so the statements run are the shipped bytes;
``test_bootstrap_store_root_matches_pm.py`` pins the two bodies byte-identical.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

# name -> (path, BEGIN markers, naming the twin each block says it mirrors).
_PS1_BOOTSTRAPS = {
    "setup-hermes.ps1": (
        ROOT / "setup-hermes.ps1",
        "# --- BEGIN store-root resolver (mirrored in scripts/install.ps1) ---",
        "# --- BEGIN uv state pins (mirrored in scripts/install.ps1) ---",
    ),
    "install.ps1": (
        ROOT / "scripts" / "install.ps1",
        "# --- BEGIN store-root resolver (mirrored in setup-hermes.ps1) ---",
        "# --- BEGIN uv state pins (mirrored in setup-hermes.ps1) ---",
    ),
}
_RESOLVER_END = "# --- END store-root resolver ---"
_PINS_END = "# --- END uv state pins ---"

_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
# ``any``: the OS lanes import only platforms-marked files, so without it this
# file is never selected by the macOS/Windows lanes even though its extracted
# resolver/pin blocks are designed to run under POSIX pwsh. The skipif still
# decides — PowerShell is the real prerequisite, not the host OS.
pytestmark = [
    pytest.mark.skipif(
        _POWERSHELL is None, reason="running the PowerShell bootstrap needs pwsh or powershell"
    ),
    pytest.mark.platforms("any"),
]

_HOME_SHAPES = ("default", "profile")


def _block(text: str, begin: str, end: str) -> str:
    match = re.search(re.escape(begin) + r"\r?\n(.*?)" + re.escape(end), text, re.DOTALL)
    assert match, f"the {begin!r} block is missing from the ps1 bootstrap"
    return match.group(1)


@pytest.mark.parametrize("home_kind", _HOME_SHAPES)
@pytest.mark.parametrize("bootstrap", sorted(_PS1_BOOTSTRAPS))
def test_ps1_bootstraps_pin_uv_state_under_the_machine_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home_kind: str, bootstrap: str
) -> None:
    home = tmp_path / "native-home" / ".hermes"
    hermes_home = home if home_kind == "default" else home / "profiles" / "coder"
    monkeypatch.setenv("HOME", str(tmp_path / "native-home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "native-home"))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_PYTHON_INSTALL_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_constants import get_default_hermes_root

    expected_cache = get_default_hermes_root() / "cache" / "uv-bootstrap"
    expected_python = expected_cache.parent / "uv-python"

    path, resolver_begin, pins_begin = _PS1_BOOTSTRAPS[bootstrap]
    text = path.read_text(encoding="utf-8-sig")
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        + _block(text, resolver_begin, _RESOLVER_END)
        + "\n"
        + _block(text, pins_begin, _PINS_END)
        + "\nSet-UvStatePins\n"
        'Write-Output "$env:UV_CACHE_DIR|$env:UV_PYTHON_INSTALL_DIR"\n'
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-Command", script],
        env={**os.environ},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cache, python_dir = result.stdout.strip().splitlines()[-1].split("|")
    assert Path(cache) == expected_cache, (
        f"{bootstrap}/{home_kind}: pinned UV_CACHE_DIR={cache}, expected {expected_cache}"
    )
    assert Path(python_dir) == expected_python, (
        f"{bootstrap}/{home_kind}: the python dir must sit beside the pinned cache"
    )
