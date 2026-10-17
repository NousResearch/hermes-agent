"""The ps1 bootstraps pin bootstrap uv state under the machine root, not the user's.

Their ``UV_CACHE_DIR`` / ``UV_PYTHON_INSTALL_DIR`` must name slots under
``get_default_hermes_root()`` — the cache in its OWN ``cache/uv-bootstrap``
slot, off ``pm.packages.uv_cache_dir()`` (the payload-seeded directory;
bootstrap bytes there first would make that seed skip subtrees yet still mark
itself done), or ``uv python install`` writes managed CPython into a
state tree pm never looks at and re-downloads it.

Each bootstrap loads the resolver and pins the way it ships: ``install.ps1`` is
piped (``irm | iex``) and carries GENERATED copies, while ``setup-hermes.ps1``
runs from a checkout and dot-sources the canonical files. The probe reproduces
that arrangement rather than string-matching markers, so the statements that
run are the bytes the shipped entry path runs.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = ROOT / "scripts" / "bootstrap"
CANON_RESOLVER = BOOTSTRAP / "store_root.ps1"
CANON_PINS = BOOTSTRAP / "uv_state_pins.ps1"

_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
# ``any``: the OS lanes import only platforms-marked files, so without it this
# file is never selected by the macOS/Windows lanes even though its extracted
# blocks are designed to run under POSIX pwsh. The skipif still decides —
# PowerShell is the real prerequisite, not the host OS.
pytestmark = [
    pytest.mark.skipif(
        _POWERSHELL is None, reason="running the PowerShell bootstrap needs pwsh or powershell"
    ),
    pytest.mark.platforms("any"),
]

# name -> the shipped file that loads the resolver and pins.
_PS1_BOOTSTRAPS = {
    "setup-hermes.ps1": ROOT / "setup-hermes.ps1",
    "install.ps1": ROOT / "scripts" / "install.ps1",
}

_HOME_SHAPES = ("default", "profile")


def _loads_canonical(text: str, name: str) -> bool:
    """Whether *text* dot-sources the canonical resolver instead of embedding it."""
    return "bootstrap\\store_root.ps1" in text


def _probe_script(bootstrap: str, path: Path) -> str:
    """The statements the shipped file runs to reach the pins, in order.

    A checkout bootstrap dot-sources the canonical files; a piped installer
    carries generated copies, so its own bytes are executed between the
    GENERATED markers. Either way ``Set-UvStatePins`` is what must have run.
    """
    if bootstrap == "setup-hermes.ps1":
        assert _loads_canonical(path.read_text(encoding="utf-8-sig"), bootstrap), (
            "setup-hermes.ps1 must dot-source the canonical resolver"
        )
        return (
            f'. "{CANON_RESOLVER.as_posix()}"\n'
            f'. "{CANON_PINS.as_posix()}"\n'
        )
    return (
        _generated_block(path, "# --- BEGIN GENERATED: store-root resolver", "# --- END GENERATED: store-root resolver ---")
        + "\n"
        + _generated_block(path, "# --- BEGIN GENERATED: uv state pins", "# --- END GENERATED: uv state pins ---")
        + "\n"
    )


def _generated_block(path: Path, begin_prefix: str, end: str) -> str:
    text = path.read_text(encoding="utf-8-sig")
    start = text.index(begin_prefix)
    start = text.index("\n", start) + 1
    stop = text.index(end, start)
    return text[start:stop]


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

    path = _PS1_BOOTSTRAPS[bootstrap]
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        + _probe_script(bootstrap, path)
        + "Set-UvStatePins\n"
        'Write-Output "$env:UV_CACHE_DIR|$env:UV_PYTHON_INSTALL_DIR"\n'
    )
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-Command", script],
        env={**os.environ},
        capture_output=True, text=True, timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"{bootstrap}/{home_kind}: {result.stdout}{result.stderr}"
    cache, python_dir = result.stdout.strip().splitlines()[-1].split("|")
    assert Path(cache) == expected_cache, (
        f"{bootstrap}/{home_kind}: pinned UV_CACHE_DIR={cache}, expected {expected_cache}"
    )
    assert Path(python_dir) == expected_python, (
        f"{bootstrap}/{home_kind}: the python dir must sit beside the pinned cache"
    )
