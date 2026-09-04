"""The bootstrap pin fragments must match pm/lock.json.

The installers (scripts/install.sh / scripts/install.ps1) are fetched
standalone (curl | sh, irm | iex) and bootstrap uv — and, on Windows, git —
BEFORE any checkout exists, so they cannot read pm/lock.json at run time.
scripts/gen-bootstrap-pins.py derives an inline fragment from the lockfile
and splices it between markers in each script. These tests enforce the
derive-don't-store contract: the stored bytes can never drift from
pm/lock.json (the same authority the pm package manager uses), and the
installers actually consume the pinned values instead of an unpinned
"latest" channel.

setup-hermes.sh is deliberately NOT a fragment carrier: it runs after a
checkout exists, so it reads the uv pin straight from pm/lock.json at run
time (the former pm.sh bootstrap).
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "gen-bootstrap-pins.py"
SETUP_HERMES_SH = REPO_ROOT / "setup-hermes.sh"
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
LOCK = REPO_ROOT / "pm" / "lock.json"

_SH_FILES = (INSTALL_SH,)
_POSIX_TARGETS = ("linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64")
_WINDOWS_TARGETS = ("win32-x64", "win32-arm64")


def _packages() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8-sig"))["packages"]


def test_fragments_match_the_pin_table():
    """--check regenerates from pm/lock.json and fails on any drift."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"bootstrap fragments drifted from pm/lock.json:\n"
        f"{result.stdout}{result.stderr}"
    )


def test_installers_carry_the_pinned_uv_version():
    """Both fragment-carrying installers hold the exact uv version pm/lock.json names."""
    version = _packages()["uv"]["version"]
    for path in _SH_FILES:
        assert f'UV_PIN_VERSION="{version}"' in path.read_text(
            encoding="utf-8"
        ), path.name
    ps1 = INSTALL_PS1.read_text(encoding="utf-8-sig")
    assert f'$script:UvPinVersion = "{version}"' in ps1


def test_dev_setup_reads_the_pin_from_the_lockfile():
    """setup-hermes.sh runs after a checkout exists, so it must NOT carry a
    duplicated fragment: it reads the uv pin from pm/lock.json at run time."""
    source = SETUP_HERMES_SH.read_text(encoding="utf-8")
    assert "UV_PIN_VERSION" not in source
    assert 'pm/lock.json' in source
    assert "sha256" in source


def test_windows_installer_carries_the_pinned_uv_and_git():
    """install.ps1's bootstrap uv and git are pm/lock.json's, not copies."""
    ps1 = INSTALL_PS1.read_text(encoding="utf-8-sig")
    uv = _packages()["uv"]
    git = _packages()["git"]
    assert f'$script:UvPinVersion = "{uv["version"]}"' in ps1
    assert f'$script:GitPinVersion = "{git["version"]}"' in ps1
    for target in _WINDOWS_TARGETS:
        for name, entry in (("uv", uv), ("git", git)):
            artifact = entry["artifacts"][target]
            assert artifact["url"] in ps1, f"{name} {target} url"
            assert artifact["sha256"] in ps1, f"{name} {target} sha256"


def test_windows_git_bootstrap_has_no_hand_written_version():
    """A second git version authority must not come back.

    The drifted copy hard-coded a PortableGit tag + version string by hand
    next to the download. Anything matching that shape means someone
    re-introduced a version this test cannot keep honest — the version and
    URL must come only from the generated fragment.
    """
    ps1 = INSTALL_PS1.read_text(encoding="utf-8-sig")
    assert "$gitTag" not in ps1
    assert "$gitVer" not in ps1
    assert "releases/download/$gitTag" not in ps1


def test_digests_are_verified_before_any_archive_is_unpacked():
    """An unverified download here is arbitrary code execution, so the
    sha256 check must sit between the download and the extract/exec step."""
    for path in _SH_FILES:
        source = path.read_text(encoding="utf-8")
        assert "digest mismatch" in source, path.name
        # The digest computation appears before the tar extraction.
        digest_at = source.index("sha256sum")
        extract_at = source.index("tar -xzf")
        assert digest_at < extract_at, f"{path.name}: verify before extract"

    ps1 = INSTALL_PS1.read_text(encoding="utf-8-sig")
    # uv: digest check precedes Expand-Archive.
    uv_hash = ps1.index("Get-FileHash -Path $zipPath")
    uv_extract = ps1.index("Expand-Archive -Path $zipPath")
    assert uv_hash < uv_extract, "uv: verify before extract"
    assert "uv digest mismatch" in ps1
    # git: digest check precedes the tar extraction.
    git_hash = ps1.index("Get-FileHash -Path $tarPath")
    git_extract = ps1.index("tar.exe -xf $tarPath")
    assert git_hash < git_extract, "git: verify before extract"
    assert "git digest mismatch" in ps1


def test_installers_stage_into_the_pm_store_slot():
    """The installers must stage into the pm store slot
    (<store>/<tool>-<version>-<target>/), not the astral ~/.local/bin
    layout — so pm adopts the exact same bytes."""
    for path in _SH_FILES:
        source = path.read_text(encoding="utf-8")
        assert f"uv-$UV_PIN_VERSION-$_target" in source, path.name
    ps1 = INSTALL_PS1.read_text(encoding="utf-8-sig")
    assert f'uv-$($script:UvPinVersion)-$target' in ps1
    assert f'git-$($script:GitPinVersion)-$target' in ps1


def test_installers_do_not_fetch_unpinned_uv():
    """The astral latest-channel installers must never come back.

    astral.sh/uv/install.sh and install.ps1 resolve "latest" at run time,
    which defeats pins-as-repo-data: a hermes install could silently get a
    uv nobody reviewed. The only astral.sh mentions allowed are the manual
    -install docs URL (docs.astral.sh).
    """
    for path in _SH_FILES + (INSTALL_PS1,):
        source = path.read_text(encoding="utf-8-sig")
        assert "astral.sh/uv/install.sh" not in source, path.name
        assert "astral.sh/uv/install.ps1" not in source, path.name
