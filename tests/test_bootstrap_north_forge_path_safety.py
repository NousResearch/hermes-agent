"""Regression: scripts/bootstrap-north-forge.ps1 must refuse a venv/data target
that IS, contains, or sits inside the checkout (Codex audit F-04 - data loss).

The bug: the original guard rejected a path *strictly inside* the repo root
(a ``StartsWith(repoRoot + '\\')`` test), but a path **equal** to the repo root
did not start with ``repoRoot + '\\'``, so it passed the guard. A ``-Force``
rebuild then runs ``Remove-Item -LiteralPath $VenvDir -Recurse -Force`` - with
``$VenvDir`` == the checkout, that recursively deletes the repository.

The fix canonicalizes both paths (``[IO.Path]::GetFullPath`` + trimmed
separators + ordinal-ignore-case compare) and rejects three directions -
venv-equals-repo, venv-inside-repo, repo-inside-venv - for both ``-VenvDir`` and
``-DataDir``.

Two layers here:
  * ``test_guard_source_*`` - source-level, runs on every platform (Linux CI
    cannot execute PowerShell), the transport-independent invariant.
  * ``test_reject_*`` - behavioural, Windows-only: actually run the script and
    confirm the dangerous input is refused and the checkout is left intact.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_PS1 = REPO_ROOT / "scripts" / "bootstrap-north-forge.ps1"


# --------------------------------------------------------------------------
# source-level - portable
# --------------------------------------------------------------------------

def _source() -> str:
    return BOOTSTRAP_PS1.read_text(encoding="utf-8")


def test_guard_source_canonicalizes_and_checks_all_three_directions() -> None:
    src = _source()

    # canonicalization of both operands
    assert "GetFullPath" in src, "guard must canonicalize paths before comparing"
    assert "OrdinalIgnoreCase" in src, "path compare must be case-insensitive ordinal"

    # an explicit equality test (the exact case the old guard missed)
    assert "[string]::Equals(" in src, (
        "guard must test venv/data EQUAL to the repo root, not only 'inside' it"
    )

    # both containment directions
    overlap = src[src.index("function Test-PathOverlap") : src.index("if (-not $RepoRoot)")]
    assert overlap.count(".StartsWith(") >= 2, (
        "guard must reject BOTH 'venv inside repo' and 'repo inside venv'"
    )

    # applied to venv AND data
    assert "-VenvDir" in src and "-DataDir" in src
    guard = src[src.index("foreach ($pair") : src.index("foreach ($pair") + 800]
    assert "$VenvDir" in guard and "$DataDir" in guard, (
        "the collision guard must cover both the venv and the data dir"
    )


def test_guard_source_no_longer_has_the_naive_startswith_only_check() -> None:
    src = _source()
    assert "$full.TrimEnd('\\').ToLower().StartsWith(" not in src, (
        "the naive inside-only guard (missed the venv==repo case) must be gone"
    )


# --------------------------------------------------------------------------
# behavioural - Windows only (bootstrap-north-forge.ps1 is a Windows launcher path)
# --------------------------------------------------------------------------

_WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt", reason="bootstrap-north-forge.ps1 runs under Windows PowerShell only"
)


@pytest.fixture
def fake_checkout(tmp_path: Path) -> Path:
    """A throwaway 'checkout': has pyproject.toml (the guard's first check) and a
    canary file whose survival proves the script did not delete the tree."""
    repo = tmp_path / "north-forge-agent"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    (repo / "CANARY.txt").write_text("this file must survive a rejected bootstrap", encoding="utf-8")
    return repo


def _run_bootstrap(repo: Path, venv: Path | str, data: Path | str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(BOOTSTRAP_PS1),
            "-RepoRoot", str(repo),
            "-VenvDir", str(venv),
            "-DataDir", str(data),
            "-Force",
        ],
        capture_output=True, text=True, timeout=120,
    )


@_WINDOWS_ONLY
def test_reject_venv_equal_to_repo_root(fake_checkout: Path, tmp_path: Path) -> None:
    # THE data-loss case: venv target == the checkout itself.
    result = _run_bootstrap(fake_checkout, fake_checkout, tmp_path / "data")

    assert result.returncode != 0, (
        f"bootstrap accepted venv==repo (rc=0). stdout+stderr:\n{result.stdout}\n{result.stderr}"
    )
    assert "checkout" in (result.stdout + result.stderr).lower()
    assert (fake_checkout / "CANARY.txt").exists(), "checkout was deleted despite a rejected run"
    assert (fake_checkout / "pyproject.toml").exists()


@_WINDOWS_ONLY
@pytest.mark.parametrize("mangle", [
    lambda p: p,                       # exact
    lambda p: str(p) + "\\",           # trailing separator
    lambda p: str(p).upper(),          # case difference
    lambda p: str(p) + "\\.\\",        # non-normalized
])
def test_reject_venv_equal_variants(fake_checkout: Path, tmp_path: Path, mangle) -> None:
    result = _run_bootstrap(fake_checkout, mangle(fake_checkout), tmp_path / "data")
    assert result.returncode != 0, f"accepted a repo-root-equivalent venv path: {mangle(fake_checkout)!r}"
    assert (fake_checkout / "CANARY.txt").exists()


@_WINDOWS_ONLY
def test_reject_venv_inside_repo_root(fake_checkout: Path, tmp_path: Path) -> None:
    result = _run_bootstrap(fake_checkout, fake_checkout / "nested" / "venv", tmp_path / "data")
    assert result.returncode != 0
    assert (fake_checkout / "CANARY.txt").exists()


@_WINDOWS_ONLY
def test_reject_repo_root_inside_venv(fake_checkout: Path) -> None:
    # venv target is the parent that CONTAINS the checkout.
    result = _run_bootstrap(fake_checkout, fake_checkout.parent, fake_checkout.parent / "data")
    assert result.returncode != 0
    assert (fake_checkout / "CANARY.txt").exists()


@_WINDOWS_ONLY
def test_reject_data_equal_to_repo_root(fake_checkout: Path, tmp_path: Path) -> None:
    # same equality rule applies to -DataDir, not just -VenvDir.
    result = _run_bootstrap(fake_checkout, tmp_path / "venv", fake_checkout)
    assert result.returncode != 0
    assert (fake_checkout / "CANARY.txt").exists()


@_WINDOWS_ONLY
def test_accepts_genuine_siblings(fake_checkout: Path) -> None:
    # The real default layout - <leaf>-venv / <leaf>-data next to the checkout -
    # must NOT be rejected by the guard. It will fail LATER (no uv/python in this
    # throwaway env), but the refusal message must not be the collision guard.
    parent = fake_checkout.parent
    result = _run_bootstrap(
        fake_checkout, parent / "north-forge-agent-venv", parent / "north-forge-agent-data"
    )
    combined = result.stdout + result.stderr
    assert "is the checkout itself, is inside it, or" not in combined, (
        f"guard wrongly rejected a legitimate sibling layout:\n{combined}"
    )
