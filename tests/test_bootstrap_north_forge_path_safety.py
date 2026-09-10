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

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_PS1 = REPO_ROOT / "scripts" / "bootstrap-north-forge.ps1"
LIB_DIR = REPO_ROOT / "scripts" / "lib"


def _toolchain_env() -> dict:
    """A minimal-but-valid Windows environment (works under the canonical runner's
    ``env -i``) PLUS python / uv on PATH and uv's %LOCALAPPDATA% cache, so the
    cases that actually build a venv run under ``scripts/run_tests.sh`` too."""
    from tests._windows_env import minimal_windows_subprocess_env

    env = minimal_windows_subprocess_env()
    extra = [str(Path(sys.executable).parent), str(Path(sys.executable).parent / "Scripts")]
    uv = shutil.which("uv")
    if uv:
        extra.append(str(Path(uv).parent))
    env["PATH"] = os.pathsep.join(extra + [env["PATH"]])
    for key in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "UV_CACHE_DIR"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


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


@_WINDOWS_ONLY
@pytest.mark.parametrize("mangle", [
    lambda p: p,                       # venv == data exactly
    lambda p: str(p) + "\\",           # trailing separator
    lambda p: str(p).upper(),          # case difference
    lambda p: str(p / "sub"),          # data nested inside venv
])
def test_reject_venv_equals_or_contains_data(fake_checkout: Path, tmp_path: Path, mangle) -> None:
    """R1: VenvDir and DataDir must be checked against EACH OTHER, not only each
    against the checkout. A -Force rebuild wipes VenvDir - if DataDir is that path
    (or under it) the rebuild would destroy HERMES_HOME."""
    venv = tmp_path / "shared"
    result = _run_bootstrap(fake_checkout, venv, mangle(venv))
    combined = (result.stdout + result.stderr).lower()
    assert result.returncode != 0, f"accepted venv/data overlap {mangle(venv)!r}:\n{result.stdout}\n{result.stderr}"
    assert "data" in combined and ("same directory" in combined or "nested inside" in combined)
    assert (fake_checkout / "CANARY.txt").exists()


# ==========================================================================
# R1 - venv ownership/state matrix: bootstrap is the single cleanup authority
# ==========================================================================
#
# Behavioural only (no source-text assertions). Every case that deletes OR
# refuses proves a binary DataDir sentinel is byte-identical and the data-dir
# file listing is unchanged. Refusal cases additionally prove the venv dir was
# left exactly as it was.

_STATE_WINDOWS_ONLY = pytest.mark.skipif(
    os.name != "nt", reason="bootstrap-north-forge.ps1 runs under Windows PowerShell only"
)
_HAVE_PWSH = shutil.which("powershell") is not None


def _r1_fake_repo(root: Path) -> Path:
    """A throwaway checkout that a real `uv pip install -e .` can install: a
    valid pyproject with a `hermes` console-script, an importable ``hermes_cli``
    package, and copies of the three scripts under test."""
    repo = root / "north-forge-agent"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "hermes_cli").mkdir()
    (repo / "pyproject.toml").write_text(textwrap.dedent("""
        [build-system]
        requires = ["setuptools>=61"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "nf-fake"
        version = "0.0.0"

        [project.scripts]
        hermes = "hermes_cli:main"

        [tool.setuptools]
        packages = ["hermes_cli"]
    """).lstrip(), encoding="utf-8")
    (repo / "hermes_cli" / "__init__.py").write_text(
        "def main():\n    print('nf-fake hermes'); return 0\n", encoding="utf-8"
    )
    for name in ("nf-readiness.ps1", "nf-venv-state.ps1", "nf-toolchain.ps1"):
        shutil.copy(LIB_DIR / name, repo / "scripts" / "lib" / name)
    shutil.copy(BOOTSTRAP_PS1, repo / "scripts" / "bootstrap-north-forge.ps1")
    shutil.copy(REPO_ROOT / "scripts" / "nf-preflight.ps1", repo / "scripts" / "nf-preflight.ps1")
    # bootstrap best-effort-invokes this; a no-op keeps the run quiet.
    (repo / "scripts" / "make-drive-root-shortcut.ps1").write_text(
        "param([string]$RepoRoot)\n", encoding="utf-8"
    )
    return repo


def _data_with_sentinel(data: Path) -> tuple[str, list[str]]:
    """Create DataDir with a non-trivial binary sentinel; return (sha256, listing)."""
    data.mkdir(parents=True, exist_ok=True)
    blob = (b"\x00NF-DO-NOT-TOUCH\xff\x01\x02" * 41) + bytes(range(256))
    (data / "DO-NOT-TOUCH.bin").write_bytes(blob)
    (data / "config.yaml").write_text("display:\n  skin: crimson\n", encoding="utf-8")
    return _data_fingerprint(data)


def _data_fingerprint(data: Path) -> tuple[str, list[str]]:
    h = hashlib.sha256((data / "DO-NOT-TOUCH.bin").read_bytes()).hexdigest()
    listing = sorted(str(p.relative_to(data)).replace("\\", "/") for p in data.rglob("*"))
    return h, listing


def _venv_fingerprint(venv: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in sorted(venv.rglob("*")):
        rel = str(p.relative_to(venv)).replace("\\", "/")
        if p.is_file():
            out.append((rel, hashlib.sha256(p.read_bytes()).hexdigest()))
        else:
            out.append((rel + "/", ""))
    return out


def _real_venv(venv: Path, checkout: Path, *, marker_repo: str) -> None:
    """A genuine, importable venv: python -m venv + a .pth onto *checkout* so
    ``import hermes_cli`` resolves there, a hermes.exe stub, and a marker."""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"& (Get-Command python).Source -m venv '{venv}'"],
        capture_output=True, text=True, timeout=120, check=True, env=_toolchain_env(),
    )
    sp = venv / "Lib" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    (sp / "nf_stub.pth").write_text(str(checkout), encoding="utf-8")
    (venv / "Scripts" / "hermes.exe").write_text("", encoding="utf-8")
    (venv / ".nf-bootstrapped").write_text(f"repo={marker_repo}\nbootstrapped=test\n", encoding="utf-8")


def _run_bootstrap_repair(repo: Path, venv: Path, data: Path, *, force: bool = True) -> subprocess.CompletedProcess:
    args = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(repo / "scripts" / "bootstrap-north-forge.ps1"),
        "-RepoRoot", str(repo), "-VenvDir", str(venv), "-DataDir", str(data),
    ]
    if force:
        args.append("-Force")
    return subprocess.run(args, capture_output=True, text=True, timeout=300, env=_toolchain_env())


def _diag(text: str) -> dict[str, str]:
    """Parse the `[bootstrap] venv_state=.. ownership=.. action=.. reason=..` line."""
    for line in text.splitlines():
        if "[bootstrap] venv_state=" in line:
            body = line.split("[bootstrap] ", 1)[1]
            fields: dict[str, str] = {}
            for key in ("venv_state", "ownership", "action"):
                if key + "=" in body:
                    seg = body.split(key + "=", 1)[1]
                    fields[key] = seg.split(" ", 1)[0]
            return fields
    return {}


# id, setup(venv, checkout) -> None, force, expect_rc, expect_state, expect_action, venv_frozen
_R1_CASES = [
    pytest.param("absent", lambda v, c: None, False, 0, "Absent", "create", False, id="absent-creates"),
    pytest.param(
        "empty", lambda v, c: v.mkdir(parents=True), False, 0, "EmptyDirectory", "create", False,
        id="empty-populated-in-place",
    ),
    pytest.param(
        "artifact-only",
        lambda v, c: [(v / "Scripts").mkdir(parents=True), (v / "Scripts" / "hermes.exe").write_text("")],
        True, 1, "UnknownDirectory", "refuse", True, id="artifact-only-partial-refused",
    ),
    pytest.param(
        "occupied",
        lambda v, c: [v.mkdir(parents=True), (v / "thesis.docx").write_bytes(b"my important work")],
        True, 1, "UnknownDirectory", "refuse", True, id="unrelated-occupied-refused",
    ),
    pytest.param(
        "junction", "JUNCTION", True, 1, "UnsafePath", "refuse", True, id="reparse-point-refused",
    ),
    pytest.param(
        "interrupted-venv",
        lambda v, c: [
            (v / "Scripts").mkdir(parents=True),
            (v / "pyvenv.cfg").write_text(
                "home = C:\\Python311\ninclude-system-site-packages = false\nversion = 3.11.9\n"
            ),
            (v / "Scripts" / "activate").write_text("rem"),
        ],
        False, 0, "RecognizablePythonVenv", "rebuild", False, id="valid-pyvenv-cfg-missing-python",
    ),
]


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
@pytest.mark.parametrize("name,setup,force,rc,state,action,venv_frozen", _R1_CASES)
def test_venv_state_matrix(tmp_path, name, setup, force, rc, state, action, venv_frozen):
    repo = _r1_fake_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    before_hash, before_list = _data_with_sentinel(data)

    if setup == "JUNCTION":
        target = tmp_path / "junction-target"
        target.mkdir()
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(venv), str(target)],
            capture_output=True, text=True, check=True,
        )
    elif callable(setup):
        setup(venv, repo)

    venv_before = _venv_fingerprint(venv) if venv.exists() else None

    r = _run_bootstrap_repair(repo, venv, data, force=force)
    combined = r.stdout + r.stderr

    assert r.returncode == rc, f"[{name}] rc={r.returncode} want {rc}\n{combined}"
    diag = _diag(combined)
    assert diag.get("venv_state") == state, f"[{name}] venv_state={diag.get('venv_state')!r} want {state!r}\n{combined}"
    assert diag.get("action") == action, f"[{name}] action={diag.get('action')!r} want {action!r}\n{combined}"

    # DataDir sentinel + listing: unchanged, every case.
    after_hash, after_list = _data_fingerprint(data)
    assert after_hash == before_hash, f"[{name}] DataDir sentinel bytes changed"
    assert after_list == before_list, f"[{name}] DataDir file listing changed: {before_list} -> {after_list}"

    if venv_frozen:
        assert _venv_fingerprint(venv) == venv_before, f"[{name}] a refused venv dir was modified"
        assert "nothing was deleted" in combined.lower()

    if action == "create":
        assert (venv / "Scripts" / "python.exe").exists(), f"[{name}] venv not created\n{combined}"
    if action == "rebuild":
        assert (venv / "Scripts" / "python.exe").exists(), f"[{name}] venv not rebuilt\n{combined}"
        # a rebuild must not have re-seeded the skin into the data folder
        assert not (data / "skins").exists(), f"[{name}] rebuild seeded skins/ into DataDir"


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_healthy_venv_is_left_alone_and_second_run_is_idempotent(tmp_path):
    """Sequence: interrupted creation -> repair rebuilds it -> a second launch is
    a no-op (action=none, no second rebuild) -> DataDir sentinel unchanged across
    BOTH runs."""
    repo = _r1_fake_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    before_hash, before_list = _data_with_sentinel(data)

    # interrupted: a real pyvenv.cfg + Scripts\, but no python.exe
    (venv / "Scripts").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        "home = C:\\Python311\ninclude-system-site-packages = false\nversion = 3.11.9\n", encoding="utf-8"
    )

    r1 = _run_bootstrap_repair(repo, venv, data, force=False)
    assert r1.returncode == 0, f"first run (recovery) failed:\n{r1.stdout}\n{r1.stderr}"
    assert _diag(r1.stdout + r1.stderr).get("action") == "rebuild"
    assert (venv / "Scripts" / "python.exe").exists()

    mid_hash, mid_list = _data_fingerprint(data)
    assert (mid_hash, mid_list) == (before_hash, before_list), "DataDir changed during recovery"

    r2 = _run_bootstrap_repair(repo, venv, data, force=False)
    assert r2.returncode == 0, f"second run failed:\n{r2.stdout}\n{r2.stderr}"
    d2 = _diag(r2.stdout + r2.stderr)
    assert d2.get("action") == "none", f"second run rebuilt again: {d2}\n{r2.stdout}"
    assert d2.get("venv_state") == "Ready"

    after_hash, after_list = _data_fingerprint(data)
    assert (after_hash, after_list) == (before_hash, before_list), "DataDir changed on the idempotent second run"


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_foreign_marker_venv_is_rebuilt_not_refused(tmp_path):
    """A venv whose ONLY defect is a foreign marker repo= is still North-Forge
    -owned (evidence 1) -> rebuild, not refuse. DataDir untouched."""
    repo = _r1_fake_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    before = _data_with_sentinel(data)
    _real_venv(venv, repo, marker_repo="D:\\north-forge-agent-on-another-pc")

    r = _run_bootstrap_repair(repo, venv, data, force=False)
    diag = _diag(r.stdout + r.stderr)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert diag.get("ownership") == "north-forge"
    assert diag.get("action") == "rebuild"
    assert _data_fingerprint(data) == before


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_venv_path_with_spaces_is_handled(tmp_path):
    repo = _r1_fake_repo(tmp_path)
    venv = tmp_path / "north forge agent venv"   # spaces
    data = tmp_path / "north forge agent data"
    before = _data_with_sentinel(data)

    r = _run_bootstrap_repair(repo, venv, data, force=False)
    combined = r.stdout + r.stderr
    assert r.returncode == 0, combined
    assert _diag(combined).get("action") == "create"
    assert (venv / "Scripts" / "python.exe").exists(), combined
    assert _data_fingerprint(data) == before


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_force_does_not_override_unknown_directory(tmp_path):
    """-Force is a rebuild REQUEST, never a deletion override: an UnknownDirectory
    is refused even with -Force, and nothing is deleted."""
    repo = _r1_fake_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"
    before = _data_with_sentinel(data)
    venv.mkdir()
    (venv / "someone-elses-project").mkdir()
    (venv / "someone-elses-project" / "main.c").write_text("int main(){}", encoding="utf-8")
    venv_before = _venv_fingerprint(venv)

    r = _run_bootstrap_repair(repo, venv, data, force=True)   # -Force
    combined = r.stdout + r.stderr
    assert r.returncode != 0, combined
    assert _diag(combined).get("action") == "refuse"
    assert _venv_fingerprint(venv) == venv_before, "a -Force run deleted an UnknownDirectory"
    assert _data_fingerprint(data) == before


# ==========================================================================
# DECISION-2026-09-09-001 - drive-native bundled toolchain
# ==========================================================================
#
# A provisioned drive carries <parent>\<leaf>-toolchain\{uv\uv.exe, python\python.exe}
# (uv + a python-build-standalone CPython, prepared once by the admin, never
# git-tracked). bootstrap must:
#   * use it silently when present - zero PATH dependency, zero network, no prompt;
#   * fall back to uv/python on PATH otherwise, LOGGED as "using host toolchain
#     (admin/dev)" so the two are never confused;
#   * with neither, emit today's exact existing error, unchanged.
# The toolchain folder is an INPUT to venv creation - never classified, deleted,
# or treated as ownership evidence by R1 (nothing here touches nf-venv-state.ps1
# or nf-readiness.ps1).

_HAVE_UV = shutil.which("uv") is not None


def _min_env_no_toolchain() -> dict:
    """minimal_windows_subprocess_env() (System32 + PowerShell only - NO python,
    NO uv) plus the vars uv needs for its own cache/home, so a bundled-toolchain
    build can run fully offline off a warm uv cache."""
    from tests._windows_env import minimal_windows_subprocess_env

    env = minimal_windows_subprocess_env()
    for key in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "UV_CACHE_DIR"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _stage_bundled_toolchain(parent: Path, leaf: str) -> Path:
    r"""Build ``<parent>\<leaf>-toolchain\`` : a real uv.exe copied in, and
    ``python\`` as a junction onto this host's standalone CPython base (its layout
    - python.exe at the root - is exactly the python-build-standalone shape the
    real bundled interpreter has)."""
    tc = parent / f"{leaf}-toolchain"
    (tc / "uv").mkdir(parents=True)
    shutil.copy(shutil.which("uv"), tc / "uv" / "uv.exe")
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(tc / "python"), sys.base_prefix],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"mklink /J failed: {r.stdout}\n{r.stderr}"
    assert (tc / "python" / "python.exe").exists(), "staged bundled interpreter missing python.exe"
    return tc


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
@pytest.mark.skipif(not _HAVE_UV, reason="no uv on this host to stage as the bundled toolchain")
def test_bundled_toolchain_builds_venv_with_zero_path_dependency(tmp_path):
    repo = _r1_fake_repo(tmp_path)                      # tmp_path/north-forge-agent
    _stage_bundled_toolchain(tmp_path, repo.name)       # tmp_path/north-forge-agent-toolchain
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"

    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(repo / "scripts" / "bootstrap-north-forge.ps1"),
         "-RepoRoot", str(repo), "-VenvDir", str(venv), "-DataDir", str(data)],
        capture_output=True, text=True, timeout=420, env=_min_env_no_toolchain(),
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "[bootstrap] toolchain uv=bundled python=bundled" in out, out
    assert "bundled drive toolchain" in out, out
    assert "using host toolchain (admin/dev)" not in out, out
    assert (venv / "Scripts" / "python.exe").exists(), out   # venv built, no python/uv on PATH
    assert _diag(out).get("action") == "create"


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_no_bundled_toolchain_falls_back_to_host_path_logged_distinctly(tmp_path):
    repo = _r1_fake_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"

    r = _run_bootstrap_repair(repo, venv, data, force=False)   # _toolchain_env(): python + uv ON PATH
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "using host toolchain (admin/dev)" in out, out
    assert "toolchain: bundled drive toolchain" not in out, out
    assert "python=bundled" not in out and "uv=bundled" not in out, out
    assert "[bootstrap] toolchain uv=host" in out, out
    assert (venv / "Scripts" / "python.exe").exists(), out


@_STATE_WINDOWS_ONLY
@pytest.mark.skipif(not _HAVE_PWSH, reason="powershell not on PATH")
def test_no_toolchain_anywhere_gives_the_exact_existing_error(tmp_path):
    from tests._windows_env import minimal_windows_subprocess_env

    repo = _r1_fake_repo(tmp_path)
    venv = tmp_path / "north-forge-agent-venv"
    data = tmp_path / "north-forge-agent-data"

    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(repo / "scripts" / "bootstrap-north-forge.ps1"),
         "-RepoRoot", str(repo), "-VenvDir", str(venv), "-DataDir", str(data)],
        capture_output=True, text=True, timeout=120,
        env=minimal_windows_subprocess_env(),   # NO python, NO uv, nothing extra
    )
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "No 'uv' and no 'python' on PATH. Install Python 3.11+ or uv, then re-run." in out, out
    assert not (venv / "Scripts" / "python.exe").exists()


def test_toolchain_resolver_source_is_read_only_and_not_ownership_evidence() -> None:
    """Source-level, portable: nf-toolchain.ps1 never writes/deletes/downloads,
    and bootstrap dot-sources it and prefers the bundled path before any PATH
    lookup - without touching the R1 ownership/readiness decision."""
    tc = (LIB_DIR / "nf-toolchain.ps1").read_text(encoding="utf-8")
    for banned in ("Remove-Item", "New-Item", "Set-Content", "Out-File",
                   "Invoke-WebRequest", "Start-BitsTransfer", "curl", "exit "):
        assert banned not in tc, f"nf-toolchain.ps1 must be read-only; found {banned!r}"

    src = _source()
    assert r"lib\nf-toolchain.ps1" in src, "bootstrap must dot-source the toolchain resolver"
    i_tc = src.index("Get-NfBundledToolchain -RepoRoot")
    i_uv = src.index("Get-Command uv -ErrorAction SilentlyContinue")
    assert i_tc < i_uv, "the bundled toolchain must be resolved BEFORE any PATH lookup"
    assert "using host toolchain (admin/dev)" in src, "PATH fallback must be logged distinctly"
    # R1 decision is upstream and untouched: the state table still precedes this.
    assert src.index("action=$action reason=$reason") < i_tc
