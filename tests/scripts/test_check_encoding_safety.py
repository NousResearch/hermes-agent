"""Calibration + unit tests for scripts/check-encoding-safety.py.

Gate 1 requirements:
  * All SIX historical instances (pre-fix shapes) MUST be flagged.
  * Fixed forms MUST produce zero findings.
  * Suppression comment must silence an otherwise-flaggable line.
  * Benign internal-only reads must not flag.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check-encoding-safety.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "encoding_safety"

# Historical instances that must each produce ≥1 finding (Gate 1).
HISTORICAL = {
    "h1_env_loader_primary.py": {"R1"},
    "h2_env_loader_fallback.py": {"R2"},
    "h3_send_cmd.py": {"R1", "R2"},
    "h4_sanitize.py": {"R3"},
    "h5_quote_env_read.py": {"R1", "R3"},
    "h6_managed_scope.py": {"R1"},
}

FIXED = [
    "f1_env_loader_primary.py",
    "f2_env_loader_fallback.py",
    "f3_send_cmd.py",
    "f4_sanitize.py",
    "f5_quote_env_read.py",
    "f6_managed_scope.py",
    "f_benign_internal.py",
]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_encoding_safety", CHECKER_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def _rules(findings) -> set[str]:
    return {f.rule for f in findings}


# ── Gate 1: historical fixtures ──────────────────────────────────────────


@pytest.mark.parametrize("filename,expected_rules", sorted(HISTORICAL.items()))
def test_historical_instance_flagged(checker, filename, expected_rules):
    path = FIXTURES / "historical" / filename
    findings = checker.scan_file(path)
    got = _rules(findings)
    missing = expected_rules - got
    assert not missing, (
        f"{filename}: expected rules {expected_rules}, got {got}. "
        f"Findings:\n"
        + "\n".join(f"  {f.rule} L{f.lineno}: {f.message[:80]}" for f in findings)
    )


def test_gate1_all_six_historical_flagged(checker):
    """Calibration gate: 6/6 historical instances produce ≥1 finding."""
    flagged = []
    for filename in HISTORICAL:
        findings = checker.scan_file(FIXTURES / "historical" / filename)
        if findings:
            flagged.append(filename)
    assert len(flagged) == 6, f"only {len(flagged)}/6 historical flagged: {flagged}"


# ── Gate 1: fixed forms ──────────────────────────────────────────────────


@pytest.mark.parametrize("filename", FIXED)
def test_fixed_form_clean(checker, filename):
    path = FIXTURES / "fixed" / filename
    findings = checker.scan_file(path)
    assert findings == [], (
        f"{filename} should be clean, got:\n"
        + "\n".join(f"  {f.rule} L{f.lineno}: {f.line_text}" for f in findings)
    )


# ── Suppression ──────────────────────────────────────────────────────────


def test_suppression_silences_r1(checker):
    path = FIXTURES / "suppression" / "suppressed_r1.py"
    findings = checker.scan_file(path)
    assert findings == [], f"suppressed file still flagged: {findings}"


def test_suppression_marker_regex(checker):
    assert checker.SUPPRESS_MARKER.search(
        'load_dotenv(p, encoding="utf-8")  # encoding-safety: ok — reason'
    )
    assert checker.SUPPRESS_MARKER.search(
        "x  # Encoding-Safety: OK because tests"
    )
    assert not checker.SUPPRESS_MARKER.search(
        'load_dotenv(p, encoding="utf-8")  # windows-footgun: ok'
    )


# ── CLI smoke ────────────────────────────────────────────────────────────


def test_cli_list_exits_zero(checker):
    assert checker.main(["--list"]) == 0


def test_cli_on_historical_exits_one(checker):
    hist = str(FIXTURES / "historical")
    rc = checker.main([hist])
    assert rc == 1


def test_cli_on_fixed_exits_zero(checker):
    fixed = str(FIXTURES / "fixed")
    rc = checker.main([fixed])
    assert rc == 0


def test_cli_on_suppression_exits_zero(checker):
    rc = checker.main([str(FIXTURES / "suppression")])
    assert rc == 0


# ── Allowlist is explicit ────────────────────────────────────────────────


def test_user_writable_allowlist_covers_documented_layout(checker):
    """Basenames from website/docs/user-guide/configuration.md Directory Structure."""
    required = {
        ".env",
        "config.yaml",
        "auth.json",
        "SOUL.md",
        "MEMORY.md",
        "USER.md",
        "SKILL.md",
    }
    assert required <= checker.USER_WRITABLE_BASENAMES


def test_path_looks_user_writable_helpers(checker):
    assert checker._path_looks_user_writable('home / ".env"')
    assert checker._path_looks_user_writable("env_path")
    assert checker._path_looks_user_writable('managed_dir / "config.yaml"')
    assert not checker._path_looks_user_writable('root / "schemas" / "internal_v1.json"')


# ── --all file collection ────────────────────────────────────────────────


def test_all_includes_repo_root_py_modules(checker, tmp_path, monkeypatch, capsys):
    """--all must scan non-recursive REPO_ROOT/*.py (e.g. cli.py), not only subdirs."""
    # Minimal package root so --all's package walk has something to do.
    pkg = tmp_path / "hermes_cli"
    pkg.mkdir()
    (pkg / "pkg_only.py").write_text("# package module, no R1\n", encoding="utf-8")

    # Top-level module with an R1 pattern (user-writable .env + encoding=utf-8).
    root_mod = tmp_path / "root_mod.py"
    root_mod.write_text(
        "from pathlib import Path\n"
        "env_path = Path.home() / '.env'\n"
        "with open(env_path, encoding='utf-8') as f:\n"
        "    f.read()\n",
        encoding="utf-8",
    )

    # Nested decoy: must not be double-walked if REPO_ROOT were a recursive root.
    nested = tmp_path / "other_pkg"
    nested.mkdir()
    (nested / "nested_mod.py").write_text(
        "from pathlib import Path\n"
        "env_path = Path.home() / '.env'\n"
        "with open(env_path, encoding='utf-8') as f:\n"
        "    f.read()\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    rc = checker.main(["--all"])
    captured = capsys.readouterr()
    out = captured.out + captured.err

    assert "root_mod.py" in out, f"top-level module missing from --all output:\n{out}"
    assert "nested_mod.py" not in out, (
        f"--all must not recurse REPO_ROOT; nested hit leaked:\n{out}"
    )
    # R1 on the root module is enough to fail the scan.
    assert rc == 1
    assert "[R1]" in out
