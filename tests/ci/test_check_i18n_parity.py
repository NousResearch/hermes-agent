"""Tests for scripts/check_i18n_parity.mjs — the CI locale parity gate.

The checker is plain Node (no npm dependencies), so the JS lane runs it before
setup-node/npm-ci for a fast, install-free failure. The pytest lane mirrors
that gate here so that checker-only changes — and the locales/*.yaml files the
frontend classifier never routes to the JS lane — are still exercised on every
PR. The JS lane holds web/desktop locales to deep-merge parity (missing keys
fall back to English at runtime); the YAML agent catalogs ship complete
translations, so this lane additionally asserts their strict parity via the
same script the workflow calls.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "check_i18n_parity.mjs"


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    return subprocess.run(
        [node, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO,
    )


def test_all_surfaces_in_parity():
    """The exact command the JS lane runs must pass on the current corpus."""
    result = _run_checker("--all")
    assert result.returncode == 0, result.stdout + result.stderr
    # Sanity: the corpus was actually scanned, not silently empty.
    assert "All locale files in parity" in result.stdout


def test_strict_yaml_pair_passes():
    result = _run_checker("locales/en.yaml", "locales/fa.yaml")
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_key_fails_strict_but_passes_partial(tmp_path: Path):
    """Missing keys fail a strict pair, pass --partial (deep-merge locales)."""
    en = tmp_path / "en.yaml"
    loc = tmp_path / "xx.yaml"
    en.write_text("a:\n  b: one\n  c: two\n", encoding="utf-8")
    loc.write_text("a:\n  b: uno\n", encoding="utf-8")

    strict = _run_checker(str(en), str(loc))
    assert strict.returncode == 1
    assert "missing" in strict.stdout

    partial = _run_checker(str(en), str(loc), "--partial")
    assert partial.returncode == 0, partial.stdout + partial.stderr


def test_extra_key_fails_even_in_partial_mode(tmp_path: Path):
    """defineLocale() deep-merge tolerates missing keys — never stale ones."""
    en = tmp_path / "en.yaml"
    loc = tmp_path / "xx.yaml"
    en.write_text("a: one\n", encoding="utf-8")
    loc.write_text("a: uno\nstale: oops\n", encoding="utf-8")

    result = _run_checker(str(en), str(loc), "--partial")
    assert result.returncode == 1
    assert "extra" in result.stdout


def test_shape_drift_fails_partial_mode(tmp_path: Path):
    """TS locales: an interpolator leaf must stay callable.

    Compare two TS locale snippets: en exposes an arrow-function leaf, the
    locale turns it into a plain string — at runtime t.x() would render
    "undefined" instead of the interpolated text.
    """
    en = tmp_path / "en.ts"
    loc = tmp_path / "xx.ts"
    en.write_text(
        "export const en = {\n  greet: (name: string) => `hi ${name}`,\n}\n",
        encoding="utf-8",
    )
    loc.write_text(
        "export const xx = defineLocale({\n  greet: 'hi there',\n})\n",
        encoding="utf-8",
    )

    result = _run_checker(str(en), str(loc), "--partial")
    assert result.returncode == 1
    assert "shape" in result.stdout
