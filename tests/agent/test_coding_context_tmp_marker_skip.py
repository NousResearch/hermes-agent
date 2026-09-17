"""Regression: a stray project marker under the canonical /tmp must never flip
coding-context detection for sessions rooted under it — even when TMPDIR points
elsewhere (self-hosted CI runners set a deep TMPDIR but anchor pytest temproots
at /tmp). Found live: /tmp/AGENTS.md + runner TMPDIR≠/tmp made
resolve_runtime_mode report coding for a bare tmp cwd, failing
tests/agent/test_coding_context.py::test_resolves_general_outside_workspace.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agent import coding_context as cc


@pytest.fixture
def hostile_tmp_env(monkeypatch, tmp_path):
    """Stray marker directly in the canonical /tmp + TMPDIR pointing elsewhere."""
    marker = Path(tempfile.gettempdir()) / "AGENTS.md"
    if str(marker) == "/tmp/AGENTS.md" and marker.exists():
        # A stray marker is ALREADY present on this host (the live trigger);
        # do not remove or own it — just use it.
        created = None
    else:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("# stray\n")
        created = marker
    deep = tmp_path / "deep-tmpdir"
    deep.mkdir()
    monkeypatch.setenv("TMPDIR", str(deep))
    yield deep
    if created is not None:
        try:
            created.unlink()
        except OSError:
            pass


def _canonical_tmp() -> Path:
    # Resolve without TMPDIR influence so the canonical root is addressed even
    # while the fixture has TMPDIR pointed elsewhere.
    env = dict(os.environ)
    env.pop("TMPDIR", None)
    return Path("/tmp")


def test_stray_marker_in_canonical_tmp_does_not_flip_detection(hostile_tmp_env):
    cwd = tempfile.mkdtemp(prefix="hermes-pytest-tmproot-", dir=str(_canonical_tmp()))
    try:
        mode = cc.resolve_runtime_mode(platform="cli", cwd=cwd, config={})
        assert mode.is_coding is False, (
            "a stray manifest under the canonical /tmp must not make a bare tmp "
            "cwd resolve as a coding workspace"
        )
        assert cc._marker_root(Path(cwd)) is None
    finally:
        import shutil

        shutil.rmtree(cwd, ignore_errors=True)


def test_marker_root_skips_canonical_tmp_roots(hostile_tmp_env):
    cwd = _canonical_tmp() / "some-session-dir"
    # The function must return None (no marker root above a /tmp-rooted cwd),
    # proving /tmp was skipped rather than matched — even with TMPDIR pointed
    # elsewhere and a stray AGENTS.md sitting in /tmp.
    assert cc._marker_root(cwd) is None


def test_marker_root_still_finds_real_project(monkeypatch, tmp_path):
    """The skip extension must not over-skip: a genuine project root is still found."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    work = tmp_path / "pkg"
    work.mkdir()
    assert cc._marker_root(work) == tmp_path.resolve()
