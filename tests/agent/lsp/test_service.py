"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _install_mock_server(monkeypatch, script: str = "errors", server_id: str = "pyright",
                         sequence: str = "", resolve_root=None):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.

    ``sequence`` is forwarded as ``MOCK_LSP_SEQUENCE`` for the
    ``"sequence"`` script (a per-publish list of ``error``/``clean``).

    ``resolve_root`` overrides the server's root resolver — pass one that
    returns a nested directory to exercise the per-server-root vs git-root
    distinction.  Defaults to "always the git workspace root".
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        if sequence:
            env["MOCK_LSP_SEQUENCE"] = sequence
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=resolve_root or (lambda fp, ws: ws),  # default: git root
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    # Patch the SERVERS list element directly + restore on teardown.
    SERVERS[target_index] = replacement

    yield

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")  # so pyright's root resolver finds it
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass


def test_service_returns_empty_when_disabled(tmp_path):
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="auto",
    )
    assert not svc.is_active()
    f = tmp_path / "x.py"
    f.write_text("")
    assert svc.get_diagnostics_sync(str(f)) == []
    svc.shutdown()


def test_service_skips_files_outside_workspace(tmp_path):
    """Files outside any git worktree must not trigger LSP."""
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    f = tmp_path / "x.py"
    f.write_text("")
    # No .git anywhere — service should report not enabled for this file.
    assert not svc.enabled_for(str(f))
    svc.shutdown()


def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))
        # Baseline first — server pushes 1 error.
        svc.snapshot_baseline(str(f))
        # Re-poll: same error is in baseline, so delta is empty.
        new_diags = svc.get_diagnostics_sync(str(f))
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_e2e_delta_filter_with_line_shift(mock_pyright):
    """End-to-end: an edit that shifts the diagnostic's line still
    filters correctly when ``line_shift`` is supplied.

    The mock LSP server emits a fixed error at line 0; for this test
    we don't need to actually shift the server's output — we just
    need to prove that supplying a line_shift through the API works
    and doesn't break the existing delta path.  The unit tests in
    test_delta_key.py cover the shift semantics in detail.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.snapshot_baseline(str(f))
        # Identity shift — should behave exactly like no shift.
        new_diags = svc.get_diagnostics_sync(str(f), line_shift=lambda L: L)
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_status_includes_clients(mock_pyright):
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.get_diagnostics_sync(str(f))
        info = svc.get_status()
        assert info["enabled"] is True
        assert any(c["server_id"] == "pyright" for c in info["clients"])
    finally:
        svc.shutdown()


@pytest.fixture
def mock_pyright_sequence(monkeypatch, tmp_path):
    """Like ``mock_pyright`` but the server emits a per-publish sequence
    of ``error``/``clean`` states (``MOCK_LSP_SEQUENCE``), so a test can
    drive an old-error → clean → later-edit lifecycle deterministically.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(
        monkeypatch, "sequence", "pyright", sequence="error,clean,error"
    )
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass


def test_service_clears_stale_baseline_after_clean_edit(mock_pyright_sequence):
    """Regression for stale LSP diagnostics surviving a clean edit.

    Sequence of server publishes: error → clean → error.

    1. ``snapshot_baseline`` captures the pre-edit error as the baseline.
    2. A clean edit yields no diagnostics; the empty fresh snapshot must
       roll the baseline *forward* (clearing it) rather than leaving the
       now-fixed error cached.
    3. A later edit re-introduces the same error.  With a stale baseline
       still holding the old error, the delta filter would wrongly mask
       this genuine new diagnostic; with the baseline correctly cleared,
       it is surfaced.
    """
    repo = mock_pyright_sequence
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))

        # 1. Baseline captures the pre-existing error (publish #1).
        svc.snapshot_baseline(str(f))

        # 2. Clean edit (publish #2): delta is empty, and the empty fresh
        #    snapshot must clear the cached baseline.
        assert svc.get_diagnostics_sync(str(f)) == []

        # 3. Later edit re-introduces the error (publish #3).  It must be
        #    reported — a stale baseline would suppress it.
        later = svc.get_diagnostics_sync(str(f))
        assert len(later) == 1
        assert later[0]["code"] == "MOCK001"
    finally:
        svc.shutdown()


@pytest.fixture
def mock_pyright_nested_root(monkeypatch, tmp_path):
    """Install the mock with a per-server root that differs from the git
    root: git root is ``repo/``, but the server resolves to the nested
    ``repo/pkg/`` package.  Exercises the ``_current_diags_async`` client
    lookup, which must key on the per-server root, not the git root.
    """
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (repo / ".git").mkdir()
    (pkg / "pyproject.toml").write_text("")
    monkeypatch.chdir(str(repo))

    def _nested_root(fp, ws):
        # Server root is the directory containing the edited file (the
        # nested package), not the git workspace root.
        return os.path.dirname(os.path.abspath(fp))

    gen = _install_mock_server(
        monkeypatch, "errors", "pyright", resolve_root=_nested_root
    )
    next(gen)
    yield pkg
    try:
        next(gen)
    except StopIteration:
        pass


def test_service_baseline_rolls_forward_under_nested_root(mock_pyright_nested_root):
    """Pre-existing diagnostics must stay filtered across repeated polls
    even when the server's root differs from the git root.

    The roll-forward reads the live diagnostics via ``_current_diags_async``;
    if that lookup uses the wrong client key it returns a false-empty
    snapshot, clears the baseline, and the pre-existing error wrongly
    re-surfaces on the next unchanged poll.
    """
    pkg = mock_pyright_nested_root
    f = pkg / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))
        # Baseline captures the pre-existing error.
        svc.snapshot_baseline(str(f))
        # First poll: error is in the baseline, so delta is empty.
        assert svc.get_diagnostics_sync(str(f)) == []
        # Second unchanged poll: the same pre-existing error must remain
        # filtered — the baseline must not have been cleared by a
        # false-empty roll-forward.
        assert svc.get_diagnostics_sync(str(f)) == []
    finally:
        svc.shutdown()
