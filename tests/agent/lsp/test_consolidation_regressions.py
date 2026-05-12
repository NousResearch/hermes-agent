"""Regression tests for defects discovered during PR #24168 LSP consolidation review.

Six defects (D1-D6) addressed by the consolidation patch.  Each test
below pins the EXPECTED post-fix behavior; running these against
unmodified #24168 HEAD should produce specific, documented failures
(D1 fails outright; D3-D6 fail on attribute/identity assertions).

Defect classes
==============
D1  patch_replace double-lint silently drops every LSP diagnostic
    because the recursive write_file call rolls the baseline forward.
    Fix: token-keyed baseline + threading the token through.
D2  Path-keyed _delta_baseline collides under concurrent gateway load.
    Fix: same token-keyed map.
D3  Idle LSP subprocesses never reaped (one per (lang, workspace) for
    the life of the gateway process).
    Fix: background asyncio reaper.
D4  Workspace cache unbounded and not thread-safe.
    Fix: OrderedDict LRU + lock.
D5  Default ``install_strategy="auto"`` triggers silent npm/go/pip
    installs on first edit — SOC 2 / ISO 27001 problem.
    Fix: default to "manual".
D6  INSTALL_RECIPES unpinned (``pyright``, ``gopls@latest``).
    Fix: pin every auto-install recipe to a verified version.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import pytest

from agent.lsp import eventlog


@pytest.fixture(autouse=True)
def _reset():
    """Reset module-level LSP state between tests so they don't bleed."""
    eventlog.reset_announce_caches()
    try:
        from agent.lsp.workspace import _workspace_cache, _workspace_cache_lock
        with _workspace_cache_lock:
            _workspace_cache.clear()
    except Exception:
        pass
    import agent.lsp as lsp_mod
    if hasattr(lsp_mod, "_service"):
        lsp_mod._service = None
    yield


# ---------------------------------------------------------------------------
# Shared faithful LSP service mock
# ---------------------------------------------------------------------------

class FaithfulLSPServiceMock:
    """Mock that mirrors the real LSPService contract closely.

    Maintains an explicit ``current_diags`` list which is the
    "ground truth" the server would report at any given moment.
    ``snapshot_baseline`` calls capture a COPY of that list (not an
    empty list as a shortcut), exactly as the real service does via
    ``_snapshot_async``.

    Tests can mutate ``current_diags`` to simulate pre-write vs.
    post-write states.  ``tokens`` is exposed so tests can assert
    eviction.
    """

    def __init__(self, current_diags: Optional[List[Dict[str, Any]]] = None):
        self.current_diags: List[Dict[str, Any]] = list(current_diags or [])
        self.tokens: Dict[str, List[Dict[str, Any]]] = {}
        self._next_id = 0
        self.snapshot_calls: List[str] = []
        self.diag_calls: List[Dict[str, Any]] = []

    def enabled_for(self, p: str) -> bool:
        return True

    def snapshot_baseline(self, p: str) -> str:
        self._next_id += 1
        tok = f"tok-{self._next_id}"
        # Capture a snapshot of current diagnostics at this moment —
        # matches LSPService.snapshot_baseline's behavior of fetching
        # the server's current view.
        self.tokens[tok] = list(self.current_diags)
        self.snapshot_calls.append(p)
        return tok

    def get_diagnostics_sync(self, p: str, delta: bool = True,
                              baseline_token: Optional[str] = None) -> List[Dict[str, Any]]:
        self.diag_calls.append({"path": p, "delta": delta, "token": baseline_token})
        diags = list(self.current_diags)
        if delta and baseline_token is not None:
            # Real service POPS the token (consumes it).  We mirror that
            # so any double-consumption shows up immediately.
            baseline = self.tokens.pop(baseline_token, [])
            seen = {(d["message"], d.get("code")) for d in baseline}
            diags = [d for d in diags if (d["message"], d.get("code")) not in seen]
        return diags


# ---------------------------------------------------------------------------
# Defect 1: patch_replace double-lint drops LSP diagnostics
# ---------------------------------------------------------------------------

def test_patch_replace_surfaces_lsp_diagnostic_introduced_by_edit(tmp_path, monkeypatch):
    """The headline D1 fix: a real type error introduced by `patch_replace`
    must appear in PatchResult.lint.  On #24168 HEAD this fails because
    the recursive write_file call rolls the baseline forward; the outer
    patch_replace _check_lint_delta then sees an empty delta.

    Faithful mock: ``current_diags`` is mutated BETWEEN the snapshot
    and the post-write call to simulate "no diagnostics before, one
    diagnostic after".  This is exactly what a real LSP server would
    do for the edit ``return 0`` → ``return 'str'`` on a ``-> int`` func.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    workspace = tmp_path / "proj"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    target = workspace / "broken.py"
    target.write_text("def f() -> int:\n    return 0\n")

    fops = ShellFileOperations(LocalEnvironment(cwd=str(workspace)))

    diag = {
        "severity": 1,
        "message": "Type 'str' not assignable to 'int'",
        "code": "reportReturnType",
        "source": "Pyright",
        "range": {"start": {"line": 1, "character": 11}, "end": {"line": 1, "character": 16}},
    }
    # Pre-write: no diagnostics (the file was valid before the edit).
    mock = FaithfulLSPServiceMock(current_diags=[])

    # Patch the LSP service accessor so the diagnostic mutation can be
    # threaded into the write_file flow.  We simulate the real-server
    # behavior by flipping current_diags right after snapshot_baseline
    # is called (i.e. between baseline capture and post-write fetch).
    original_snapshot = mock.snapshot_baseline
    def snapshot_then_introduce_error(p):
        tok = original_snapshot(p)
        mock.current_diags = [diag]  # post-edit state
        return tok
    mock.snapshot_baseline = snapshot_then_introduce_error

    monkeypatch.setattr("agent.lsp.get_service", lambda: mock)
    monkeypatch.setattr(
        "tools.file_operations.ShellFileOperations._lsp_local_only",
        lambda self: True,
    )

    result = fops.patch_replace(str(target), "return 0", "return 'str'")

    assert result.success, f"patch failed: {result.error}"
    lint = result.lint
    assert lint is not None, "lint result missing entirely"
    output = lint.get("output", "") if isinstance(lint, dict) else ""
    assert "reportReturnType" in output or "Type 'str' not assignable" in output, (
        f"patch_replace dropped the LSP diagnostic.\nlint output: {output!r}"
    )

    # Pin the token-pop contract: after the run, the snapshot's token
    # MUST have been consumed.  If a future refactor switches to non-
    # destructive .get, this assertion fires.
    assert mock.tokens == {}, (
        f"baseline token leaked: {list(mock.tokens.keys())}.  "
        "get_diagnostics_sync must POP the token, not GET it."
    )


def test_patch_replace_inner_write_does_not_double_consume_lsp(tmp_path, monkeypatch):
    """The fix routes through the write_file ``_lsp_baseline_token`` escape
    hatch + ``skip_lsp=True`` for the inner _check_lint_delta call.  This
    test pins that contract: snapshot_baseline must be called exactly
    ONCE per patch_replace, and get_diagnostics_sync must be called
    exactly ONCE on the patch_replace's token.

    On a buggy refactor that snapshots twice (e.g. removes the token
    hand-off) or runs LSP twice (e.g. removes skip_lsp), this fails.
    """
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    workspace = tmp_path / "proj"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    target = workspace / "ok.py"
    target.write_text("x = 1\n")

    fops = ShellFileOperations(LocalEnvironment(cwd=str(workspace)))
    mock = FaithfulLSPServiceMock(current_diags=[])
    monkeypatch.setattr("agent.lsp.get_service", lambda: mock)
    monkeypatch.setattr(
        "tools.file_operations.ShellFileOperations._lsp_local_only",
        lambda self: True,
    )

    result = fops.patch_replace(str(target), "x = 1", "x = 2")
    assert result.success

    assert len(mock.snapshot_calls) == 1, (
        f"snapshot_baseline called {len(mock.snapshot_calls)} times; "
        f"expected exactly 1 (patch_replace owns the snapshot)"
    )
    # exactly one diagnostic fetch using a token (the outer patch_replace's)
    token_calls = [c for c in mock.diag_calls if c["token"] is not None]
    assert len(token_calls) == 1, (
        f"token-bearing get_diagnostics_sync called {len(token_calls)} times; "
        f"expected exactly 1: {mock.diag_calls}"
    )


def test_snapshot_baseline_returns_none_when_lsp_disabled():
    """LSP off → snapshot_baseline returns None; downstream code must
    handle None at every call site without TypeError."""
    from agent.lsp.manager import LSPService

    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
    )
    try:
        assert svc.snapshot_baseline("/some/path.py") is None
        # get_diagnostics_sync with baseline_token=None must not crash
        result = svc.get_diagnostics_sync(
            "/some/path.py", delta=True, baseline_token=None
        )
        assert result == []
    finally:
        svc.shutdown()


def test_baseline_token_evicted_after_get_diagnostics_sync():
    """Pin the pop-not-get contract: consuming a token MUST remove it
    from _delta_baseline.  Catches the regression where get_diagnostics_sync
    accidentally switches to .get() and leaks tokens forever."""
    from agent.lsp.manager import LSPService

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
    )
    try:
        # Pre-seed a baseline entry to avoid needing a real LSP server.
        token = "fake-token-abc"
        with svc._baseline_lock:
            svc._delta_baseline[token] = {
                "path": "/x.py",
                "diags": [],
                "created": time.time(),
            }
        assert token in svc._delta_baseline

        # Bypass the actual diagnostic fetch (no server spawned) by
        # monkeypatching the inner async path.  We're testing the pop
        # behavior, not the diagnostic content.
        async def fake_open_wait(file_path):
            return []
        svc._open_and_wait_async = fake_open_wait
        svc._enabled = True
        # enabled_for() also gates on resolve_workspace_for_file — bypass:
        svc.enabled_for = lambda p: True

        svc.get_diagnostics_sync("/x.py", delta=True, baseline_token=token)

        assert token not in svc._delta_baseline, (
            f"token {token!r} leaked: still in _delta_baseline after consume"
        )
    finally:
        svc.shutdown()


def test_evict_stale_baselines_drops_old_tokens():
    """Tokens older than the 5-minute TTL must be evicted.  Catches the
    unbounded-growth regression where crashed writes never pop their
    token and the map fills."""
    from agent.lsp.manager import LSPService

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
    )
    try:
        # One fresh token, one stale token.
        with svc._baseline_lock:
            svc._delta_baseline["fresh"] = {
                "path": "/a.py", "diags": [], "created": time.time(),
            }
            svc._delta_baseline["stale"] = {
                "path": "/b.py", "diags": [], "created": time.time() - 600,  # 10 min ago
            }

        svc._evict_stale_baselines()

        with svc._baseline_lock:
            assert "fresh" in svc._delta_baseline
            assert "stale" not in svc._delta_baseline
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Defect 2: token-keyed baseline isolates concurrent edits
# ---------------------------------------------------------------------------

def test_concurrent_snapshot_baseline_returns_unique_tokens():
    """50 threads each calling snapshot_baseline(same_path) simultaneously
    must produce 50 distinct tokens, all coexisting in _delta_baseline.

    On the old path-keyed design, only the last write survives — chats
    stomp each other.  The token-keyed design must isolate them.
    """
    from agent.lsp.manager import LSPService

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
    )
    try:
        # Bypass the real LSP roundtrip with a stub _snapshot_async.
        async def fake_snapshot(file_path):
            return []
        svc._snapshot_async = fake_snapshot
        svc.enabled_for = lambda p: True

        tokens: List[Optional[str]] = []
        tokens_lock = threading.Lock()
        barrier = threading.Barrier(50)

        def worker():
            barrier.wait()  # force simultaneous entry
            tok = svc.snapshot_baseline("/shared/path.py")
            with tokens_lock:
                tokens.append(tok)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(tokens) == 50
        non_none = [t for t in tokens if t is not None]
        assert len(non_none) == 50, "every snapshot must return a token"
        assert len(set(non_none)) == 50, (
            f"token collision: {len(non_none)} calls but only "
            f"{len(set(non_none))} unique tokens"
        )

        with svc._baseline_lock:
            for tok in non_none:
                assert tok in svc._delta_baseline, (
                    f"token {tok!r} missing from _delta_baseline — "
                    f"concurrent writers stomped each other"
                )
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Defect 3: idle clients reaped
# ---------------------------------------------------------------------------

def test_idle_reaper_exists_and_reaps_stale_clients():
    """``_reap_idle`` is the unit-testable single-pass reaper.  Drive it
    once and verify a backdated client is removed and shutdown()'d."""
    from agent.lsp.manager import LSPService

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0.1,
    )
    try:
        assert hasattr(svc, "_reap_idle")
        assert hasattr(svc, "_reaper_loop")
        assert svc._reaper_handle is not None, "reaper not scheduled in __init__"

        class FakeClient:
            def __init__(self):
                self.server_id = "fake"
                self.workspace_root = "/tmp/fake-ws"
                self.shutdown_called = False
            async def shutdown(self):
                self.shutdown_called = True

        fake = FakeClient()
        key = ("fake", "/tmp/fake-ws")
        with svc._state_lock:
            svc._clients[key] = fake
            svc._last_used[key] = time.time() - 600.0

        svc._loop.run(svc._reap_idle(), timeout=5.0)

        with svc._state_lock:
            assert key not in svc._clients
            assert key not in svc._last_used
        assert fake.shutdown_called
    finally:
        svc.shutdown()


def test_idle_reaper_leaves_fresh_clients_alone():
    """Recently-used clients survive a reap pass."""
    from agent.lsp.manager import LSPService

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=300.0,
    )
    try:
        class FakeClient:
            def __init__(self):
                self.server_id = "fresh"
                self.workspace_root = "/tmp/fresh-ws"
                self.shutdown_called = False
            async def shutdown(self):
                self.shutdown_called = True

        fake = FakeClient()
        key = ("fresh", "/tmp/fresh-ws")
        with svc._state_lock:
            svc._clients[key] = fake
            svc._last_used[key] = time.time()

        svc._loop.run(svc._reap_idle(), timeout=5.0)

        with svc._state_lock:
            assert key in svc._clients
        assert not fake.shutdown_called
    finally:
        svc.shutdown()


def test_reaper_handle_cancelled_on_shutdown():
    """``shutdown()`` must cancel the scheduled reaper task so the
    background loop can exit cleanly.  Without this, the loop holds
    a reference to a coroutine that's been promised to keep running
    forever, leaking the thread on process exit."""
    from agent.lsp.manager import LSPService

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )
    handle = svc._reaper_handle
    assert handle is not None
    svc.shutdown()
    # After shutdown(): handle either cancelled or done; in either case
    # the LSPService cleared its own reference.
    assert svc._reaper_handle is None
    # The future itself: cancelled or completed without exception.
    assert handle.cancelled() or handle.done()


# ---------------------------------------------------------------------------
# Defect 4: workspace cache LRU + lock
# ---------------------------------------------------------------------------

def test_workspace_cache_is_bounded_lru_and_locked():
    """Cache is an OrderedDict guarded by a Lock, bounded by
    _WORKSPACE_CACHE_MAX, with LRU eviction."""
    from agent.lsp import workspace

    assert isinstance(workspace._workspace_cache, OrderedDict)
    assert hasattr(workspace, "_workspace_cache_lock")
    assert hasattr(workspace, "_WORKSPACE_CACHE_MAX")
    assert workspace._WORKSPACE_CACHE_MAX > 0

    with workspace._workspace_cache_lock:
        workspace._workspace_cache.clear()
    cap = workspace._WORKSPACE_CACHE_MAX
    for i in range(cap + 50):
        workspace._cache_set(f"/synthetic/path/{i}", (None, False))
    assert len(workspace._workspace_cache) == cap
    assert "/synthetic/path/0" not in workspace._workspace_cache
    assert f"/synthetic/path/{cap + 49}" in workspace._workspace_cache

    # Touch a middle entry: it should be promoted past the oldest survivor.
    middle_key = f"/synthetic/path/{cap}"
    assert workspace._cache_get(middle_key) is not None
    # Now write one more entry — the oldest non-touched should evict, not
    # the middle one we just touched.
    workspace._cache_set("/synthetic/path/extra", (None, False))
    assert middle_key in workspace._workspace_cache, (
        "LRU eviction killed a recently-touched entry — move_to_end missing?"
    )

    with workspace._workspace_cache_lock:
        workspace._workspace_cache.clear()


def test_workspace_cache_concurrent_writes_are_safe():
    """20 threads × 100 writes hit the cache simultaneously (Barrier
    forces entry-at-once).  Cap is enforced and the OrderedDict is
    not corrupted (iteration succeeds without KeyError)."""
    from agent.lsp import workspace

    with workspace._workspace_cache_lock:
        workspace._workspace_cache.clear()
    barrier = threading.Barrier(20)

    def worker(thread_id):
        barrier.wait()  # force simultaneous entry
        for i in range(100):
            workspace._cache_set(f"/t{thread_id}/p{i}", (None, False))
            workspace._cache_get(f"/t{thread_id}/p{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(workspace._workspace_cache) <= workspace._WORKSPACE_CACHE_MAX
    # Iterate the entire dict to verify OrderedDict internals are intact
    # — a corrupted linked list would manifest as KeyError or wrong size.
    items = list(workspace._workspace_cache.items())
    assert len(items) == len(workspace._workspace_cache)
    for k, v in items:
        assert isinstance(k, str)
        assert isinstance(v, tuple)
    with workspace._workspace_cache_lock:
        workspace._workspace_cache.clear()


# ---------------------------------------------------------------------------
# Defect 4b: workspace cache invalidation hook
# ---------------------------------------------------------------------------

def test_invalidate_workspace_cache_clears_everything_when_path_is_none():
    """``invalidate_workspace_cache(None)`` is the nuclear reset."""
    from agent.lsp import workspace

    workspace._workspace_cache.clear()
    workspace._cache_set("/a", ("/a", True))
    workspace._cache_set("/b", ("/b", True))
    workspace._cache_set("/c", (None, False))
    assert len(workspace._workspace_cache) == 3

    n = workspace.invalidate_workspace_cache(None)

    assert n == 3
    assert len(workspace._workspace_cache) == 0


def test_invalidate_workspace_cache_drops_descendants_after_topology_change():
    """The critical use case: ``/x/.git`` is removed.  Every cached
    descendant that resolved up to ``/x`` is stale.  Invalidating ``/x``
    must purge both the entry keyed at ``/x`` AND the entries keyed at
    deeper paths whose cached root WAS ``/x``."""
    from agent.lsp import workspace

    workspace._workspace_cache.clear()
    # Pre-populate: /x, /x/sub, /x/sub/file.py all resolved to /x
    workspace._cache_set("/x", ("/x", True))
    workspace._cache_set("/x/sub", ("/x", True))
    workspace._cache_set("/x/sub/file.py", ("/x", True))
    # An unrelated entry under /y should survive.
    workspace._cache_set("/y", ("/y", True))
    workspace._cache_set("/y/sub", ("/y", True))

    n = workspace.invalidate_workspace_cache("/x")

    assert n == 3, f"expected 3 invalidations, got {n}"
    assert "/x" not in workspace._workspace_cache
    assert "/x/sub" not in workspace._workspace_cache
    assert "/x/sub/file.py" not in workspace._workspace_cache
    # Unrelated entries untouched.
    assert "/y" in workspace._workspace_cache
    assert "/y/sub" in workspace._workspace_cache

    workspace._workspace_cache.clear()


def test_invalidate_workspace_cache_handles_value_side_match():
    """If ``/x/.git`` is removed but the cache was warmed via a deeper
    walk (e.g. cached at key ``/some/other/symlink-target`` with value
    pointing at ``/x``), the value-side invalidation must catch it.

    Mirrors the symlink-folded case where the cache key and the cached
    root sit in different directory trees.
    """
    from agent.lsp import workspace

    workspace._workspace_cache.clear()
    # Cache key sits outside /x, but the cached *root* is /x.
    workspace._cache_set("/symlinked-elsewhere/file.py", ("/x", True))
    workspace._cache_set("/unrelated/file.py", ("/unrelated", True))

    n = workspace.invalidate_workspace_cache("/x")

    assert n == 1
    assert "/symlinked-elsewhere/file.py" not in workspace._workspace_cache
    assert "/unrelated/file.py" in workspace._workspace_cache

    workspace._workspace_cache.clear()


def test_invalidate_workspace_cache_returns_zero_when_nothing_matches():
    """No-op invalidation returns 0 cleanly; doesn't raise."""
    from agent.lsp import workspace

    workspace._workspace_cache.clear()
    workspace._cache_set("/a", ("/a", True))

    n = workspace.invalidate_workspace_cache("/nonexistent")

    assert n == 0
    assert "/a" in workspace._workspace_cache
    workspace._workspace_cache.clear()


def test_clear_workspace_cache_is_alias_for_full_invalidation():
    """``clear_workspace_cache()`` is the public convenience name."""
    from agent.lsp import workspace

    workspace._workspace_cache.clear()
    workspace._cache_set("/a", ("/a", True))
    workspace._cache_set("/b", ("/b", True))

    n = workspace.clear_workspace_cache()

    assert n == 2
    assert len(workspace._workspace_cache) == 0


def test_invalidate_workspace_cache_prefix_does_not_overmatch():
    """``/x`` invalidation must NOT touch ``/xy/...`` — the prefix check
    needs to respect path separators.  Bug bait: a naive ``startswith``
    would catch ``/xylophone`` when invalidating ``/x``."""
    from agent.lsp import workspace

    workspace._workspace_cache.clear()
    workspace._cache_set("/x", ("/x", True))
    workspace._cache_set("/xylophone", ("/xylophone", True))
    workspace._cache_set("/xy/file.py", ("/xy", True))

    n = workspace.invalidate_workspace_cache("/x")

    assert n == 1, f"expected only /x removed, got {n}"
    assert "/x" not in workspace._workspace_cache
    assert "/xylophone" in workspace._workspace_cache
    assert "/xy/file.py" in workspace._workspace_cache

    workspace._workspace_cache.clear()


# ---------------------------------------------------------------------------
# Defect 5: default config opt-in for installs
# ---------------------------------------------------------------------------

def test_default_config_is_opt_in_for_audit_compliance():
    """Both ``lsp.enabled`` and ``lsp.install_strategy`` default to opt-in:
    LSP starts dormant (no subprocesses, no event loop) and never auto-
    installs anything.  Users explicitly opt in to either by setting
    ``lsp.enabled: true`` and/or ``lsp.install_strategy: auto`` in their
    config.

    Rationale: SOC 2 / ISO 27001 audited environments treat any silent
    network call or background subprocess as a finding.  The feature is
    valuable; making it explicit is cheap.
    """
    from hermes_cli.config import DEFAULT_CONFIG

    lsp = DEFAULT_CONFIG.get("lsp", {})
    assert lsp.get("enabled") is False, (
        f"lsp.enabled must default to False for audit compliance; "
        f"got {lsp.get('enabled')!r}"
    )
    assert lsp.get("install_strategy") == "manual", (
        f"install_strategy must default to 'manual' for audit compliance; "
        f"got {lsp.get('install_strategy')!r}"
    )


def test_manager_create_from_config_defaults_install_manual(monkeypatch):
    """create_from_config fallback also defaults to manual + disabled."""
    from agent.lsp.manager import LSPService

    # Force load_config to return a config WITHOUT lsp keys at all,
    # so we hit the .get(..., default) fallback branches.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {}},
    )
    svc = LSPService.create_from_config()
    try:
        # With enabled=False as the default, create_from_config returns
        # a dormant service (or None — both are acceptable signals that
        # we did not auto-spawn).
        if svc is not None:
            assert svc._install_strategy == "manual"
            assert svc._enabled is False
    finally:
        if svc:
            svc.shutdown()


def test_manager_create_from_config_when_user_opts_in(monkeypatch):
    """When the user explicitly opts in (``lsp.enabled: true``),
    create_from_config returns a live service with manual install."""
    from agent.lsp.manager import LSPService

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": True}},
    )
    svc = LSPService.create_from_config()
    try:
        assert svc is not None
        assert svc._enabled is True
        assert svc._install_strategy == "manual"
    finally:
        if svc:
            svc.shutdown()


def test_create_from_config_with_disabled_lsp_returns_dormant_service(monkeypatch):
    """When lsp.enabled is False, create_from_config still returns a
    service but the background loop and reaper are NOT started.  Catches
    the regression where a disabled config still spawns the asyncio
    thread + reaper coroutine."""
    from agent.lsp.manager import LSPService

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False}},
    )
    svc = LSPService.create_from_config()
    try:
        assert svc is not None
        assert svc._enabled is False
        # When disabled, no reaper handle should have been scheduled.
        assert svc._reaper_handle is None, (
            "disabled LSP still scheduled a reaper coroutine — wasted "
            "background-loop thread"
        )
    finally:
        if svc:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Defect 6: install recipes are pinned
# ---------------------------------------------------------------------------

def test_install_recipes_are_pinned_to_specific_versions():
    """Every auto-install recipe pins a specific version (no @latest,
    no bare names).  Manual-strategy entries are exempt because their
    pkg field is empty by design."""
    from agent.lsp.install import INSTALL_RECIPES

    for name, recipe in INSTALL_RECIPES.items():
        if recipe.get("strategy") == "manual":
            continue
        pkg = recipe["pkg"]
        assert "@" in pkg, f"{name}: pkg {pkg!r} has no version pin"
        assert not pkg.endswith("@latest"), (
            f"{name}: pkg {pkg!r} pins to @latest — must be a specific version"
        )
        version_part = pkg.rsplit("@", 1)[-1]
        assert version_part and version_part[0] in "0123456789v", (
            f"{name}: version part {version_part!r} doesn't look like a version"
        )


# ---------------------------------------------------------------------------
# Defect 7: orphan LSP subprocesses on hermes exit
# ---------------------------------------------------------------------------
#
# Bug shape: spawned language-server subprocesses (pyright, gopls, etc.)
# are started by the background asyncio loop running in a daemon thread.
# When hermes exits, the daemon thread dies but the child processes do NOT
# inherit termination — they live on as orphans, each holding 80-200 MB
# of RAM, until the kernel reaps them.
#
# Fix: ``agent.lsp.get_service`` registers ``shutdown_service`` with
# ``atexit`` on first successful service creation, AND ``cli._run_cleanup``
# calls ``shutdown_service()`` explicitly.  Belt + suspenders.

def test_get_service_registers_atexit_hook_on_first_creation(monkeypatch):
    """First successful service creation registers an atexit hook so
    spawned subprocesses are cleaned up on interpreter exit, even when
    the caller bypasses cli._run_cleanup (scripts, pytest sessions)."""
    import agent.lsp as lsp_mod
    from agent.lsp.manager import LSPService

    # Reset module state to force fresh creation path.
    lsp_mod._service = None
    lsp_mod._atexit_registered = False

    registered = []
    monkeypatch.setattr(
        "agent.lsp.atexit.register",
        lambda fn: registered.append(fn),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": True}},
    )

    svc = lsp_mod.get_service()
    try:
        assert svc is not None, "service should have been created"
        assert lsp_mod._atexit_registered is True
        assert len(registered) == 1, (
            f"expected exactly 1 atexit registration, got {len(registered)}"
        )
        assert registered[0] is lsp_mod.shutdown_service

        # Second call must NOT re-register (idempotent).
        lsp_mod.get_service()
        assert len(registered) == 1, "atexit re-registered on second get_service call"
    finally:
        if svc:
            svc.shutdown()
        lsp_mod._service = None
        lsp_mod._atexit_registered = False


def test_shutdown_service_is_idempotent_and_safe():
    """Both the atexit hook and cli._run_cleanup invoke shutdown_service.
    Calling twice (in any order) must not raise."""
    import agent.lsp as lsp_mod

    lsp_mod._service = None
    # First call when no service exists: no-op.
    lsp_mod.shutdown_service()
    assert lsp_mod._service is None

    # Second call: still no-op.
    lsp_mod.shutdown_service()
    assert lsp_mod._service is None


def test_shutdown_service_swallows_inner_exceptions():
    """If LSPService.shutdown() raises (e.g. event loop already closed),
    the wrapper must swallow it so an exit hook doesn't crash the process."""
    import agent.lsp as lsp_mod

    class BrokenService:
        def shutdown(self):
            raise RuntimeError("simulated shutdown error")

    lsp_mod._service = BrokenService()  # type: ignore[assignment]
    # Must not raise.
    lsp_mod.shutdown_service()
    assert lsp_mod._service is None


def test_cli_run_cleanup_invokes_lsp_shutdown(monkeypatch):
    """The main CLI exit path calls shutdown_service.  Catches the
    regression where cli._run_cleanup is refactored and the LSP block
    is accidentally dropped."""
    import cli as cli_mod
    import agent.lsp as lsp_mod

    called = []
    monkeypatch.setattr(
        "agent.lsp.shutdown_service",
        lambda: called.append("lsp"),
    )
    # Force _cleanup_done flag to False so _run_cleanup actually runs.
    monkeypatch.setattr(cli_mod, "_cleanup_done", False)
    # Stub out the other cleanup helpers so we can isolate the LSP call.
    monkeypatch.setattr(cli_mod, "_cleanup_all_terminals", lambda: None)
    monkeypatch.setattr(cli_mod, "_cleanup_all_browsers", lambda: None)

    try:
        cli_mod._run_cleanup()
    except Exception:
        pass  # other cleanup paths may fail; we only care about LSP

    assert "lsp" in called, (
        "_run_cleanup did not invoke agent.lsp.shutdown_service — "
        "orphan LSP subprocesses will leak on hermes exit"
    )
