"""Tests for gateway.import_sanity — the source-swap canary (#96464).

The canary exists to catch a checkout switched under a running gateway
(update flow racing the launch, a concurrent git operation, a stray test
suite): old modules stay cached in sys.modules while lazy imports of the new
files fail. These tests drive the canary against throwaway modules instead
of the real run_agent/tools chains.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import pytest

from gateway import import_sanity


@pytest.fixture(autouse=True)
def _reset_canary_state(monkeypatch):
    # Isolate every test from the real repo's birth receipt (captured when
    # this test process imported the module); tests that exercise the
    # two-time proof install a controlled receipt explicitly.
    monkeypatch.setattr(import_sanity, "_BIRTH", None)
    import_sanity._snapshots.clear()
    import_sanity._reported.clear()
    yield
    import_sanity._snapshots.clear()
    import_sanity._reported.clear()


def _install_fake_module(monkeypatch, tmp_path, name: str, content: str = "x = 1\n"):
    """Register a module in sys.modules backed by a real file on disk."""
    path = tmp_path / f"{name}.py"
    path.write_text(content)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, mod)
    return path


def _stage_receipt(monkeypatch, name: str, path, commit: str | None = None):
    """Install a birth receipt describing the CURRENT state of ``path``.

    ``commit=None`` keeps the HEAD comparison out of the picture — these
    tests exercise the file-identity half of the proof; the commit half has
    its own test with a controlled ``_read_head_commit``.
    """
    st = os.stat(path)
    monkeypatch.setattr(
        import_sanity,
        "_BIRTH",
        (commit, {name: (str(path), st.st_mtime_ns, st.st_size)}),
    )


# --- The vertical regression the review asked for: a swap that already
# --- happened before the smoke runs must NOT become the healthy baseline.


def test_startup_smoke_detects_file_swapped_before_smoke(monkeypatch, tmp_path, caplog):
    # Module imported from generation A, its backing file swapped to B on
    # disk BEFORE the smoke runs. import_module() returns the cached object
    # and never re-reads disk, so only the birth receipt can catch it —
    # without the receipt comparison the smoke would stat the NEW file and
    # adopt the mixed state as its baseline.
    path = _install_fake_module(
        monkeypatch, tmp_path, "fake_swapped_mod", "x = 1  # generation A\n"
    )
    _stage_receipt(monkeypatch, "fake_swapped_mod", path)
    path.write_text(
        "x = 2222222222  # generation B — swapped under the launching process\n"
    )
    with caplog.at_level(logging.ERROR, logger="gateway.import_sanity"):
        healthy = import_sanity.startup_import_smoke(
            modules=("fake_swapped_mod",), snapshot_root=str(tmp_path)
        )
    assert healthy is False
    assert any("changed on disk during the startup window" in r.message for r in caplog.records)
    # The failure must point the operator at a restart, not at the individual
    # lazy-import sites where the symptom will appear.
    assert any("restart the gateway" in r.message for r in caplog.records)
    # And the NEW generation must not be adopted as the runtime baseline:
    assert "fake_swapped_mod" not in import_sanity._snapshots


def test_startup_smoke_detects_commit_change_during_startup(monkeypatch, tmp_path, caplog):
    _install_fake_module(monkeypatch, tmp_path, "fake_commit_mod")
    monkeypatch.setattr(import_sanity, "_BIRTH", ("aaaaaaaaaaaa", {}))
    monkeypatch.setattr(import_sanity, "_read_head_commit", lambda root: "bbbbbbbbbbbb")
    with caplog.at_level(logging.ERROR, logger="gateway.import_sanity"):
        healthy = import_sanity.startup_import_smoke(
            modules=(), snapshot_root=str(tmp_path)
        )
    assert healthy is False
    assert any("checkout changed during the startup window" in r.message for r in caplog.records)
    assert "fake_commit_mod" not in import_sanity._snapshots


def test_startup_smoke_unreadable_second_head_is_inconclusive_then_recovers(
    monkeypatch, tmp_path, caplog
):
    # Review #5045506662: the birth receipt captured a valid commit A, but
    # the SECOND HEAD read at smoke time fails (ref lock, repack, a checkout
    # mid-transition). A missing second observation is not proof of equality
    # — modules first imported after the receipt have no birth stat, so the
    # HEAD leg is their only first observation. The smoke must be unhealthy
    # and must not seed, without claiming a confirmed swap; a later smoke
    # recovers once the ref is readable again.
    path = _install_fake_module(monkeypatch, tmp_path, "fake_head_unreadable_mod")
    _stage_receipt(monkeypatch, "fake_head_unreadable_mod", path, commit="aaaaaaaaaaaa")
    monkeypatch.setattr(import_sanity, "_read_head_commit", lambda root: None)
    with caplog.at_level(logging.WARNING, logger="gateway.import_sanity"):
        healthy = import_sanity.startup_import_smoke(
            modules=(), snapshot_root=str(tmp_path)
        )
    assert healthy is False
    assert any(
        "cannot verify the checkout generation" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
    # An unreadable ref is inconclusive, not a confirmed swap.
    assert not any(r.levelno == logging.ERROR for r in caplog.records)
    assert import_sanity._snapshots == {}

    # Ref readable again and matching the receipt → a later smoke verifies
    # and seeds normally.
    caplog.clear()
    monkeypatch.setattr(import_sanity, "_read_head_commit", lambda root: "aaaaaaaaaaaa")
    healthy = import_sanity.startup_import_smoke(
        modules=(), snapshot_root=str(tmp_path)
    )
    assert healthy is True
    assert "fake_head_unreadable_mod" in import_sanity._snapshots


def test_startup_smoke_detects_birth_file_vanished(monkeypatch, tmp_path, caplog):
    path = _install_fake_module(monkeypatch, tmp_path, "fake_gone_at_birth_mod")
    _stage_receipt(monkeypatch, "fake_gone_at_birth_mod", path)
    path.unlink()
    with caplog.at_level(logging.ERROR, logger="gateway.import_sanity"):
        healthy = import_sanity.startup_import_smoke(
            modules=(), snapshot_root=str(tmp_path)
        )
    assert healthy is False
    assert any("vanished during the startup window" in r.message for r in caplog.records)


def test_startup_smoke_birth_transient_stat_error_does_not_seed(monkeypatch, tmp_path, caplog):
    # A stat failure during the receipt comparison can't confirm the world,
    # so the smoke must not seed a baseline from an unverified state — but it
    # also must not claim a confirmed swap for a merely invisible file.
    path = _install_fake_module(monkeypatch, tmp_path, "fake_birth_stat_mod")
    _stage_receipt(monkeypatch, "fake_birth_stat_mod", path)

    real_stat = os.stat

    def _raising_stat(p, **kwargs):
        if str(p).endswith("fake_birth_stat_mod.py"):
            raise PermissionError("EACCES (simulated)")
        return real_stat(p, **kwargs)

    monkeypatch.setattr(import_sanity.os, "stat", _raising_stat)
    with caplog.at_level(logging.WARNING, logger="gateway.import_sanity"):
        healthy = import_sanity.startup_import_smoke(
            modules=(), snapshot_root=str(tmp_path)
        )
    assert healthy is False
    assert any("Cannot stat" in r.message for r in caplog.records)
    assert not any(r.levelno == logging.ERROR and "restarted" in r.message for r in caplog.records)
    assert import_sanity._snapshots == {}


def test_startup_smoke_matching_receipt_seeds_snapshots(monkeypatch, tmp_path):
    path = _install_fake_module(monkeypatch, tmp_path, "fake_verified_mod")
    _stage_receipt(monkeypatch, "fake_verified_mod", path)
    healthy = import_sanity.startup_import_smoke(
        modules=("fake_verified_mod",), snapshot_root=str(tmp_path)
    )
    assert healthy is True
    # Passing the proof is what licenses the ongoing baseline:
    assert "fake_verified_mod" in import_sanity._snapshots
    assert import_sanity.verify_runtime_snapshots() is True


def test_startup_smoke_imports_monitored_modules_and_snapshots_repo_tree(monkeypatch, tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    _install_fake_module(monkeypatch, tmp_path, "fake_inside_mod")
    _install_fake_module(monkeypatch, outside, "fake_outside_mod")

    healthy = import_sanity.startup_import_smoke(
        modules=("json",),  # import succeeds but lives outside the snapshot root
        snapshot_root=str(tmp_path),
    )
    assert healthy is True
    # Snapshot sweeps ALL loaded modules under the root — not just the
    # monitored list — so any repo file later touched by a swap is covered.
    assert "fake_inside_mod" in import_sanity._snapshots
    assert "fake_outside_mod" not in import_sanity._snapshots
    assert "json" not in import_sanity._snapshots  # outside the root
    path, mtime_ns, size = import_sanity._snapshots["fake_inside_mod"]
    assert path.startswith(str(tmp_path))
    assert mtime_ns > 0
    assert size > 0


def test_startup_smoke_reports_import_failure(caplog):
    with caplog.at_level(logging.ERROR, logger="gateway.import_sanity"):
        healthy = import_sanity.startup_import_smoke(
            modules=("definitely_not_a_module_xyz",), snapshot_root="/nonexistent-root"
        )
    assert healthy is False
    assert any("definitely_not_a_module_xyz" in r.message for r in caplog.records)
    # The failure message must point the operator at a restart, not at the
    # individual lazy-import sites where the symptom will appear.
    assert any("restarted" in r.message for r in caplog.records)


def test_startup_smoke_never_raises_on_bad_modules():
    # A canary must not crash the startup sequence it guards.
    import_sanity.startup_import_smoke(
        modules=("definitely_not_a_module_xyz",), snapshot_root="/nonexistent-root"
    )


# --- Runtime verification (periodic re-stat of the seeded baseline). ---


def test_verify_clean_while_unchanged(monkeypatch, tmp_path):
    _install_fake_module(monkeypatch, tmp_path, "fake_stable_mod")
    import_sanity.startup_import_smoke(modules=(), snapshot_root=str(tmp_path))
    assert import_sanity.verify_runtime_snapshots() is True


def test_verify_detects_modified_file(monkeypatch, tmp_path, caplog):
    path = _install_fake_module(monkeypatch, tmp_path, "fake_drifted_mod")
    import_sanity.startup_import_smoke(modules=(), snapshot_root=str(tmp_path))
    path.write_text("x = 222  # changed under the running process\n")
    with caplog.at_level(logging.ERROR, logger="gateway.import_sanity"):
        healthy = import_sanity.verify_runtime_snapshots()
    assert healthy is False
    assert any("changed on disk" in r.message for r in caplog.records)
    assert any("restarted" in r.message for r in caplog.records)


def test_verify_reports_drift_once_but_stays_unhealthy(monkeypatch, tmp_path, caplog):
    # The ERROR is logged once, yet the return value must stay False for as
    # long as the drift persists — "already reported" must never read as
    # "recovered" to a caller consuming the bool.
    path = _install_fake_module(monkeypatch, tmp_path, "fake_once_mod")
    import_sanity.startup_import_smoke(modules=(), snapshot_root=str(tmp_path))
    path.write_text("x = 222\n")
    results = []
    with caplog.at_level(logging.ERROR, logger="gateway.import_sanity"):
        for _ in range(3):
            results.append(import_sanity.verify_runtime_snapshots())
    assert results == [False, False, False]
    drift_reports = [r for r in caplog.records if "changed on disk" in r.message]
    assert len(drift_reports) == 1


def test_verify_detects_deleted_file(monkeypatch, tmp_path, caplog):
    path = _install_fake_module(monkeypatch, tmp_path, "fake_gone_mod")
    import_sanity.startup_import_smoke(modules=(), snapshot_root=str(tmp_path))
    path.unlink()
    with caplog.at_level(logging.ERROR, logger="gateway.import_sanity"):
        healthy = import_sanity.verify_runtime_snapshots()
    assert healthy is False
    assert any("disappeared" in r.message for r in caplog.records)


def test_verify_transient_stat_error_warns_without_swap_claim(monkeypatch, tmp_path, caplog):
    # stat failing with a non-ENOENT error (permissions, stale NFS handle)
    # must not be diagnosed as a confirmed checkout swap.
    _install_fake_module(monkeypatch, tmp_path, "fake_stat_mod")
    import_sanity.startup_import_smoke(modules=(), snapshot_root=str(tmp_path))

    real_stat = os.stat

    def _raising_stat(path, **kwargs):
        if str(path).endswith("fake_stat_mod.py"):
            raise PermissionError("EACCES (simulated)")
        return real_stat(path, **kwargs)

    monkeypatch.setattr(import_sanity.os, "stat", _raising_stat)
    with caplog.at_level(logging.WARNING, logger="gateway.import_sanity"):
        healthy = import_sanity.verify_runtime_snapshots()
    assert healthy is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Cannot stat" in r.message for r in warnings)
    # No false "restart required" ERROR for a merely invisible file.
    assert not any("restarted" in r.message for r in caplog.records if r.levelno == logging.ERROR)


def test_verify_noop_without_snapshots():
    # Housekeeping may tick before the startup smoke ran (or after a smoke
    # that recorded nothing); verify must stay quiet and healthy.
    assert import_sanity.verify_runtime_snapshots() is True


# --- The birth receipt's HEAD-commit reader (plain .git file reads). ---


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_read_head_commit_plain_loose_ref(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(tmp_path / ".git" / "refs" / "heads" / "main", "0123456789abcdef\n")
    assert import_sanity._read_head_commit(str(tmp_path)) == "0123456789abcdef"


def test_read_head_commit_detached(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "fedcba9876543210\n")
    assert import_sanity._read_head_commit(str(tmp_path)) == "fedcba9876543210"


def test_read_head_commit_packed_refs(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(
        tmp_path / ".git" / "packed-refs",
        "# pack-refs with: peeled fully-peeled sorted \n"
        "1111111111111111 refs/heads/other\n"
        "0123456789abcdef refs/heads/main\n"
        "9999999999999999 refs/heads/main^{}\n",
    )
    assert import_sanity._read_head_commit(str(tmp_path)) == "0123456789abcdef"


def test_read_head_commit_linked_worktree(tmp_path):
    # A linked worktree's .git is a FILE pointing into the main repo's
    # worktrees dir; its commondir file points back at the shared refs.
    main_git = tmp_path / "main" / ".git"
    wt_git = main_git / "worktrees" / "wt"
    _write(tmp_path / "worktree" / ".git", f"gitdir: {wt_git}\n")
    _write(wt_git / "HEAD", "ref: refs/heads/feat/canary\n")
    _write(wt_git / "commondir", "../..\n")  # back to the main repo's .git
    _write(main_git / "refs" / "heads" / "feat" / "canary", "aaaaaaaaaaaaaaaa\n")
    assert import_sanity._read_head_commit(str(tmp_path / "worktree")) == "aaaaaaaaaaaaaaaa"


def test_read_head_commit_unresolvable_returns_none(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/nope\n")
    assert import_sanity._read_head_commit(str(tmp_path)) is None


def test_read_head_commit_missing_git_dir(tmp_path):
    assert import_sanity._read_head_commit(str(tmp_path)) is None
