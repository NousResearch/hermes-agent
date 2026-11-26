"""Doctor's PM-store check reports reclaimable debris and orphan install states, and skips a
store another PM operation holds (``hermes_cli/doctor_tools.py::_check_pm_store_reclaim``)."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

from pm.lock import Facts
from pm.store import Store


def _empty_plan():
    return ({"store_entries": [], "staging": [], "set_aside": [], "partials": []}, 2)


def _run_check(monkeypatch, tmp_path, plan, orphans=()):
    """Run the check against a patched plan; returns (finding, stdout)."""
    import pm.cli
    import pm.install
    import pm.install_states
    from hermes_cli import doctor_tools
    from pm import paths

    store_dir = tmp_path / "tools"
    store_dir.mkdir()
    monkeypatch.setattr(paths, "writable_store_root", lambda: store_dir)
    monkeypatch.setattr(pm.install, "_store", lambda: Store(store_dir))
    monkeypatch.setattr(pm.install, "_facts", lambda: Facts(store_dir / "facts.json"))
    if callable(plan):
        monkeypatch.setattr(pm.cli, "_gc_store_plan", plan)
    else:
        monkeypatch.setattr(pm.cli, "_gc_store_plan", lambda store, facts, *, lock_wait=True: plan)
    monkeypatch.setattr(pm.install_states, "orphan_install_states", lambda installs: list(orphans))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finding = doctor_tools._check_pm_store_reclaim(False)
    return finding, out.getvalue()


def test_clean_store_is_an_ok_row(tmp_path, monkeypatch):
    finding, out = _run_check(monkeypatch, tmp_path, _empty_plan())
    assert "✓" in out and "clean" in out
    assert "reclaimable debris" not in out
    assert finding.issues == []


def test_large_debris_warns_with_the_reclaim_remedy(tmp_path, monkeypatch):
    plan = ({"store_entries": [{"name": "fetch-old", "bytes": 256 << 20}],
             "staging": [], "set_aside": [], "partials": []}, 3)
    _, out = _run_check(monkeypatch, tmp_path, plan)
    assert "⚠" in out and "holds reclaimable debris" in out
    assert "hermes pm gc --dry-run" in out


def test_minor_turnover_stays_an_ok_row(tmp_path, monkeypatch):
    plan = ({"store_entries": [{"name": "fetch-old", "bytes": 8 << 20}],
             "staging": [], "set_aside": [], "partials": []}, 3)
    _, out = _run_check(monkeypatch, tmp_path, plan)
    assert "✓" in out and "minor turnover" in out
    assert "⚠" not in out


def test_any_orphan_install_state_warns_even_when_the_store_is_small(tmp_path, monkeypatch):
    orphan = tmp_path / "installs" / "deadbeef"
    _, out = _run_check(monkeypatch, tmp_path, _empty_plan(), orphans=[orphan])
    assert "⚠" in out and "1 install state dir(s) of deleted checkouts" in out


def test_busy_store_is_skipped_not_failed(tmp_path, monkeypatch):
    _, out = _run_check(monkeypatch, tmp_path, None)
    assert "PM store is busy" in out
    assert "⚠" not in out


def test_a_failing_probe_degrades_to_a_warn_row(tmp_path, monkeypatch):
    def _boom(store, facts, *, lock_wait=True):
        raise RuntimeError("disk gone")

    _, out = _run_check(monkeypatch, tmp_path, _boom)
    assert "could not check" in out
