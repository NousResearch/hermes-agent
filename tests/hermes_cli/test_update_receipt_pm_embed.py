"""Tests: update receipts embed ONLY their own pm sync sections.

The updater embeds the sync receipt that carries ITS correlation id —
read from pm.receipt's per-context completions, never from latest.json.
A sync that finished before the update began, or one running in a
concurrent thread, must never be misattributed to the invoking update.
"""

from __future__ import annotations

import json
import threading

import pytest

import hermes_cli.update_receipt as ur


@pytest.fixture
def homed(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    import pm.receipt as pm_receipt_mod

    monkeypatch.setattr(pm_receipt_mod, "_receipt_dir", lambda: tmp_path / "logs" / "update_receipts")
    monkeypatch.setattr(ur, "_receipt_dir", lambda: tmp_path / "logs" / "update_receipts")
    return tmp_path


@pytest.fixture(autouse=True)
def _isolated_receipt_state():
    """No leaked in-flight receipts, completion maps, or correlation
    between tests."""
    import pm.receipt as pm_receipt_mod

    pm_receipt_mod._current.set(None)
    pm_receipt_mod._completed_by_update.set(None)
    ur._current.set(None)
    yield
    pm_receipt_mod._current.set(None)
    pm_receipt_mod._completed_by_update.set(None)
    ur._current.set(None)


def _sync_under_update(**sections):
    """A pm sync that runs WHILE the update receipt is open — the real
    in-process update flow (sync_venv)."""
    import pm.receipt as pm_receipt_mod

    pm_receipt_mod.begin("sync")
    if "venv_rebuild" in sections:
        pm_receipt_mod.record_venv_rebuild(**sections.pop("venv_rebuild"))
    if "features" in sections:
        pm_receipt_mod.record_feature_list(sections.pop("features"))
    if "warnings" in sections:
        for message in sections.pop("warnings"):
            pm_receipt_mod.record_warning(message)
    if "refusal" in sections:
        pm_receipt_mod.record_refusal(*sections.pop("refusal"))
    pm_receipt_mod.finalize("ok")


def test_update_receipt_embeds_pm_sections(homed):
    ur.begin_update_receipt()
    _sync_under_update(
        venv_rebuild={"ok": True, "reason": ""},
        features=["web", "acp"],
    )
    ur.record_step("git-pull", True)
    path = ur.finalize_update_receipt("success")

    data = json.loads((homed / "logs" / "update_receipts" / "latest.json").read_text(encoding="utf-8"))
    assert data["outcome"] == "success"
    assert data["pm_venv_rebuild"] == {"ok": True, "reason": ""}
    assert data["pm_feature_list"] == ["web", "acp"]
    assert data["pm_sync_outcome"] == "ok"
    assert data["update_id"]


def test_legacy_worker_receipt_fields_still_embed(homed):
    from pm.receipt import accept_worker_receipt

    ur.begin_update_receipt()
    update_id = ur.current_correlation_id()
    legacy = {"update_id": update_id, "outcome": "ok",
              "plugin_bisect": [{"plugin": "old-plugin", "action": "disabled"}]}
    accept_worker_receipt(legacy, update_id)
    path = ur.finalize_update_receipt("success")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pm_plugin_bisect"] == legacy["plugin_bisect"]


def test_update_receipt_without_sync_embeds_nothing(homed):
    ur.begin_update_receipt()
    ur.finalize_update_receipt("success")
    data = json.loads((homed / "logs" / "update_receipts" / "latest.json").read_text(encoding="utf-8"))
    assert "pm_venv_rebuild" not in data
    assert data["outcome"] == "success"


def test_stale_sync_from_before_the_update_is_never_embedded(homed):
    """The audit defect: an old sync receipt on disk (latest.json's newest)
    must not ride along on an update that caused no sync of its own."""
    import pm.receipt as pm_receipt_mod

    pm_receipt_mod.begin("sync")
    pm_receipt_mod.record_venv_rebuild(True, "stale run")
    pm_receipt_mod.finalize("ok")
    assert pm_receipt_mod.latest()["venv_rebuild"]["reason"] == "stale run"

    ur.begin_update_receipt()
    path = ur.finalize_update_receipt("success")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "pm_venv_rebuild" not in data
    assert "pm_sync_outcome" not in data


def test_concurrent_thread_sync_is_never_embedded(homed):
    """A sync finalizing in another thread (cadence plugin-check, parallel
    ensure) finalizes into ITS context — the invoking update must not
    embed it, even though it hit latest.json first."""
    import pm.receipt as pm_receipt_mod

    ur.begin_update_receipt()
    seen: dict = {}

    def concurrent_sync():
        pm_receipt_mod.begin("sync")
        pm_receipt_mod.record_venv_rebuild(True, "not my update")
        seen["path"] = pm_receipt_mod.finalize("ok")

    t = threading.Thread(target=concurrent_sync)
    t.start()
    t.join()
    assert seen["path"] is not None  # it did write a receipt...

    path = ur.finalize_update_receipt("success")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "pm_venv_rebuild" not in data  # ...but not into THIS update


def test_nested_update_ids_do_not_cross_embed(homed):
    """The steering-required scenario: outer update → outer sync → nested
    update → nested sync → finalize inner → finalize outer. BOTH update
    receipts finalize with a path, and each embeds ONLY its own sync —
    the nested sync never displaces the outer update's."""
    import pm.receipt as pm_receipt_mod

    ur.begin_update_receipt()
    outer_id = ur.current_correlation_id()
    pm_receipt_mod.begin("sync")
    pm_receipt_mod.record_venv_rebuild(True, "outer rebuild")
    pm_receipt_mod.finalize("ok")

    ur.begin_update_receipt()  # nested update — new id, outer preserved behind the token
    inner_id = ur.current_correlation_id()
    assert inner_id != outer_id
    pm_receipt_mod.begin("sync")
    pm_receipt_mod.record_venv_rebuild(True, "inner rebuild")
    pm_receipt_mod.finalize("ok")

    inner_path = ur.finalize_update_receipt("success")
    assert inner_path is not None
    assert json.loads(inner_path.read_text(encoding="utf-8"))["pm_venv_rebuild"]["reason"] == "inner rebuild"

    # outer finalize: the outer sync is STILL its own completion — the
    # nested update's finalize restored the outer receipt and must not
    # have dropped it or let the inner sync ride along.
    outer_path = ur.finalize_update_receipt("partial")
    assert outer_path is not None
    outer_data = json.loads(outer_path.read_text(encoding="utf-8"))
    assert outer_data["update_id"] == outer_id
    assert outer_data["pm_venv_rebuild"]["reason"] == "outer rebuild"
    assert "inner rebuild" not in json.dumps(outer_data)

    # outermost finalize popped the last receipt: no correlation remains
    assert ur.current_correlation_id() is None


def test_both_nested_receipts_still_read_correctly_after_all_finalizations(homed):
    """Invariant: inner and outer receipts have UNIQUE file names (same
    process, possibly the same second) and each exact path still reads its
    own story AFTER every finalize — no overwrite, no cross-contamination,
    including the latest.json pointer (outer finished last)."""
    import pm.receipt as pm_receipt_mod

    ur.begin_update_receipt()
    pm_receipt_mod.begin("sync")
    pm_receipt_mod.record_venv_rebuild(True, "outer rebuild")
    pm_receipt_mod.finalize("ok")

    ur.begin_update_receipt()
    pm_receipt_mod.begin("sync")
    pm_receipt_mod.record_venv_rebuild(True, "inner rebuild")
    pm_receipt_mod.finalize("ok")

    inner_path = ur.finalize_update_receipt("success")
    outer_path = ur.finalize_update_receipt("success")

    assert inner_path != outer_path
    assert inner_path.is_file() and outer_path.is_file()
    inner_data = json.loads(inner_path.read_text(encoding="utf-8"))
    outer_data = json.loads(outer_path.read_text(encoding="utf-8"))
    assert inner_data["pm_venv_rebuild"]["reason"] == "inner rebuild"
    assert outer_data["pm_venv_rebuild"]["reason"] == "outer rebuild"
    # latest.json points at the LAST finalization (the outer run)
    latest = json.loads((homed / "logs" / "update_receipts" / "latest.json").read_text(encoding="utf-8"))
    assert latest["update_id"] == outer_data["update_id"]
    assert latest["pm_venv_rebuild"]["reason"] == "outer rebuild"


def test_sync_after_nested_update_finalizes_embeds_into_outer(homed):
    """Continuing the outer update after the nested one closed: a sync
    begun under the restored outer id embeds into the outer receipt."""
    import pm.receipt as pm_receipt_mod

    ur.begin_update_receipt()
    ur.begin_update_receipt()
    ur.finalize_update_receipt("success")
    pm_receipt_mod.begin("sync")
    pm_receipt_mod.record_venv_rebuild(False, "already in sync")
    pm_receipt_mod.finalize("ok")
    path = ur.finalize_update_receipt("success")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pm_venv_rebuild"] == {"ok": False, "reason": "already in sync"}


def test_pm_warnings_and_refusal_embed(homed):
    ur.begin_update_receipt()
    _sync_under_update(
        warnings=["shim quarantine skipped: file locked"],
        refusal=("lazy-install", "extras outside the frozen feature set"),
    )
    path = ur.finalize_update_receipt("success")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pm_warnings"][0]["message"] == "shim quarantine skipped: file locked"
    assert data["pm_refusal"]["code"] == "lazy-install"
    assert data["pm_refusal"]["detail"] == "extras outside the frozen feature set"


def test_embed_failure_never_breaks_the_update_receipt(homed, monkeypatch):
    def boom():
        raise RuntimeError("pm import exploded")

    import pm.receipt as pm_receipt_mod

    monkeypatch.setattr(pm_receipt_mod, "last_for_update", boom)
    ur.begin_update_receipt()
    path = ur.finalize_update_receipt("success")
    assert path is not None  # receipt written despite the embed failure
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["outcome"] == "success"
