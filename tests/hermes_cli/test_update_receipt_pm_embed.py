"""Tests: update receipts embed the pm sync sections (the settled contract).

finalize_update_receipt folds the newest pm sync receipt's
venv_rebuild / plugin_bisect / feature_list into the update receipt
(pm_*-prefixed), so one latest.json carries the whole story.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _seed_sync_receipt(homed, **sections):
    import pm.receipt as pm_receipt_mod

    pm_receipt_mod.begin("sync")
    if "venv_rebuild" in sections:
        pm_receipt_mod.record_venv_rebuild(**sections.pop("venv_rebuild"))
    if "bisect" in sections:
        pm_receipt_mod.record_bisect(sections.pop("bisect"))
    pm_receipt_mod.finalize("ok")


def test_update_receipt_embeds_pm_sections(homed):
    _seed_sync_receipt(
        homed,
        venv_rebuild={"ok": True, "reason": ""},
        bisect=[{"plugin": "bad", "action": "disabled", "reason": "conflict"}],
    )
    ur.begin_update_receipt()
    ur.record_step("git-pull", True)
    path = ur.finalize_update_receipt("success")

    data = json.loads((homed / "logs" / "update_receipts" / "latest.json").read_text(encoding="utf-8"))
    assert data["outcome"] == "success"
    assert data["pm_venv_rebuild"] == {"ok": True, "reason": ""}
    assert data["pm_plugin_bisect"][0]["plugin"] == "bad"
    assert data["pm_sync_outcome"] == "ok"


def test_update_receipt_without_sync_embeds_nothing(homed):
    ur.begin_update_receipt()
    ur.finalize_update_receipt("success")
    data = json.loads((homed / "logs" / "update_receipts" / "latest.json").read_text(encoding="utf-8"))
    assert "pm_venv_rebuild" not in data
    assert data["outcome"] == "success"


def test_embed_failure_never_breaks_the_update_receipt(homed, monkeypatch):
    def boom():
        raise RuntimeError("pm import exploded")

    import pm.receipt as pm_receipt_mod

    monkeypatch.setattr(pm_receipt_mod, "latest", boom)
    ur.begin_update_receipt()
    path = ur.finalize_update_receipt("success")
    assert path is not None  # receipt written despite the embed failure
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["outcome"] == "success"
