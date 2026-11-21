"""Desktop Settings → Model → Auxiliary can set a task's reasoning effort (#89259, salvage #90649).

``POST /api/model/set`` carries ``reasoning_effort`` for an auxiliary task: omitted → the task's
override is left alone; explicit null → cleared (inherit); a level → set. Runtime reads it from
``auxiliary.<task>.reasoning_effort`` (``agent/auxiliary_client.py``).
"""

import pytest
from fastapi import HTTPException

from hermes_cli.web_server_config import _UNSET, _apply_aux_assignment_sync


@pytest.fixture
def saved(monkeypatch):
    store: dict = {}
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: store.update(cfg))
    return store


def test_reasoning_effort_field_semantics_omitted_null_and_level(saved):
    cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": "m1", "reasoning_effort": "low"}}}

    # A plain provider/model re-assignment leaves an existing override alone.
    _apply_aux_assignment_sync(cfg, "openrouter", "m2", "vision", "", "")
    assert cfg["auxiliary"]["vision"] == {"provider": "openrouter", "model": "m2", "reasoning_effort": "low"}

    # A level sets it (canonicalised) and the response echoes it; a disable is stored as the explicit "none".
    out = _apply_aux_assignment_sync(cfg, "openrouter", "m2", "vision", "", "", reasoning_effort="HIGH")
    assert cfg["auxiliary"]["vision"]["reasoning_effort"] == "high" and out["reasoning_effort"] == "high"
    _apply_aux_assignment_sync(cfg, "openrouter", "m2", "vision", "", "", reasoning_effort="disabled")
    assert cfg["auxiliary"]["vision"]["reasoning_effort"] == "none"

    # Explicit null clears only this task's key; siblings and the pick survive.
    cfg["auxiliary"]["compression"] = {"provider": "openrouter", "model": "m3", "reasoning_effort": "max"}
    _apply_aux_assignment_sync(cfg, "openrouter", "m2", "vision", "", "", reasoning_effort=None)
    assert "reasoning_effort" not in cfg["auxiliary"]["vision"]
    assert cfg["auxiliary"]["vision"]["model"] == "m2"
    assert cfg["auxiliary"]["compression"]["reasoning_effort"] == "max"
    assert saved["auxiliary"] == cfg["auxiliary"]


def test_unknown_level_is_rejected_and_reset_clears_overrides(saved):
    cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": "m1"}}}
    with pytest.raises(HTTPException) as exc:
        _apply_aux_assignment_sync(cfg, "openrouter", "m1", "vision", "", "", reasoning_effort="turbo")
    assert exc.value.status_code == 400 and "reasoning_effort" in exc.value.detail
    assert "reasoning_effort" not in cfg["auxiliary"]["vision"]

    cfg["auxiliary"]["vision"]["reasoning_effort"] = "high"
    _apply_aux_assignment_sync(cfg, "", "", "__reset__", "", "", reasoning_effort=_UNSET)
    assert all("reasoning_effort" not in slot for slot in cfg["auxiliary"].values())


def test_bulk_reset_and_bulk_assign_skip_moa_slots(saved):
    """``__reset__`` and assign-all (task="") must never touch the MoA slots (#125435 review).

    Before the exclusion, the widened ``_AUX_TASK_SLOTS`` swept ``auxiliary.moa_reference``:
    a pinned reference provider (e.g. ollama on ``http://box:11434``) was silently rewritten
    to ``provider="auto"`` with ``base_url`` popped and its endpoint credentials cleared —
    collapsing MoA back into N copies of the main model and rerouting the local pin.
    """
    cfg = {"auxiliary": {
        "vision": {"provider": "openrouter", "model": "m1"},
        "moa_reference": {"provider": "ollama", "model": "ref-model",
                          "base_url": "http://box:11434", "api_key": "sk-local"},
        "moa_aggregator": {"provider": "openrouter", "model": "agg-model"},
    }}

    out = _apply_aux_assignment_sync(cfg, "", "", "__reset__", "", "", reasoning_effort=_UNSET)
    assert out["reset"] is True
    assert cfg["auxiliary"]["vision"] == {"provider": "auto", "model": ""}
    # MoA pins survive the bulk reset untouched — provider, model, base_url AND credentials.
    assert cfg["auxiliary"]["moa_reference"] == {
        "provider": "ollama", "model": "ref-model",
        "base_url": "http://box:11434", "api_key": "sk-local",
    }
    assert cfg["auxiliary"]["moa_aggregator"] == {"provider": "openrouter", "model": "agg-model"}

    # Bulk assign-all (task="") sweeps the same non-excluded range: vision moves, MoA stays.
    _apply_aux_assignment_sync(cfg, "nous", "Hermes-4.5", "", "", "")
    assert cfg["auxiliary"]["vision"]["provider"] == "nous"
    assert cfg["auxiliary"]["moa_reference"]["provider"] == "ollama"
    assert cfg["auxiliary"]["moa_aggregator"]["provider"] == "openrouter"

    # Single-slot assignment to a MoA slot is still allowed (exclusion is bulk-only).
    _apply_aux_assignment_sync(cfg, "ollama", "other-ref", "moa_reference", "", "")
    assert cfg["auxiliary"]["moa_reference"]["model"] == "other-ref"
