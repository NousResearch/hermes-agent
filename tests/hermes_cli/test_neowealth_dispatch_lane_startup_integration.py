import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from neoengine_local.dev_lane_heartbeat import DevLaneStore

SCRIPT = Path("/Users/neoengine/.hermes/scripts/neowealth-agent-dispatch-heartbeat.py")


def load_dispatch_module(name="neowealth_dispatch_under_test"):
    if not SCRIPT.exists():
        pytest.skip(f"deployed NeoWealth dispatcher script is not present: {SCRIPT}")
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def packet(lane):
    return {
        "id": "startup-proof",
        "body_of_work": "startup proof",
        "classification": "startup_proof",
        "status": "active",
        "nonclaims": "not merge evidence",
        "lanes": [lane],
    }


def test_neowealth_prompt_renders_lane_string_not_lane_dict(tmp_path):
    module = load_dispatch_module("neowealth_prompt_lane_string_test")
    lane = {"lane": "codex", "provider": "codex", "scope": "startup", "lane_type": "primary_implementation"}

    prompt = module.prompt_for(packet(lane), lane, tmp_path / "wt", "codex/startup-proof", [])

    assert "Dispatcher-enforced heartbeat pickup handshake" in prompt
    assert "<lane_id>` = `{'" not in prompt
    assert "python -m neoengine_local.dev_lane_heartbeat --root ~/.hermes/state/dev-lane-heartbeat pickup" not in prompt


def test_neowealth_unknown_lane_fails_closed_before_worktree_or_spawn(tmp_path, monkeypatch):
    module = load_dispatch_module("neowealth_unknown_lane_preflight_test")
    module.DEV_LANE_HEARTBEAT_ROOT = tmp_path / "heartbeat"
    module.STATE_DIR = tmp_path / "dispatch"
    module.LOG_DIR = module.STATE_DIR / "logs"
    touched = {"worktree": False, "spawn": False}

    def fake_prepare(*args, **kwargs):
        touched["worktree"] = True
        raise AssertionError("prepare_worktree must not run before lane preflight")

    monkeypatch.setattr(module, "prepare_worktree", fake_prepare)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: touched.__setitem__("spawn", True))

    lane = {"lane": "unknown-codex", "provider": "codex", "scope": "startup", "lane_type": "primary_implementation"}

    with pytest.raises(RuntimeError, match="unknown_dev_lane"):
        module.launch_agent(packet(lane), lane)

    assert touched == {"worktree": False, "spawn": False}


def test_neowealth_dispatcher_preflight_runs_before_worktree_and_passes_packet(tmp_path, monkeypatch):
    module = load_dispatch_module("neowealth_central_preflight_test")
    module.DEV_LANE_HEARTBEAT_ROOT = tmp_path / "heartbeat"
    module.STATE_DIR = tmp_path / "dispatch"
    module.LOG_DIR = module.STATE_DIR / "logs"
    module.ensure_default_dev_lane_registrations()
    store = DevLaneStore(module.DEV_LANE_HEARTBEAT_ROOT)
    store.add_work_item({
        "work_item_id": "neowealth:startup-proof:codex",
        "repo_scope": "neowealth",
        "authorized_scopes": ["startup-proof"],
        "status": "queued",
        "operator_priority": 1,
    })

    lane = {"lane": "codex", "provider": "codex", "scope": "startup", "lane_type": "primary_implementation"}
    pkt = packet(lane)
    seen = {"preflight_before_worktree": False, "prompt_packet": None}

    def fake_prepare(packet_arg, lane_arg):
        hb = json.loads((module.DEV_LANE_HEARTBEAT_ROOT / "heartbeats" / "codex.json").read_text())
        assert hb["current_state"] == "working"
        assert hb["claimed_work_item_id"] == "neowealth:startup-proof:codex"
        seen["preflight_before_worktree"] = True
        wt = tmp_path / "wt"
        wt.mkdir()
        return wt, "codex/startup-proof"

    def fake_prompt(packet_arg, lane_arg, wt, branch, peer_lanes, startup_packet=None):
        seen["prompt_packet"] = startup_packet
        return "prompt"

    class FakeProc:
        pid = 12345

    monkeypatch.setattr(module, "prepare_worktree", fake_prepare)
    monkeypatch.setattr(module, "prompt_for", fake_prompt)
    monkeypatch.setattr(module, "agent_env", lambda: dict(os.environ))
    module.provider_canary = SimpleNamespace(canary=lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FakeProc())

    result = module.launch_agent(pkt, lane)

    assert seen["preflight_before_worktree"] is True
    assert seen["prompt_packet"]["action"] == "claimed"
    assert seen["prompt_packet"]["work_item_id"] == "neowealth:startup-proof:codex"
    assert result["lane_control"]["action"] == "claimed"


def test_neowealth_codex_and_sonnet_registered_and_visible_in_status(tmp_path):
    module = load_dispatch_module("neowealth_status_registration_test")
    module.DEV_LANE_HEARTBEAT_ROOT = tmp_path / "heartbeat"
    module.ensure_default_dev_lane_registrations()
    store = DevLaneStore(module.DEV_LANE_HEARTBEAT_ROOT)

    report = store.status_report(now="2026-06-24T00:00:00Z")

    assert {row["lane_id"] for row in report["lanes"]} >= {"codex", "nw-sonnet-01-fin-mvp-integration-recovery"}


def test_neowealth_duplicate_expected_claim_fails_closed_before_spawn(tmp_path, monkeypatch):
    module = load_dispatch_module("neowealth_duplicate_claim_preflight_test")
    module.DEV_LANE_HEARTBEAT_ROOT = tmp_path / "heartbeat"
    module.STATE_DIR = tmp_path / "dispatch"
    module.LOG_DIR = module.STATE_DIR / "logs"
    module.ensure_default_dev_lane_registrations()
    store = DevLaneStore(module.DEV_LANE_HEARTBEAT_ROOT)
    work_item_id = "neowealth:startup-proof:codex"
    now = "2026-06-24T00:00:00Z"
    store.add_work_item({"work_item_id": work_item_id, "repo_scope": "neowealth", "authorized_scopes": ["**"], "status": "queued"})
    store.claim_work(work_item_id, "codex", "other-live-session", now, 999999, "claims/other.json")
    touched = {"worktree": False, "spawn": False}
    monkeypatch.setattr(module, "prepare_worktree", lambda *args, **kwargs: touched.__setitem__("worktree", True))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: touched.__setitem__("spawn", True))
    lane = {"lane": "codex", "provider": "codex", "scope": "startup", "lane_type": "primary_implementation"}

    with pytest.raises(RuntimeError, match="active_claim_exists"):
        module.launch_agent(packet(lane), lane)

    assert touched == {"worktree": False, "spawn": False}


def test_neowealth_protected_item_does_not_spawn_provider(tmp_path, monkeypatch):
    module = load_dispatch_module("neowealth_protected_preflight_test")
    module.DEV_LANE_HEARTBEAT_ROOT = tmp_path / "heartbeat"
    module.STATE_DIR = tmp_path / "dispatch"
    module.LOG_DIR = module.STATE_DIR / "logs"
    module.ensure_default_dev_lane_registrations()
    store = DevLaneStore(module.DEV_LANE_HEARTBEAT_ROOT)
    store.add_work_item({
        "work_item_id": "neowealth:startup-proof:codex",
        "repo_scope": "neowealth",
        "authorized_scopes": ["**"],
        "status": "queued",
        "labels": ["do-not-merge"],
    })
    touched = {"worktree": False, "spawn": False}
    monkeypatch.setattr(module, "prepare_worktree", lambda *args, **kwargs: touched.__setitem__("worktree", True))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: touched.__setitem__("spawn", True))
    lane = {"lane": "codex", "provider": "codex", "scope": "startup", "lane_type": "primary_implementation"}

    result = module.launch_agent(packet(lane), lane)

    assert result["spawned"] is False
    assert result["lane_control"]["action"] == "idle-no-work"
    assert touched == {"worktree": False, "spawn": False}
