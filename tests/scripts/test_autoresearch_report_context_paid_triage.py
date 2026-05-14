from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path.home() / ".hermes" / "scripts" / "autoresearch_report_context.py"


@pytest.fixture()
def autoresearch_module():
    spec = importlib.util.spec_from_file_location("autoresearch_report_context", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ready_paid_triage_candidate(module, now):
    handoff_fields = {
        "target_surface": "assets/bid_readiness_launch_pack/asset_manifest.json",
        "audience": "Paid bid-readiness triage prospects",
        "cta": "Send the paid triage proof pack to the prospect list",
        "owner": "The One",
        "launch_window_et": "2026-05-09 after morning report delivery",
        "primary_metric": "Paid triage replies booked",
        "metric_source": "tracking_sheet.csv",
        "success_threshold": "2 qualified replies",
        "fail_threshold": "0 qualified replies",
        "check_in_time_et": "2026-05-09T12:00:00-04:00",
    }
    candidate = {
        "source": "ledger",
        "source_session_date": "2026-05-08",
        "slice_index": 6,
        "focus": "Paid bid readiness triage launch asset pack",
        "recommendation": "Launch the paid first-pass bid/no-bid triage offer to qualified prospects.",
        "next_action": "Use the asset pack, prospects, send queue, reply routing, tracking sheet, outreach sequence, and outcome taxonomy.",
        "confidence": 95,
        "timestamp_et": "2026-05-08T23:30:00-04:00",
        "handoff_fields": handoff_fields,
    }
    enriched = module.enrich_candidate(candidate, now, "2026-05-08")
    assert enriched["ready_for_launch"] is True
    return enriched


def test_paid_triage_missing_assets_block_launch_contract_id(autoresearch_module, tmp_path):
    now = autoresearch_module.TZ.fromutc(__import__("datetime").datetime(2026, 5, 9, 10, 0, tzinfo=autoresearch_module.TZ))
    run_dir = tmp_path / "2026-05-08"
    report_path = run_dir / "morning_report.md"
    report_path.parent.mkdir(parents=True)
    selected = _ready_paid_triage_candidate(autoresearch_module, now)

    packet = autoresearch_module.build_candidate_packet(selected, now, report_path)

    assert packet["ready_for_launch"] is False
    assert packet["handoff_ready"] is False
    assert packet["blocker_state"] == "paid_triage_assets_missing"
    assert packet["launch_contract_id"] is None
    assert packet["paid_triage_asset_pack"]["applies"] is True
    assert set(packet["paid_triage_asset_pack"]["missing_files"]) == set(autoresearch_module.PAID_TRIAGE_REQUIRED_FILES)


def test_paid_triage_complete_assets_allow_launch_contract_id(autoresearch_module, tmp_path):
    now = autoresearch_module.TZ.fromutc(__import__("datetime").datetime(2026, 5, 9, 10, 0, tzinfo=autoresearch_module.TZ))
    run_dir = tmp_path / "2026-05-08"
    report_path = run_dir / "morning_report.md"
    asset_dir = run_dir / autoresearch_module.PAID_TRIAGE_ASSET_PACK_DIR
    asset_dir.mkdir(parents=True)
    for name in autoresearch_module.PAID_TRIAGE_REQUIRED_FILES:
        (asset_dir / name).write_text("ok\n")
    selected = _ready_paid_triage_candidate(autoresearch_module, now)

    packet = autoresearch_module.build_candidate_packet(selected, now, report_path)

    assert packet["ready_for_launch"] is True
    assert packet["blocker_state"] == "ready"
    assert packet["paid_triage_asset_pack"]["missing_files"] == []
    assert packet["launch_contract_id"]


def test_2026_05_09_040056_replay_blocks_paid_triage_contract_without_assets(autoresearch_module, tmp_path, monkeypatch, capsys):
    root = tmp_path / "autoresearch"
    run_dir = root / "2026-05-08"
    run_dir.mkdir(parents=True)
    handoff_fields = {
        "target_surface": "assets/bid_readiness_launch_pack/asset_manifest.json",
        "audience": "Paid bid-readiness triage prospects",
        "cta": "Send the paid triage proof pack to qualified prospects",
        "owner": "The One",
        "launch_window_et": "2026-05-09 after morning report delivery",
        "primary_metric": "Qualified paid-triage replies",
        "metric_source": "tracking_sheet.csv",
        "success_threshold": "2 qualified replies",
        "fail_threshold": "0 qualified replies",
        "check_in_time_et": "2026-05-09T12:00:00-04:00",
    }
    entry = {
        "slice_index": 6,
        "decision": "keep",
        "focus": "Paid bid readiness triage launch asset pack",
        "recommendation": "Launch the paid first-pass bid/no-bid triage offer.",
        "next_action": "Use asset_manifest.json, sample output, snippets, prospects, send queue, reply routing, tracking sheet, outreach sequence, and outcome taxonomy.",
        "confidence": 95,
        "timestamp_et": "2026-05-08T23:30:00-04:00",
        "handoff_fields": handoff_fields,
    }
    (run_dir / "ledger.jsonl").write_text(json.dumps(entry) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({
        "expected_slice_total": 6,
        "top_candidates": [{
            "slice_index": 6,
            "focus": entry["focus"],
            "recommendation": entry["recommendation"],
            "confidence": 95,
            "timestamp_et": entry["timestamp_et"],
        }],
    }))
    (run_dir / "status.json").write_text(json.dumps({"state": "compiled", "delivered": False}))
    (run_dir / "notes.md").write_text("notes\n")
    (root / "carry_forward.json").write_text("{}")
    (root / "active_experiment.json").write_text("{}")

    monkeypatch.setattr(autoresearch_module, "ROOT", root)
    monkeypatch.setattr(autoresearch_module, "GLOBAL_CARRY_FORWARD_PATH", root / "carry_forward.json")
    monkeypatch.setattr(autoresearch_module, "GLOBAL_ACTIVE_EXPERIMENT_PATH", root / "active_experiment.json")
    monkeypatch.setenv("AUTORESEARCH_NOW_ET", "2026-05-09T04:00:56-04:00")

    autoresearch_module.main()
    payload = json.loads(capsys.readouterr().out)
    packet = json.loads((run_dir / "candidate_packet.json").read_text())
    report_inputs = json.loads((run_dir / "report_inputs.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())

    assert payload["session_date"] == "2026-05-08"
    assert packet["blocker_state"] == "paid_triage_assets_missing"
    assert packet["ready_for_launch"] is False
    assert packet["launch_contract_id"] is None
    assert report_inputs["launch_contract_id"] is None
    assert status["launch_contract_id"] is None
    assert "paid_triage_assets_missing" in report_inputs["report_block_reason"]


def test_2026_05_09_040056_delivered_replay_clears_stale_paid_triage_contract(autoresearch_module, tmp_path, monkeypatch, capsys):
    root = tmp_path / "autoresearch"
    run_dir = root / "2026-05-08"
    run_dir.mkdir(parents=True)
    candidate_id = "2026-05-08-s6-asset-existence-compile-gate-stop-the-proof-pack-recommendation-"
    stale_contract_id = f"{candidate_id}:stale12345678"
    packet = {
        "candidate_id": candidate_id,
        "source_focus": "Paid-triage asset-existence compile gate for the bid-readiness proof pack",
        "target_surface": "assets/bid_readiness_launch_pack/asset_manifest.json",
        "audience": "Paid bid-readiness triage prospects",
        "cta": "Send the paid triage proof pack",
        "owner": "The One",
        "launch_window_et": "2026-05-09 after morning report delivery",
        "measurement_contract": {"primary_metric": "qualified replies"},
        "handoff_ready": True,
        "ready_for_launch": True,
        "launch_contract_id": stale_contract_id,
        "generated_at_et": "2026-05-09T04:00:56-04:00",
    }
    report_inputs = {
        "handoff_ready": True,
        "report_ready": True,
        "render_verification_state": "verified",
        "launch_contract_id": stale_contract_id,
        "promoted_candidate_id": candidate_id,
        "delivery_state": "delivered_with_launch_brief",
        "generated_at_et": "2026-05-09T04:00:56-04:00",
    }
    status = dict(report_inputs, delivered=True, state="delivered")
    (run_dir / "candidate_packet.json").write_text(json.dumps(packet))
    (run_dir / "report_inputs.json").write_text(json.dumps(report_inputs))
    (run_dir / "status.json").write_text(json.dumps(status))
    (run_dir / "promotion_decision.json").write_text(json.dumps({"selected_candidate_id": candidate_id}))
    (run_dir / "summary.json").write_text(json.dumps({"expected_slice_total": 6}))
    (run_dir / "ledger.jsonl").write_text("")
    (run_dir / "notes.md").write_text("notes\n")
    (run_dir / "morning_report.md").write_text(
        f"🚀 Launch Brief: READY\n🧷 Promoted Candidate ID: {candidate_id}\n🧷 Launch Contract ID: {stale_contract_id}\n"
    )
    (root / "carry_forward.json").write_text("{}")
    (root / "active_experiment.json").write_text("{}")

    monkeypatch.setattr(autoresearch_module, "ROOT", root)
    monkeypatch.setattr(autoresearch_module, "GLOBAL_CARRY_FORWARD_PATH", root / "carry_forward.json")
    monkeypatch.setattr(autoresearch_module, "GLOBAL_ACTIVE_EXPERIMENT_PATH", root / "active_experiment.json")
    monkeypatch.setenv("AUTORESEARCH_NOW_ET", "2026-05-09T04:00:56-04:00")

    autoresearch_module.main()
    payload = json.loads(capsys.readouterr().out)
    packet = json.loads((run_dir / "candidate_packet.json").read_text())
    report_inputs = json.loads((run_dir / "report_inputs.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())

    assert payload["delivered_replay_preserved"] is True
    assert packet["launch_contract_id"] is None
    assert report_inputs["launch_contract_id"] is None
    assert status["launch_contract_id"] is None
    assert packet["blocker_state"] == "paid_triage_assets_missing"
    assert report_inputs["report_ready"] is False
    assert status["delivery_state"] == "blocked_brief_delivered"
