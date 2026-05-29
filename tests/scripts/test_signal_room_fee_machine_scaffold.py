from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_fee_machine_scaffold.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_fee_machine_scaffold", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_rig_dir(tmp_path: Path) -> Path:
    rig = tmp_path / "rig"
    poses = rig / "poses"
    poses.mkdir(parents=True)
    for pose in [
        "neutral_read",
        "bill_shock",
        "look_to_machine",
        "skeptical_point",
        "slight_lean",
    ]:
        (poses / f"{pose}.svg").write_text(f"<svg><title>{pose}</title></svg>")
    (rig / "rig_manifest.json").write_text(json.dumps({"rig": "test", "status": "review-only"}))
    return rig


def test_scaffold_writes_review_project_with_required_assets(tmp_path: Path) -> None:
    module = load_module()
    rig = make_rig_dir(tmp_path)
    out = tmp_path / "scaffold"

    manifest = module.create_scaffold(rig, out)

    assert manifest["status"] == "review-only"
    assert "run scripts/signal_room_video_env_gate.py before local render attempts" in manifest["required_review"]
    assert (out / "index.html").exists()
    assert (out / "DESIGN.md").exists()
    assert (out / "audio_cue_sheet.json").exists()
    assert (out / "motion_primitives.json").exists()
    assert (out / "motion_primitives.js").exists()
    assert (out / "scene_choreography.json").exists()
    assert (out / "retention_frame_plan.json").exists()
    assert (out / ".hyperframes" / "expanded-prompt.md").exists()
    assert (out / "assets" / "poses" / "skeptical_point.svg").exists()
    audio = json.loads((out / "audio_cue_sheet.json").read_text())
    assert audio["status"] == "review-only"
    assert audio["duration_seconds"] == 15
    assert audio["mix_targets"]["phone_speaker"] is True
    assert [cue["id"] for cue in audio["cues"]] == [
        "room_tone_bed",
        "paper_bill_snap",
        "fee_tick_base",
        "fee_tick_service",
        "fee_tick_processing",
        "fee_tick_platform",
        "wall_reveal_thump",
        "machine_drive_loop",
        "pointing_accent",
        "final_lever_click",
        "memory_hold_tail",
    ]
    frame_plan = json.loads((out / "retention_frame_plan.json").read_text())
    assert frame_plan["status"] == "review-only"
    assert frame_plan["contact_sheet_required"] is True
    assert [frame["id"] for frame in frame_plan["frames"]] == [
        "ordinary_bill",
        "number_split",
        "machine_reveal",
        "acting_read",
        "memory_anchor",
    ]
    assert [frame["sample_time"] for frame in frame_plan["frames"]] == [1.25, 3.75, 6.5, 9.25, 13.2]
    assert "five similar cards" in frame_plan["failure_conditions"]
    primitives = json.loads((out / "motion_primitives.json").read_text())
    assert primitives["status"] == "review-only"
    assert [primitive["id"] for primitive in primitives["primitives"]] == [
        "continuous_drift",
        "ambient_loop",
        "character_micro_acting",
        "pose_transition",
        "fee_stack_simmer",
        "machine_idle",
        "machine_causality",
    ]
    choreography = json.loads((out / "scene_choreography.json").read_text())
    assert choreography["status"] == "review-only"
    assert choreography["composition_id"] == "fee-machine-v2-review"
    assert [beat["id"] for beat in choreography["beats"]] == [
        "ordinary_bill_hold",
        "fee_split_reaction",
        "machine_reveal_handoff",
        "skeptical_point_drive",
        "memory_anchor_settle",
    ]
    assert choreography["beats"][2]["acting_objective"] == "look from bill to the revealed machine"
    assert choreography["beats"][3]["primary_motion"] == "pointing pose locks machine cause to fee stack effect"
    assert choreography["beats"][1]["acting_phases"] == ["anticipation", "reaction_hold", "settle"]
    assert choreography["beats"][3]["acting_phases"] == ["anticipation", "point_hold", "settle"]
    assert all(beat["overlap_in_seconds"] >= 0.12 for beat in choreography["beats"][1:])
    html = (out / "index.html").read_text()
    library = (out / "motion_primitives.js").read_text()
    assert "data-composition-id=\"fee-machine-v2-review\"" in html
    assert "motion_primitives.js" in html
    assert "hold -> split -> reveal -> mechanical pulse -> acting beat -> settle" in html
    assert "assets/poses/neutral_read.svg" in html
    assert "service fee" in html
    assert "motion quality primitives" in html
    assert "SignalRoomMotion.continuousDrift(tl)" in html
    assert "SignalRoomMotion.ambientLoop(tl)" in html
    assert "SignalRoomMotion.microActing(tl" in html
    assert "SignalRoomMotion.poseTransition(tl" in html
    assert "SignalRoomMotion.feeStackSimmer(tl" in html
    assert "SignalRoomMotion.machineIdle(tl" in html
    assert "SignalRoomMotion.machineCausality(tl" in html
    assert "window.SignalRoomMotion" in library
    assert "continuousDrift(tl)" in library
    assert "ambientLoop(tl)" in library
    assert "microActing(tl, selector, start, duration, amount)" in library
    assert "poseTransition(tl, outgoingSelector, incomingSelector, start, options)" in library
    assert "machineCausality(tl, start, options)" in library
    assert ".lever" in library
    assert ".gear-a" in library
    assert ".fee-tag" in library


def test_scaffold_rejects_missing_required_pose(tmp_path: Path) -> None:
    module = load_module()
    rig = make_rig_dir(tmp_path)
    (rig / "poses" / "look_to_machine.svg").unlink()

    try:
        module.create_scaffold(rig, tmp_path / "out")
    except FileNotFoundError as exc:
        assert "look_to_machine.svg" in str(exc)
    else:
        raise AssertionError("expected missing pose to fail")
