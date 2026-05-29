from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_scaffold_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_scaffold_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def required_motion_primitives() -> list[dict[str, object]]:
    return [
        {
            "id": "continuous_drift",
            "timeline_marker": "SignalRoomMotion.continuousDrift(",
            "start": 0.0,
            "end": 15.0,
        },
        {
            "id": "ambient_loop",
            "timeline_marker": "SignalRoomMotion.ambientLoop(",
            "start": 0.0,
            "end": 15.0,
        },
        {
            "id": "character_micro_acting",
            "timeline_marker": "SignalRoomMotion.microActing(",
            "start": 0.2,
            "end": 14.7,
        },
        {
            "id": "pose_transition",
            "timeline_marker": "SignalRoomMotion.poseTransition(",
            "start": 2.35,
            "end": 12.15,
        },
        {
            "id": "fee_stack_simmer",
            "timeline_marker": "SignalRoomMotion.feeStackSimmer(",
            "start": 3.0,
            "end": 14.5,
        },
        {
            "id": "machine_idle",
            "timeline_marker": "SignalRoomMotion.machineIdle(",
            "start": 5.8,
            "end": 14.7,
        },
        {
            "id": "machine_causality",
            "timeline_marker": "SignalRoomMotion.machineCausality(",
            "start": 6.05,
            "end": 13.3,
        },
    ]


def write_scaffold(root: Path) -> None:
    (root / ".hyperframes").mkdir(parents=True)
    (root / "assets" / "poses").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "review-only",
                "public_release": False,
                "required_review": [
                    "run scripts/signal_room_video_env_gate.py before local render attempts",
                    "run npx hyperframes lint",
                    "run npx hyperframes inspect",
                    "source or design SFX from audio_cue_sheet.json",
                    "run scripts/signal_room_retention_frame_gate.py on sampled proof frames",
                    "replace v0 vector rig if Blender/Moho candidate passes",
                ],
            }
        )
    )
    (root / "audio_cue_sheet.json").write_text(json.dumps({"cues": [{"id": "paper_bill_snap"}]}))
    (root / "motion_primitives.js").write_text(
        """
window.SignalRoomMotion = {
  continuousDrift(tl) {},
  ambientLoop(tl) {},
  microActing(tl, selector, start, duration, amount) {},
  poseTransition(tl, outgoingSelector, incomingSelector, start, options) {},
  feeStackSimmer(tl, start, duration) {},
  machineIdle(tl, start, duration) {},
  machineCausality(tl, start, options) {},
};
"""
    )
    (root / "motion_primitives.json").write_text(
        json.dumps(
            {
                "status": "review-only",
                "public_release": False,
                "primitives": required_motion_primitives(),
            }
        )
    )
    (root / "scene_choreography.json").write_text(
        json.dumps(
            {
                "status": "review-only",
                "public_release": False,
                "composition_id": "fee-machine-v2-review",
                "beats": [
                    {
                        "id": "ordinary_bill_hold",
                        "start": 0.0,
                        "end": 2.5,
                        "acting_objective": "read bill before reacting",
                        "primary_motion": "quiet camera drift and paper entrance",
                        "overlap_in_seconds": 0.0,
                    },
                    {
                        "id": "fee_split_reaction",
                        "start": 2.35,
                        "end": 5.1,
                        "acting_objective": "shock reads before the fee labels finish",
                        "primary_motion": "fee tags push into view with synced ticks",
                        "acting_phases": ["anticipation", "reaction_hold", "settle"],
                        "overlap_in_seconds": 0.15,
                    },
                    {
                        "id": "machine_reveal_handoff",
                        "start": 5.0,
                        "end": 8.65,
                        "acting_objective": "look from bill to the revealed machine",
                        "primary_motion": "wall slide reveals active gears and lever",
                        "acting_phases": ["anticipation", "look_hold", "settle"],
                        "overlap_in_seconds": 0.1,
                    },
                    {
                        "id": "skeptical_point_drive",
                        "start": 8.5,
                        "end": 12.1,
                        "acting_objective": "point with skepticism at fee source",
                        "primary_motion": "pointing pose locks machine cause to fee stack effect",
                        "acting_phases": ["anticipation", "point_hold", "settle"],
                        "overlap_in_seconds": 0.15,
                    },
                    {
                        "id": "memory_anchor_settle",
                        "start": 12.0,
                        "end": 15.0,
                        "acting_objective": "settle into final memory frame",
                        "primary_motion": "lever pulse resolves fee stack and caption",
                        "acting_phases": ["anticipation", "memory_hold", "settle"],
                        "overlap_in_seconds": 0.1,
                    },
                ],
            }
        )
    )
    (root / "retention_frame_plan.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"id": "ordinary_bill"},
                    {"id": "number_split"},
                    {"id": "machine_reveal"},
                    {"id": "acting_read"},
                    {"id": "memory_anchor"},
                ]
            }
        )
    )
    (root / ".hyperframes" / "expanded-prompt.md").write_text("Rhythm: hold -> split -> reveal")
    (root / "DESIGN.md").write_text("Status: review-only")
    (root / "source_rig_manifest.json").write_text(json.dumps({"status": "review-only"}))
    for pose in ["neutral_read", "bill_shock", "look_to_machine", "skeptical_point", "slight_lean"]:
        (root / "assets" / "poses" / f"{pose}.svg").write_text(f"<svg><title>{pose}</title></svg>")
    (root / "index.html").write_text(
        """
<div data-composition-id="fee-machine-v2-review" data-duration="15" data-width="1080" data-height="1920">
  <img src="assets/poses/neutral_read.svg" />
  <img src="assets/poses/bill_shock.svg" />
  <img src="assets/poses/look_to_machine.svg" />
  <img src="assets/poses/skeptical_point.svg" />
  <img src="assets/poses/slight_lean.svg" />
</div>
<script>window.__timelines["fee-machine-v2-review"] = {};</script>
<script src="motion_primitives.js"></script>
<script>
SignalRoomMotion.continuousDrift(tl);
SignalRoomMotion.ambientLoop(tl);
SignalRoomMotion.microActing(tl, ".pose-neutral", .2, 2.1, 10);
SignalRoomMotion.poseTransition(tl, ".pose-neutral", ".pose-shock", 2.35, { anticipation: .12, hold: 2.4, settle: .18 });
SignalRoomMotion.feeStackSimmer(tl, 3.0, 11.5);
SignalRoomMotion.machineIdle(tl, 5.8, 8.9);
SignalRoomMotion.machineCausality(tl, 6.05, { duration: 3.2, pulses: 3 });
</script>
"""
    )


def test_scaffold_gate_passes_for_complete_review_scaffold(tmp_path: Path) -> None:
    module = load_module()
    write_scaffold(tmp_path)

    result = module.evaluate_scaffold(tmp_path)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["composition_id"] == "fee-machine-v2-review"


def test_scaffold_gate_fails_for_public_manifest_missing_pose_and_missing_plan(tmp_path: Path) -> None:
    module = load_module()
    write_scaffold(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "published", "public_release": True}))
    (tmp_path / "assets" / "poses" / "skeptical_point.svg").unlink()
    (tmp_path / "retention_frame_plan.json").unlink()

    result = module.evaluate_scaffold(tmp_path)

    assert result["passed"] is False
    assert "manifest status must be review-only" in result["errors"]
    assert "manifest public_release must be false" in result["errors"]
    assert "missing required file: retention_frame_plan.json" in result["errors"]
    assert "missing required pose asset: assets/poses/skeptical_point.svg" in result["errors"]


def test_scaffold_gate_requires_motion_primitive_registry_and_timeline_markers(tmp_path: Path) -> None:
    module = load_module()
    write_scaffold(tmp_path)
    primitives = json.loads((tmp_path / "motion_primitives.json").read_text())
    primitives["primitives"] = [
        primitive for primitive in primitives["primitives"] if primitive["id"] != "machine_idle"
    ]
    (tmp_path / "motion_primitives.json").write_text(json.dumps(primitives))
    (tmp_path / "index.html").write_text((tmp_path / "index.html").read_text().replace("SignalRoomMotion.feeStackSimmer(", "SignalRoomMotion.feeStackDormant("))

    result = module.evaluate_scaffold(tmp_path)

    assert result["passed"] is False
    assert "motion_primitives.json missing primitive: machine_idle" in result["errors"]
    assert "index.html missing motion primitive marker: SignalRoomMotion.feeStackSimmer(" in result["errors"]


def test_scaffold_gate_requires_pose_transition_and_acting_phases(tmp_path: Path) -> None:
    module = load_module()
    write_scaffold(tmp_path)
    primitives = json.loads((tmp_path / "motion_primitives.json").read_text())
    primitives["primitives"] = [
        primitive for primitive in primitives["primitives"] if primitive["id"] != "pose_transition"
    ]
    (tmp_path / "motion_primitives.json").write_text(json.dumps(primitives))
    choreography = json.loads((tmp_path / "scene_choreography.json").read_text())
    choreography["beats"][1].pop("acting_phases")
    (tmp_path / "scene_choreography.json").write_text(json.dumps(choreography))
    (tmp_path / "index.html").write_text(
        (tmp_path / "index.html").read_text().replace("SignalRoomMotion.poseTransition(", "SignalRoomMotion.poseCut(")
    )

    result = module.evaluate_scaffold(tmp_path)

    assert result["passed"] is False
    assert "motion_primitives.json missing primitive: pose_transition" in result["errors"]
    assert "index.html missing motion primitive marker: SignalRoomMotion.poseTransition(" in result["errors"]
    assert "scene choreography beat fee_split_reaction missing acting_phases" in result["errors"]


def test_scaffold_gate_requires_reusable_motion_library(tmp_path: Path) -> None:
    module = load_module()
    write_scaffold(tmp_path)
    (tmp_path / "motion_primitives.js").unlink()
    (tmp_path / "index.html").write_text((tmp_path / "index.html").read_text().replace("motion_primitives.js", "inline-only.js"))

    result = module.evaluate_scaffold(tmp_path)

    assert result["passed"] is False
    assert "missing required file: motion_primitives.js" in result["errors"]
    assert "index.html missing reusable motion library script: motion_primitives.js" in result["errors"]


def test_scaffold_gate_requires_scene_choreography_for_staged_motion(tmp_path: Path) -> None:
    module = load_module()
    write_scaffold(tmp_path)
    choreography = json.loads((tmp_path / "scene_choreography.json").read_text())
    choreography["beats"] = choreography["beats"][:3]
    choreography["beats"][1].pop("acting_objective")
    choreography["beats"][2]["overlap_in_seconds"] = 0
    (tmp_path / "scene_choreography.json").write_text(json.dumps(choreography))

    result = module.evaluate_scaffold(tmp_path)

    assert result["passed"] is False
    assert "scene_choreography.json must include at least five staged beats" in result["errors"]
    assert "scene choreography beat fee_split_reaction missing acting_objective" in result["errors"]
    assert "scene choreography beat machine_reveal_handoff must overlap adjacent motion by at least 0.10s" in result["errors"]
