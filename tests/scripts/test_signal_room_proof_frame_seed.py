from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_proof_frame_seed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_proof_frame_seed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_frame_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "review-only",
                "frames": [
                    {"id": "ordinary_bill", "sample_time": 1.25},
                    {"id": "number_split", "sample_time": 3.75},
                    {"id": "machine_reveal", "sample_time": 6.5},
                    {"id": "acting_read", "sample_time": 9.25},
                    {"id": "memory_anchor", "sample_time": 13.2},
                ],
            }
        )
    )


def read_png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    return struct.unpack(">II", data[16:24])


def test_proof_frame_seed_writes_distinct_review_frames_and_manifest(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    out = tmp_path / "proof_frames"
    write_frame_plan(plan)

    manifest = module.create_proof_frame_seed(plan, out)

    assert manifest["status"] == "review-only"
    assert manifest["public_release"] is False
    assert manifest["source_composition"] == "fee-machine-v2-review"
    assert manifest["render_tool"] == "placeholder seed; replace with HyperFrames frame export"
    assert [frame["id"] for frame in manifest["frames"]] == [
        "ordinary_bill",
        "number_split",
        "machine_reveal",
        "acting_read",
        "memory_anchor",
    ]
    assert (out / "proof_frame_manifest.json").exists()
    hashes = set()
    for frame in manifest["frames"]:
        frame_path = out / f"{frame['id']}.png"
        assert frame_path.exists()
        assert read_png_size(frame_path) == (1080, 1920)
        hashes.add(frame_path.read_bytes())
    assert len(hashes) == 5


def test_proof_frame_seed_refuses_existing_output_without_force(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "retention_frame_plan.json"
    out = tmp_path / "proof_frames"
    write_frame_plan(plan)
    out.mkdir()

    try:
        module.create_proof_frame_seed(plan, out)
    except FileExistsError as exc:
        assert str(out) in str(exc)
    else:
        raise AssertionError("expected existing output directory to fail without force")
