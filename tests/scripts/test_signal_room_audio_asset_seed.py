from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_audio_asset_seed.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_audio_asset_seed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_cue_sheet(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "review-only",
                "cues": [
                    {"id": "room_tone_bed", "duration": 15.0},
                    {"id": "paper_bill_snap", "duration": 0.25},
                    {"id": "final_lever_click", "duration": 0.35},
                ],
            }
        )
    )


def test_audio_asset_seed_writes_review_wavs_and_manifest(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    out = tmp_path / "audio_assets"
    write_cue_sheet(cue_sheet)

    manifest = module.create_audio_asset_seed(cue_sheet, out)

    assert manifest["status"] == "review-only"
    assert manifest["public_release"] is False
    assert manifest["source"] == "placeholder seed; replace with final sound design"
    assert [asset["cue_id"] for asset in manifest["assets"]] == [
        "room_tone_bed",
        "paper_bill_snap",
        "final_lever_click",
    ]
    assert (out / "audio_asset_manifest.json").exists()
    for asset in manifest["assets"]:
        data = (out / asset["filename"]).read_bytes()
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"


def test_audio_asset_seed_writes_audible_distinct_placeholder_content(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    out = tmp_path / "audio_assets"
    write_cue_sheet(cue_sheet)

    module.create_audio_asset_seed(cue_sheet, out)

    payloads = {}
    for filename in ["room_tone_bed.wav", "paper_bill_snap.wav", "final_lever_click.wav"]:
        data = (out / filename).read_bytes()
        data_offset = data.index(b"data") + 8
        samples = struct.unpack("<" + "h" * ((len(data) - data_offset) // 2), data[data_offset:])
        assert any(sample != 0 for sample in samples)
        payloads[filename] = data[data_offset:]
    assert len(set(payloads.values())) == 3


def test_audio_asset_seed_refuses_existing_output_without_force(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    out = tmp_path / "audio_assets"
    write_cue_sheet(cue_sheet)
    out.mkdir()

    try:
        module.create_audio_asset_seed(cue_sheet, out)
    except FileExistsError as exc:
        assert str(out) in str(exc)
    else:
        raise AssertionError("expected existing output directory to fail without force")
