from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_audio_asset_designer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_audio_asset_designer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_audio_asset_designer_creates_non_placeholder_review_wavs(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    cue_sheet.write_text(
        json.dumps(
            {
                "cues": [
                    {"id": "paper_bill_snap", "start": 0.35, "duration": 0.25, "sound": "dry paper snap"},
                    {"id": "machine_drive_loop", "start": 6.05, "duration": 1.2, "sound": "gear belt rhythm"},
                    {"id": "final_lever_click", "start": 12.2, "duration": 0.35, "sound": "lever click"},
                ]
            }
        )
    )
    out_dir = tmp_path / "audio_assets"

    manifest = module.create_review_audio_assets(cue_sheet, out_dir)

    assert manifest["source"] == "procedural review sound design"
    assert manifest["public_release"] is False
    assert all(asset["placeholder"] is False for asset in manifest["assets"])
    assert all(asset["replacement_required"] == "final mix/master approval" for asset in manifest["assets"])
    assert (out_dir / "paper_bill_snap.wav").read_bytes().startswith(b"RIFF")
    assert (out_dir / "machine_drive_loop.wav").read_bytes().startswith(b"RIFF")
    assert (out_dir / "final_lever_click.wav").read_bytes().startswith(b"RIFF")
    assert (out_dir / "paper_bill_snap.wav").read_bytes() != (out_dir / "final_lever_click.wav").read_bytes()
    assert (out_dir / "audio_asset_manifest.json").exists()


def test_audio_asset_designer_refuses_existing_output_without_force(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    cue_sheet.write_text(json.dumps({"cues": [{"id": "room_tone_bed", "duration": 0.25}]}))
    out_dir = tmp_path / "audio_assets"
    out_dir.mkdir()

    try:
        module.create_review_audio_assets(cue_sheet, out_dir)
    except FileExistsError as exc:
        assert exc.args[0] == out_dir
    else:
        raise AssertionError("expected FileExistsError")
