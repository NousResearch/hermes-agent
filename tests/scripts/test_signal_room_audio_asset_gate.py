from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_audio_asset_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_audio_asset_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, *, samples: int = 1600) -> None:
    sample_rate = 16000
    data = b"\x00\x00" * samples
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def write_cue_sheet(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "review-only",
                "cues": [
                    {"id": "room_tone_bed", "start": 0.0, "duration": 15.0},
                    {"id": "paper_bill_snap", "start": 0.35, "duration": 0.25},
                    {"id": "final_lever_click", "start": 12.2, "duration": 0.35},
                ],
            }
        )
    )


def write_manifest(asset_dir: Path, cue_ids: list[str], *, public_release: bool = False) -> None:
    (asset_dir / "audio_asset_manifest.json").write_text(
        json.dumps(
            {
                "status": "review-only",
                "public_release": public_release,
                "source": "placeholder seed; replace with final sound design",
                "assets": [{"cue_id": cue_id, "filename": f"{cue_id}.wav"} for cue_id in cue_ids],
            }
        )
    )


def test_audio_asset_gate_passes_for_review_only_wav_assets(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    assets = tmp_path / "audio_assets"
    assets.mkdir()
    write_cue_sheet(cue_sheet)
    cue_ids = ["room_tone_bed", "paper_bill_snap", "final_lever_click"]
    write_manifest(assets, cue_ids)
    for cue_id in cue_ids:
        write_wav(assets / f"{cue_id}.wav")

    result = module.evaluate_audio_assets(cue_sheet, assets)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["required_cue_ids"] == cue_ids
    assert result["assets"]["final_lever_click.wav"]["sample_rate"] == 16000


def test_audio_asset_gate_fails_for_missing_bad_or_public_assets(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    assets = tmp_path / "audio_assets"
    assets.mkdir()
    write_cue_sheet(cue_sheet)
    write_manifest(assets, ["room_tone_bed", "paper_bill_snap"], public_release=True)
    write_wav(assets / "room_tone_bed.wav")
    (assets / "paper_bill_snap.wav").write_bytes(b"not wave")

    result = module.evaluate_audio_assets(cue_sheet, assets)

    assert result["passed"] is False
    assert "missing required audio asset: final_lever_click.wav" in result["errors"]
    assert "paper_bill_snap.wav not a RIFF/WAVE file" in result["errors"]
    assert "manifest public_release must be false for review gate" in result["errors"]
