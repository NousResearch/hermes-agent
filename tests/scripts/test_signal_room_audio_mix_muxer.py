from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_audio_mix_muxer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_audio_mix_muxer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    cue_sheet = tmp_path / "audio_cue_sheet.json"
    cue_sheet.write_text(
        json.dumps(
            {
                "duration_seconds": 15,
                "cues": [
                    {"id": "room_tone_bed", "start": 0.0, "duration": 15.0},
                    {"id": "paper_bill_snap", "start": 0.35, "duration": 0.25},
                ],
            }
        )
    )
    asset_dir = tmp_path / "audio_assets"
    asset_dir.mkdir()
    (asset_dir / "room_tone_bed.wav").write_bytes(b"RIFF0000WAVE")
    (asset_dir / "paper_bill_snap.wav").write_bytes(b"RIFF0000WAVE")
    (asset_dir / "audio_asset_manifest.json").write_text(
        json.dumps(
            {
                "status": "review-only",
                "public_release": False,
                "assets": [
                    {"cue_id": "room_tone_bed", "filename": "room_tone_bed.wav", "placeholder": False},
                    {"cue_id": "paper_bill_snap", "filename": "paper_bill_snap.wav", "placeholder": False},
                ],
            }
        )
    )
    render = tmp_path / "draft.mp4"
    render.write_bytes(b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 1500))
    out = tmp_path / "draft_with_audio.mp4"
    return cue_sheet, asset_dir, render, out


def test_audio_mix_muxer_builds_delayed_mix_command(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet, asset_dir, render, out = write_inputs(tmp_path)

    command, metadata = module.build_mux_command(cue_sheet, asset_dir, render, out)

    assert command[:4] == ["ffmpeg", "-y", "-i", str(render)]
    assert str(asset_dir / "room_tone_bed.wav") in command
    assert str(asset_dir / "paper_bill_snap.wav") in command
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "adelay=0:all=1" in filter_complex
    assert "adelay=350:all=1" in filter_complex
    assert "amix=inputs=2:duration=longest:normalize=0" in filter_complex
    assert command[-8:] == ["-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
    assert metadata["cue_count"] == 2
    assert metadata["duration_seconds"] == 15


def test_audio_mix_muxer_writes_scorecard_after_runner(tmp_path: Path) -> None:
    module = load_module()
    cue_sheet, asset_dir, render, out = write_inputs(tmp_path)
    scorecard = tmp_path / "mux_scorecard.json"
    commands = []

    def fake_runner(command: list[str]) -> None:
        commands.append(command)
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 2048))

    result = module.create_audio_muxed_review(
        cue_sheet,
        asset_dir,
        render,
        out,
        scorecard_out=scorecard,
        runner=fake_runner,
    )

    assert result["passed"] is True
    assert result["output"] == str(out)
    assert result["size_bytes"] > 1024
    assert commands
    assert json.loads(scorecard.read_text())["passed"] is True
