from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_first_frame_pack.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_first_frame_pack", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_first_frame_pack_exports_hook_candidates(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "draft.mp4"
    render.write_bytes(b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 1500))
    out_dir = tmp_path / "first_frame_candidates"
    commands = []

    def fake_runner(command: list[str]) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"\x89PNG\r\n\x1a\ncandidate")

    manifest = module.create_first_frame_pack(render, out_dir, runner=fake_runner)

    assert manifest["status"] == "review-only"
    assert manifest["source_render"] == str(render)
    assert [candidate["id"] for candidate in manifest["candidates"]] == [
        "cold_open_bill",
        "human_read",
        "pre_split_tension",
    ]
    assert all(candidate["placeholder"] is False for candidate in manifest["candidates"])
    assert (out_dir / "cold_open_bill.png").exists()
    assert (out_dir / "first_frame_review.md").exists()
    assert "who is affected" in (out_dir / "first_frame_review.md").read_text()
    assert commands[0][:4] == ["ffmpeg", "-y", "-ss", "0.35"]
    assert commands[0][-5:] == ["-frames:v", "1", "-update", "1", str(out_dir / "cold_open_bill.png")]


def test_first_frame_pack_refuses_existing_output_without_force(tmp_path: Path) -> None:
    module = load_module()
    render = tmp_path / "draft.mp4"
    render.write_bytes(b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 1500))
    out_dir = tmp_path / "first_frame_candidates"
    out_dir.mkdir()

    try:
        module.create_first_frame_pack(render, out_dir)
    except FileExistsError as exc:
        assert exc.args[0] == out_dir
    else:
        raise AssertionError("expected FileExistsError")
