from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_render_asset_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_render_asset_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_render_asset_gate_passes_for_review_only_mp4(tmp_path: Path) -> None:
    module = load_module()
    render_path = tmp_path / "fee_machine_v2_draft.mp4"
    render_path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + (b"\x00" * 1500))

    result = module.evaluate_render_asset(render_path)

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["render_path"] == str(render_path)
    assert result["format"] == "mp4"
    assert result["review_only"] is True


def test_render_asset_gate_fails_missing_tiny_or_wrong_extension(tmp_path: Path) -> None:
    module = load_module()

    missing = module.evaluate_render_asset(tmp_path / "missing.mp4")
    assert missing["passed"] is False
    assert "missing render asset" in missing["errors"]

    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"ftyp")
    tiny_result = module.evaluate_render_asset(tiny)
    assert tiny_result["passed"] is False
    assert "render asset is too small to be a usable review draft" in tiny_result["errors"]

    mov = tmp_path / "draft.mov"
    mov.write_bytes(b"\x00" * 2000)
    mov_result = module.evaluate_render_asset(mov)
    assert mov_result["passed"] is False
    assert "render asset must be an .mp4 file" in mov_result["errors"]
