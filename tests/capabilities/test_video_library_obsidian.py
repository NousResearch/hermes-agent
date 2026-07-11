from pathlib import Path

import pytest

from capabilities.video_library.obsidian import ObsidianProjector, write_markdown_atomic


def _asset(root: Path) -> dict:
    return {
        "id": "asset_123",
        "library_id": "beef-noodle",
        "original_name": "厨师下午拉面01.mp4",
        "sha256": "abc123",
        "source_path": str(root / "01_原始素材" / "厨师下午拉面01.mp4"),
        "status": "analyzed",
    }


def _clips(root: Path) -> list[dict]:
    return [
        {
            "confidence": 0.92,
            "description": "厨师将拉好的细面下入沸水锅",
            "end_seconds": 13.7,
            "id": "clip_1",
            "keyframe_path": str(root / "03_关键帧" / "asset_123" / "clip-0000.jpg"),
            "quality_score": 0.89,
            "start_seconds": 8.4,
            "status": "ready",
            "tags": [
                {"confidence": 0.95, "name": "动作/拉面", "source": "semantic-controlled"},
                {"confidence": 1, "name": "竖屏", "source": "technical"},
            ],
        }
    ]


def test_projection_writes_readable_asset_page(tmp_path):
    root = tmp_path / "牛肉面资产库"
    root.mkdir()
    projector = ObsidianProjector(root)

    path = projector.write_asset(_asset(root), _clips(root))

    text = path.read_text(encoding="utf-8")
    assert "# 厨师下午拉面01" in text
    assert "动作/拉面" in text
    assert "00:08.400-00:13.700" in text
    assert "03_关键帧/asset_123/clip-0000.jpg" in text


def test_projection_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "04_素材分析" / "单条视频分析" / "asset.md"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr("capabilities.video_library.obsidian.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(OSError, match="disk"):
        write_markdown_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "old"


def test_stats_page_summarizes_statuses(tmp_path):
    root = tmp_path / "牛肉面资产库"
    root.mkdir()
    projector = ObsidianProjector(root)

    path = projector.write_stats([_asset(root)], [*_clips(root), {**_clips(root)[0], "id": "clip_2", "status": "unusable"}])

    text = path.read_text(encoding="utf-8")
    assert "素材数量 | 1" in text
    assert "镜头数量 | 2" in text
    assert "不可用镜头 | 1" in text
