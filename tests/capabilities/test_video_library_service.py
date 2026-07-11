import json
from pathlib import Path

import pytest

from capabilities.video_library.service import VideoLibraryService
from capabilities.video_library.store import VideoLibraryStore


def _fake_keyframe(_source, output, **_kwargs):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpeg")
    return path


def test_analyze_asset_probes_splits_and_tags_clips(tmp_path, monkeypatch):
    source = tmp_path / "merchant.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    service = VideoLibraryService(store)
    asset = store.import_asset(source)

    monkeypatch.setattr(
        "capabilities.video_library.service.media.probe_media",
        lambda _path: {"duration_seconds": 6.0, "fps": 30.0, "height": 1920, "width": 1080},
    )
    monkeypatch.setattr(
        "capabilities.video_library.service.media.detect_scene_boundaries",
        lambda *_args, **_kwargs: [(0.0, 2.5), (2.5, 6.0)],
    )

    def fake_clip(_source, output, **_kwargs):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"clip")
        return path

    def fake_keyframe(_source, output, **_kwargs):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"jpeg")
        return path

    monkeypatch.setattr("capabilities.video_library.service.media.extract_clip", fake_clip)
    monkeypatch.setattr("capabilities.video_library.service.media.extract_keyframe", fake_keyframe)

    result = service.analyze_asset(asset["id"])

    assert result["job"]["state"] == "complete"
    assert result["asset"]["status"] == "analyzed"
    assert len(result["clips"]) == 2
    assert {tag["name"] for tag in result["clips"][0]["tags"]} == {"中镜头", "高清", "竖屏"}


def test_create_timeline_uses_managed_clip_files(tmp_path):
    source = tmp_path / "merchant.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    service = VideoLibraryService(store)
    asset = store.import_asset(source)
    clips = store.replace_clips(
        asset["id"],
        [
            {"start_seconds": 0, "end_seconds": 2, "file_path": str(store.clips_dir / "a.mp4")},
            {"start_seconds": 2, "end_seconds": 5.5, "file_path": str(store.clips_dir / "b.mp4")},
        ],
    )

    result = service.create_timeline(
        [clip["id"] for clip in clips],
        aspect="9:16",
        script=[{"id": "line-1", "text": "开头文案", "start": 0, "end": 2}],
    )

    persisted = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert persisted["version"] == 1
    assert persisted["aspect"] == "9:16"
    assert [item["start"] for item in persisted["tracks"]["video"]] == [0.0, 2.0]
    assert [item["end"] for item in persisted["tracks"]["video"]] == [2.0, 5.5]
    assert persisted["script"][0]["text"] == "开头文案"


def test_failed_reanalysis_preserves_previous_clips(tmp_path, monkeypatch):
    source = tmp_path / "merchant.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=tmp_path / "library")
    service = VideoLibraryService(store)
    asset = store.import_asset(source)
    old_clip = store.clips_dir / asset["id"] / "clip-0000.mp4"
    old_keyframe = store.keyframes_dir / asset["id"] / "clip-0000.jpg"
    old_clip.parent.mkdir(parents=True)
    old_keyframe.parent.mkdir(parents=True)
    old_clip.write_bytes(b"old clip")
    old_keyframe.write_bytes(b"old keyframe")
    previous = store.replace_clips(
        asset["id"],
        [
            {
                "start_seconds": 0,
                "end_seconds": 2,
                "file_path": str(old_clip),
                "keyframe_path": str(old_keyframe),
            }
        ],
    )

    monkeypatch.setattr(
        "capabilities.video_library.service.media.probe_media",
        lambda _path: {"duration_seconds": 3.0, "fps": 24.0, "height": 1920, "width": 1080},
    )
    monkeypatch.setattr(
        "capabilities.video_library.service.media.detect_scene_boundaries",
        lambda *_args, **_kwargs: [(0.0, 3.0)],
    )
    monkeypatch.setattr(
        "capabilities.video_library.service.media.extract_clip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ffmpeg failed")),
    )

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        service.analyze_asset(asset["id"])

    clips = store.list_clips(asset_id=asset["id"])
    assert [clip["id"] for clip in clips] == [previous[0]["id"]]
    assert old_clip.read_bytes() == b"old clip"
    assert old_keyframe.read_bytes() == b"old keyframe"


def test_linked_analysis_extracts_keyframes_without_clip_mp4(tmp_path, monkeypatch):
    root = tmp_path / "牛肉面资产库"
    source_dir = root / "01_原始素材"
    source_dir.mkdir(parents=True)
    source = source_dir / "长素材.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(
        root=root,
        db_path=root / ".hermes-assets" / "index.sqlite",
        clips_dir=root / "02_精选镜头",
        keyframes_dir=root / "03_关键帧",
    )
    service = VideoLibraryService(store)
    asset = store.import_asset(source, source_mode="linked", library_id="beef-noodle")
    monkeypatch.setattr(
        "capabilities.video_library.service.media.probe_media",
        lambda _path: {"duration_seconds": 10.0, "fps": 30.0, "height": 1920, "width": 1080},
    )
    monkeypatch.setattr(
        "capabilities.video_library.service.media.detect_scene_boundaries",
        lambda *_args, **_kwargs: [(0.0, 5.0), (5.0, 10.0)],
    )
    monkeypatch.setattr("capabilities.video_library.service.media.extract_keyframe", _fake_keyframe)
    monkeypatch.setattr(
        "capabilities.video_library.service.media.extract_clip",
        lambda *_args, **_kwargs: pytest.fail("linked analysis must not materialize clips"),
    )

    result = service.analyze_asset(asset["id"])

    assert len(result["clips"]) == 2
    assert all(not clip["materialized"] for clip in result["clips"])
    assert all(clip["file_path"] == "" for clip in result["clips"])
    assert all(Path(clip["keyframe_path"]).is_file() for clip in result["clips"])


def test_materialize_clip_extracts_exact_source_range(tmp_path, monkeypatch):
    root = tmp_path / "牛肉面资产库"
    source_dir = root / "01_原始素材"
    source_dir.mkdir(parents=True)
    source = source_dir / "长素材.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(
        root=root,
        db_path=root / ".hermes-assets" / "index.sqlite",
        clips_dir=root / "02_精选镜头",
        keyframes_dir=root / "03_关键帧",
    )
    service = VideoLibraryService(store)
    asset = store.import_asset(source, source_mode="linked", library_id="beef-noodle")
    clip = store.replace_clips(
        asset["id"],
        [
            {
                "end_seconds": 7.5,
                "file_path": "",
                "materialized": False,
                "source_file_path": str(source),
                "start_seconds": 2.0,
            }
        ],
    )[0]
    captured = {}

    def fake_extract(source_path, output, **kwargs):
        captured.update(source=Path(source_path), output=Path(output), **kwargs)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"clip")
        return Path(output)

    monkeypatch.setattr("capabilities.video_library.service.media.extract_clip", fake_extract)

    result = service.materialize_clip(clip["id"])

    assert captured["source"] == source.resolve()
    assert captured["start_seconds"] == 2.0
    assert captured["end_seconds"] == 7.5
    assert result["materialized"] is True
    assert Path(result["file_path"]).read_bytes() == b"clip"
