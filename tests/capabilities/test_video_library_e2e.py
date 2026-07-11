import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from capabilities.video_library.batch import VideoLibraryBatchRunner
from capabilities.video_library.config import VideoLibraryConfig
from capabilities.video_library.semantic import SemanticClipResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is unavailable")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=3:r=24",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=3:r=24",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "mpeg4",
            "-y",
            str(path),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg sample generation failed: {completed.stderr[-500:]}")


def test_linked_library_full_ingest_search_and_lazy_timeline(tmp_path):
    root = tmp_path / "牛肉面资产库"
    source_dir = root / "01_原始素材"
    source_dir.mkdir(parents=True)
    source = source_dir / "门店实拍.mp4"
    _sample_video(source)
    original_hash = _sha256(source)
    library = VideoLibraryConfig(
        id="beef-noodle",
        mode="linked",
        name="牛肉面资产库",
        root=root,
        source_roots=(source_dir,),
        taxonomy="beef-noodle-v1",
    )

    def fake_semantic(_frames, *, taxonomy):
        assert taxonomy == "beef-noodle-v1"
        return SemanticClipResult(
            confidence=0.94,
            controlled_tags=["主体/厨师", "动作/拉面", "用途/制作过程"],
            free_tags=["门店实拍"],
            model="fake-vision-e2e",
            quality_score=0.9,
            raw={"content": {"actions": ["拉面"], "summary": "厨师手工拉面"}},
            search_text="后厨厨师手工拉面 制作过程",
            summary="厨师手工拉面",
        )

    runner = VideoLibraryBatchRunner(library, semantic_analyzer=fake_semantic)

    first = runner.scan()

    assert first.complete == 1
    assert first.failed == 0
    assert library.database_path.is_file()
    assert _sha256(source) == original_hash
    service = runner._service()
    clips = service.store.search_clips("厨师手工拉面")
    assert clips
    assert all(not clip["materialized"] for clip in clips)
    assert all(Path(clip["keyframe_path"]).is_file() for clip in clips)
    assert list((root / "04_素材分析" / "单条视频分析").glob("*.md"))

    timeline = service.create_timeline([clips[0]["id"]], aspect="9:16")

    materialized = service.store.get_clip(clips[0]["id"])
    assert materialized is not None
    assert materialized["materialized"] is True
    assert Path(materialized["file_path"]).is_file()
    assert Path(timeline["path"]).is_file()
    assert _sha256(source) == original_hash

    second = runner.scan()

    assert second.skipped == 1
    assert second.complete == 0
    assert len(service.store.list_assets()) == 1
