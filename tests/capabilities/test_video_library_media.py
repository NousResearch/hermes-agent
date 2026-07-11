import json
import shutil
import subprocess

import pytest

from capabilities.video_library import media


def test_scene_boundaries_filter_short_segments_and_keep_duration():
    stderr = "\n".join(
        [
            "[Parsed_showinfo_1] n:0 pts:123 pts_time:0.4",
            "[Parsed_showinfo_1] n:1 pts:800 pts_time:2.0",
            "[Parsed_showinfo_1] n:2 pts:2000 pts_time:5.0",
        ]
    )

    boundaries = media.parse_scene_boundaries(
        stderr,
        duration_seconds=8.0,
        min_clip_seconds=1.0,
        max_clips=10,
    )

    assert boundaries == [(0.0, 2.0), (2.0, 5.0), (5.0, 8.0)]


def test_scene_boundaries_fall_back_to_fixed_windows():
    assert media.parse_scene_boundaries(
        "",
        duration_seconds=12.0,
        min_clip_seconds=1.0,
        max_clips=10,
        fallback_clip_seconds=5.0,
    ) == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]


def test_probe_media_parses_ffprobe_json(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    monkeypatch.setattr(media, "resolve_ffprobe", lambda _ffmpeg=None: "/tmp/ffprobe")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "format": {"duration": "7.25"},
                    "streams": [{"height": 1920, "r_frame_rate": "30/1", "width": 1080}],
                }
            ),
            stderr="",
        ),
    )

    assert media.probe_media(source) == {
        "duration_seconds": 7.25,
        "fps": 30.0,
        "height": 1920,
        "width": 1080,
    }


def test_extract_clip_rejects_output_outside_library(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    with pytest.raises(ValueError, match="managed library root"):
        media.extract_clip(
            source,
            tmp_path / "outside.mp4",
            start_seconds=0,
            end_seconds=2,
            library_root=tmp_path / "library",
            ffmpeg_path="/tmp/ffmpeg",
        )


def test_real_ffmpeg_scene_split_smoke(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg integration binaries are unavailable")
    source = tmp_path / "color-scenes.mp4"
    generated = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=1.5:r=24",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=1.5:r=24",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:d=1.5:r=24",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert generated.returncode == 0, generated.stderr

    metadata = media.probe_media(source, ffmpeg_path=ffmpeg)
    boundaries = media.detect_scene_boundaries(
        source,
        duration_seconds=metadata["duration_seconds"],
        threshold=0.1,
        min_clip_seconds=0.5,
        fallback_clip_seconds=10,
        ffmpeg_path=ffmpeg,
    )
    output = tmp_path / "library" / "clips" / "first.mp4"
    media.extract_clip(
        source,
        output,
        start_seconds=boundaries[0][0],
        end_seconds=boundaries[0][1],
        library_root=tmp_path / "library",
        ffmpeg_path=ffmpeg,
    )

    assert len(boundaries) >= 2
    assert output.is_file()
    assert media.probe_media(output, ffmpeg_path=ffmpeg)["duration_seconds"] > 0
