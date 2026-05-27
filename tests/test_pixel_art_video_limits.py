import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "creative"
    / "pixel-art"
    / "scripts"
    / "pixel_art_video.py"
)


class FakeImage:
    size = (64, 64)
    saved_paths = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def convert(self, mode):
        return self

    def copy(self):
        return self

    def save(self, path):
        self.saved_paths.append(path)


class FakeDraw:
    def __init__(self, image):
        self.im = image

    def rectangle(self, *args, **kwargs):
        pass


def load_pixel_art_video(monkeypatch):
    fake_image_module = types.SimpleNamespace(open=lambda path: FakeImage())
    fake_draw_module = types.SimpleNamespace(Draw=FakeDraw)
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_image_module
    fake_pil.ImageDraw = fake_draw_module
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", fake_draw_module)

    spec = importlib.util.spec_from_file_location("pixel_art_video_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("duration", "fps", "message"),
    [
        (0, 15, "duration must be between 1 and 30"),
        (31, 15, "duration must be between 1 and 30"),
        (6, 0, "fps must be between 1 and 30"),
        (6, 31, "fps must be between 1 and 30"),
    ],
)
def test_pixel_art_video_rejects_unbounded_timing(monkeypatch, duration, fps, message):
    module = load_pixel_art_video(monkeypatch)

    with pytest.raises(ValueError, match=message):
        module.pixel_art_video("input.png", "out.mp4", duration=duration, fps=fps)


def test_pixel_art_video_rejects_large_source_images(monkeypatch):
    module = load_pixel_art_video(monkeypatch)
    FakeImage.size = (2048, 2048)
    monkeypatch.setattr(module, "_ensure_ffmpeg", lambda: None)

    with pytest.raises(ValueError, match="source image must not exceed"):
        module.pixel_art_video("input.png", "out.mp4")


def test_pixel_art_video_uses_bounded_ffmpeg_timeout(monkeypatch):
    module = load_pixel_art_video(monkeypatch)
    FakeImage.size = (64, 64)
    FakeImage.saved_paths = []
    run_calls = []

    monkeypatch.setattr(module, "_ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: run_calls.append((args, kwargs)))

    module.pixel_art_video("input.png", "out.mp4", duration=1, fps=1)

    assert len(FakeImage.saved_paths) == 1
    assert len(run_calls) == 1
    assert run_calls[0][1]["timeout"] == module.FFMPEG_TIMEOUT_SECONDS
