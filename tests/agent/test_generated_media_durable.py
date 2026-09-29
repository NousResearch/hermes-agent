"""Generated image/video output must survive the gateway media sweep (#126445).

``save_b64_image`` / ``save_*_video`` write the ONLY copy of base64-returning
provider output. The hourly gateway housekeeping deletes 24h-old files from
``cache/images`` / ``cache/videos`` (transient inbound media), so generated
deliverables live in ``cache/generated/<kind>/`` instead — otherwise the
transcript and the Artifacts panel point at deleted files a day later, with no
way to recover the bytes.

Living outside the swept dirs means ``cache/generated`` needs its own
credential-files mount/sync entry: Docker/SSH/Modal backends only reach paths
that map through ``tools.credential_files``, and an unmapped path also drops
``agent_visible_image`` from the provider result.

RED on main: every test here fails — the writers target the swept caches and
``cache/generated`` is not mounted at all.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de00000010494441547801635c0e000000feff03000006000557bfabd400"
    "00000049454e44ae426082"
)


def _backdate_hours(path, hours: float = 25.0) -> None:
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


class TestGeneratedImageSurvivesSweep:
    def test_old_generated_image_not_swept(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.image_gen_provider import save_b64_image
        from gateway.platforms.base import cleanup_image_cache

        path = save_b64_image(base64.b64encode(PNG_1PX).decode(), prefix="red_test")
        _backdate_hours(path)
        removed = cleanup_image_cache(max_age_hours=24)
        assert path.exists(), (
            f"25h-old generated image was swept ({removed} removed): {path}"
        )

    def test_generated_image_outside_swept_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.image_gen_provider import save_b64_image
        from gateway.platforms.base import get_image_cache_dir

        path = save_b64_image(base64.b64encode(PNG_1PX).decode(), prefix="red_test")
        assert path.parent != get_image_cache_dir()
        assert path.exists()


class TestGeneratedVideoSurvivesSweep:
    def test_old_generated_video_not_swept(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.video_gen_provider import save_b64_video
        from gateway.platforms.base import cleanup_video_cache

        path = save_b64_video(base64.b64encode(b"fake-video-bytes").decode(), prefix="red_test")
        _backdate_hours(path)
        removed = cleanup_video_cache(max_age_hours=24)
        assert path.exists(), (
            f"25h-old generated video was swept ({removed} removed): {path}"
        )

    def test_generated_video_outside_swept_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.video_gen_provider import save_b64_video
        from gateway.platforms.base import get_video_cache_dir

        path = save_b64_video(base64.b64encode(b"fake-video-bytes").decode(), prefix="red_test")
        assert path.parent != get_video_cache_dir()
        assert path.exists()


class TestGeneratedMediaReachesSandbox:
    """Out-of-sweep is only half the job: remote backends must still see the file."""

    def test_generated_dir_is_mounted_from_the_writers_location(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.image_gen_provider import save_b64_image
        from tools.credential_files import get_cache_directory_mounts

        path = save_b64_image(base64.b64encode(PNG_1PX).decode(), prefix="red_test")
        mounts = {m["container_path"]: m["host_path"] for m in get_cache_directory_mounts()}
        assert "/root/.hermes/cache/generated" in mounts, sorted(mounts)
        assert path.is_relative_to(Path(mounts["/root/.hermes/cache/generated"]).resolve())

    def test_generated_image_maps_into_the_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.image_gen_provider import save_b64_image
        from tools.credential_files import map_cache_path_to_container

        path = save_b64_image(base64.b64encode(PNG_1PX).decode(), prefix="red_test")
        rel = f"cache/generated/images/{path.name}"
        assert map_cache_path_to_container(str(path)) == f"/root/.hermes/{rel}"
        assert map_cache_path_to_container(str(path), container_base="~/.hermes") == f"~/.hermes/{rel}"

    def test_generated_video_is_in_the_per_file_sync_list(self, tmp_path, monkeypatch):
        """Modal uploads file-by-file through iter_cache_files()."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.video_gen_provider import save_bytes_video
        from tools.credential_files import iter_cache_files

        path = save_bytes_video(b"fake-video-bytes", prefix="red_test")
        # container paths are POSIX; normalize the host separator pytest sees on Windows
        entries = {e["container_path"].replace("\\", "/") for e in iter_cache_files("~/.hermes")}
        assert f"~/.hermes/cache/generated/videos/{path.name}" in entries
