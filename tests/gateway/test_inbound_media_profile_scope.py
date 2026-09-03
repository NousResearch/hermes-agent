"""Regression: inbound attachments must land in the routed profile's cache (#101134).

A platform adapter downloads and caches an attachment before the gateway knows
which multiplexed profile owns the turn, so the bytes go to the process/launch
home's cache.  The turn then runs under ``_profile_runtime_scope``, where the
cache mount table — and the Docker bind mounts built from it — resolve the
*profile's* cache.  The agent was handed ``/root/.hermes/cache/documents/x.pdf``:
a path that exists, is mounted, and is empty.

Covered here:

* documents and photos are moved into the active profile's cache;
* the sandbox path the agent is told about now names a directory that actually
  holds the file;
* the pass is idempotent, single-profile gateways never touch disk, and a
  failed move degrades to the pre-fix path instead of dropping the attachment;
* ``event.media_urls`` carries one coordinate system (host paths), which is
  the second, independent defect the issue reports.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.inbound_media_scope import (
    _is_existing_file,
    rehome_event_media,
    rehome_media_paths,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A process/launch home plus a routed profile home, with the profile bound."""
    process_home = tmp_path / "hermes"
    profile_home = process_home / "profiles" / "profile-a"
    for home in (process_home, profile_home):
        (home / "cache" / "documents").mkdir(parents=True)
        (home / "cache" / "images").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    token = set_hermes_home_override(str(profile_home))
    try:
        yield SimpleNamespace(process=process_home, profile=profile_home)
    finally:
        reset_hermes_home_override(token)


def _stage(home: Path, kind: str, name: str, body: bytes = b"%PDF-1.4 body") -> Path:
    path = home / "cache" / kind / name
    path.write_bytes(body)
    return path


class TestRehomeMediaPaths:
    def test_document_moves_into_the_routed_profile_cache(self, homes):
        staged = _stage(homes.process, "documents", "report.pdf")

        (rehomed,) = rehome_media_paths([str(staged)])

        assert Path(rehomed) == homes.profile / "cache" / "documents" / "report.pdf"
        assert Path(rehomed).read_bytes() == b"%PDF-1.4 body"
        assert not staged.exists(), "the process-home copy must not linger"

    def test_agent_visible_path_names_the_mounted_directory(self, homes, monkeypatch):
        """The bug in one assertion: sandbox path + profile mount must agree."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        from tools.credential_files import to_agent_visible_cache_path

        staged = _stage(homes.process, "documents", "report.pdf")

        # Pre-fix: translating the process-home path under the profile scope
        # yields a sandbox path whose bind-mounted host directory is empty.
        pre_fix = to_agent_visible_cache_path(str(staged))
        assert pre_fix == str(staged), "process-home file is outside the profile mounts"

        (rehomed,) = rehome_media_paths([str(staged)])
        agent_path = to_agent_visible_cache_path(rehomed)

        assert agent_path == "/root/.hermes/cache/documents/report.pdf"
        # ...and the directory Docker mounts at that path now holds the file.
        assert (homes.profile / "cache" / "documents" / "report.pdf").is_file()

    def test_image_becomes_readable_by_the_host_vision_path(self, homes, monkeypatch):
        """`_media_cache_roots()` only authorises the ACTIVE home's cache."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        from tools.image_source import _permitted_host_read_target

        staged = _stage(homes.process, "images", "img_abc.jpg", b"\xff\xd8\xff body")
        ctx = SimpleNamespace()

        assert _permitted_host_read_target(staged, ctx) is None

        (rehomed,) = rehome_media_paths([str(staged)])

        assert _permitted_host_read_target(Path(rehomed), ctx) is not None

    def test_sandbox_form_entries_are_translated_before_moving(self, homes, monkeypatch):
        """An out-of-tree adapter may still push a container-form path."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        staged = _stage(homes.process, "documents", "legacy.pdf")
        container_form = "/root/.hermes/cache/documents/legacy.pdf"

        (rehomed,) = rehome_media_paths([container_form])

        assert Path(rehomed) == homes.profile / "cache" / "documents" / "legacy.pdf"
        assert not staged.exists()

    def test_second_pass_is_a_no_op(self, homes):
        staged = _stage(homes.process, "documents", "report.pdf")

        once = rehome_media_paths([str(staged)])
        twice = rehome_media_paths(once)

        assert twice == once
        assert Path(once[0]).is_file()

    def test_single_profile_gateway_leaves_paths_alone(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "cache" / "documents").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        staged = _stage(home, "documents", "report.pdf")

        assert rehome_media_paths([str(staged)]) == [str(staged)]
        assert staged.is_file()

    def test_unknown_paths_are_returned_untouched(self, homes, tmp_path):
        outside = tmp_path / "elsewhere" / "notes.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("x", encoding="utf-8")

        assert rehome_media_paths([str(outside), "https://example.com/a.png"]) == [
            str(outside),
            "https://example.com/a.png",
        ]
        assert outside.is_file()

    def test_move_failure_keeps_the_original_path(self, homes, monkeypatch):
        staged = _stage(homes.process, "documents", "report.pdf")

        def _boom(*_args, **_kwargs):
            raise PermissionError("locked")

        monkeypatch.setattr("gateway.inbound_media_scope.os.replace", _boom)
        monkeypatch.setattr("gateway.inbound_media_scope.shutil.copy2", _boom)

        assert rehome_media_paths([str(staged)]) == [str(staged)]
        assert staged.is_file(), "a failed move must not lose the attachment"

    def test_probe_survives_an_unreadable_parent(self, monkeypatch):
        """`/root` is 0700: pathlib re-raises EACCES instead of answering False."""
        denied = Path("/root/.hermes/cache/documents/x.pdf")

        def _raise(_self):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(type(denied), "is_file", _raise, raising=False)

        assert _is_existing_file(denied) is False


class TestRehomeEventMedia:
    def test_rewrites_media_urls_and_the_baked_transcript_note(self, homes, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        staged = _stage(homes.process, "documents", "report.pdf")
        event = SimpleNamespace(
            media_urls=[str(staged)],
            text=f"[document 'report.pdf' saved at: {staged}] hi",
        )

        rehome_event_media(event)

        assert event.media_urls == [
            str(homes.profile / "cache" / "documents" / "report.pdf")
        ]
        assert "/root/.hermes/cache/documents/report.pdf" in event.text
        assert str(staged) not in event.text

    def test_event_without_media_is_untouched(self, homes):
        event = SimpleNamespace(media_urls=[], text="hello")
        rehome_event_media(event)
        assert event.media_urls == []
        assert event.text == "hello"


class TestMediaUrlsCoordinateSystem:
    """The second, independent defect: media_urls held two path forms."""

    @pytest.fixture(autouse=True)
    def _docker_cache(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "cache" / "documents").mkdir(parents=True)
        (home / "cache" / "images").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        return home

    def test_document_and_photo_agree_on_the_host_form(self):
        from gateway.platforms.base import cache_media_bytes, cache_image_from_bytes

        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d49444154789c6360000002000154a24f5f0000000049454e44ae426082"
        )
        document = cache_media_bytes(
            b"%PDF-1.4 body", filename="report.pdf", mime_type="application/pdf"
        )
        photo_direct = cache_image_from_bytes(png, ext=".png")
        photo_dispatch = cache_media_bytes(png, filename="p.png", mime_type="image/png")

        assert document is not None and photo_dispatch is not None
        for path in (document.path, photo_dispatch.path, photo_direct):
            assert os.path.isabs(path)
            assert not path.startswith("/root/.hermes"), "container form leaked"
            assert Path(path).is_file()

    def test_context_note_still_renders_the_sandbox_path(self):
        from gateway.platforms.base import cache_media_bytes

        cached = cache_media_bytes(
            b"%PDF-1.4 body", filename="report.pdf", mime_type="application/pdf"
        )

        assert cached is not None
        assert "/root/.hermes/cache/documents/" in cached.context_note()
