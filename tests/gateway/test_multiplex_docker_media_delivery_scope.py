"""Regression for #109024: a multiplexed secondary's Docker ``MEDIA:`` paths must be validated
under THAT profile's home + terminal policy on the adapter delivery side.

The routed handler runs inside ``_profile_runtime_scope``, but ``_process_message_background``
extracts and filters the reply's media AFTER that scope has exited. Docker path translation
(``_docker_sandbox_dir_candidates`` / ``_parse_docker_volume_mounts``) infers the producing
container from the ACTIVE profile, so a secondary's ``MEDIA:/output/x.png`` resolved through the
default profile's mounts (a decoy of the same name) or was dropped as "not found on this host".
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePosixPath

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _async_profile_runtime_scope
from gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)
        self.images: list[bytes] = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None, **kw):
        self.images.append(Path(image_path).read_bytes())
        return SendResult(success=True, message_id="i")


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    await (stop_event.wait() if stop_event is not None else asyncio.Event().wait())


def _docker_profile(home: Path, mount: Path, *, forward_slashes: bool = False) -> None:
    mount.mkdir(parents=True)
    host = mount.as_posix() if forward_slashes else str(mount)
    (home / "config.yaml").write_text(json.dumps(
        {"terminal": {"backend": "docker", "docker_volumes": [f"{host}:/output"]}}), encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("forward_slashes", [False, True])
async def test_secondary_docker_media_resolves_via_its_own_mounts(tmp_path, monkeypatch, forward_slashes):
    root = tmp_path / "hermes"
    default_out, public_out = root / "cache" / "output", root / "profiles" / "public" / "cache" / "output"
    _docker_profile(root, default_out, forward_slashes=forward_slashes)
    _docker_profile(root / "profiles" / "public", public_out, forward_slashes=forward_slashes)
    # The launch process carries the DEFAULT profile's bridged terminal env.
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{default_out}:/output"]))
    (public_out / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"public" * 8)
    (default_out / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"decoy!" * 8)  # same name, wrong profile

    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    adapter = _Adapter()
    adapter.gateway_runner = runner
    adapter._keep_typing = _hold_typing

    async def routed_handler(event):  # what _make_profile_message_handler does around _handle_message
        event.source.profile = "public"
        async with _async_profile_runtime_scope(root / "profiles" / "public"):
            return "here MEDIA:/output/pic.png"

    adapter.set_message_handler(routed_handler)
    event = MessageEvent(
        text="pic", message_type=MessageType.TEXT, message_id="m1",
        source=SessionSource(platform=Platform.DISCORD, chat_id="1", chat_type="dm", user_id="u"))
    await adapter._process_message_background(event, build_session_key(event.source, profile="public"))

    assert len(adapter.images) == 1
    assert b"public" in adapter.images[0]


@pytest.mark.parametrize("forward_slashes", [False, True])
def test_mount_parser_separates_host_drive_from_posix_container(tmp_path, monkeypatch, forward_slashes):
    from gateway.platforms import base

    host = tmp_path.as_posix() if forward_slashes else str(tmp_path)
    raw = json.dumps([f"{host}:/output:ro", "named-volume:/ignored", "relative:/ignored"])
    monkeypatch.setattr(base, "_tenv", lambda name, default="": raw if name == "TERMINAL_DOCKER_VOLUMES" else default)

    mounts = base._parse_docker_volume_mounts()

    assert mounts == [(tmp_path.resolve(), PurePosixPath("/output"))]
    assert mounts[0][1].is_absolute()


def test_container_parent_traversal_cannot_escape_mapped_host(tmp_path, monkeypatch):
    from gateway.platforms import base

    mount = tmp_path / "output"
    mount.mkdir()
    (tmp_path / "outside.png").write_bytes(b"not an allowed mapped file")
    monkeypatch.setattr(base, "_parse_docker_volume_mounts", lambda: [(mount, PurePosixPath("/output"))])
    monkeypatch.setattr(base, "_cache_dir_container_mounts", lambda: [])
    monkeypatch.setattr(base, "_docker_persistent_sandbox_roots", lambda *args: [])
    monkeypatch.setattr(base, "_default_docker_workspace_host_roots", lambda *args: [])

    assert base._translate_docker_container_media_path(
        PurePosixPath("/output/../outside.png")
    ) is None


def test_native_absolute_host_media_still_delivers(tmp_path, monkeypatch):
    from gateway.platforms import base

    media = tmp_path / "host-image.png"
    media.write_bytes(b"host image")
    monkeypatch.setattr(base, "_parse_docker_volume_mounts", lambda: [])
    monkeypatch.setattr(base, "_cache_dir_container_mounts", lambda: [])
    monkeypatch.setattr(base, "_docker_persistent_sandbox_roots", lambda *args: [])
    monkeypatch.setattr(base, "_default_docker_workspace_host_roots", lambda *args: [])
    monkeypatch.setattr(base, "_media_delivery_allowed_roots", lambda: [tmp_path])

    assert base.validate_media_delivery_path(str(media)) == str(media.resolve())


@pytest.mark.windows_only
def test_unmapped_container_path_never_uses_host_current_drive(tmp_path, monkeypatch):
    from gateway.platforms import base

    media = tmp_path / "must-not-be-found.png"
    media.write_bytes(b"host file outside any container mount")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(base, "_parse_docker_volume_mounts", lambda: [])
    monkeypatch.setattr(base, "_cache_dir_container_mounts", lambda: [])
    monkeypatch.setattr(base, "_docker_persistent_sandbox_roots", lambda *args: [])
    monkeypatch.setattr(base, "_default_docker_workspace_host_roots", lambda *args: [])
    monkeypatch.setattr(base, "_media_delivery_allowed_roots", lambda: [tmp_path])
    drive_relative = media.as_posix()[len(media.drive):]
    assert not Path(drive_relative).is_absolute()
    assert Path(drive_relative).is_file()  # Windows would otherwise use the current drive.

    assert base.validate_media_delivery_path(drive_relative) is None
