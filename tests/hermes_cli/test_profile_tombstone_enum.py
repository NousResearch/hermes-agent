"""Tombstoned profiles must not appear in Group Chat or cron Bot Chat lists.

``live_profile_names()`` is the shared enumerator for
``list_profile_names()`` and ``HostedRoomService.local_profiles()``. These
tests drive both consumers against a real temporary Hermes home.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cron.scheduler import BOT_CHAT_PLATFORM, cron_delivery_targets
from hermes_cli.profiles import create_profile, delete_profile, list_profile_names
from hermes_constants import live_profile_names, mark_named_profile_deleted
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


def _delete(name: str) -> None:
    with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
        "hermes_cli.profiles._stop_profile_backends"
    ):
        delete_profile(name, yes=True)


def _server():
    return SimpleNamespace(
        _methods={}, _sessions={}, _sessions_lock=threading.Lock()
    )


def _members(*profiles: str) -> list[dict[str, str]]:
    return [
        {
            "member_id": profile,
            "profile": profile,
            "handle": "hermes" if profile == "default" else profile,
        }
        for profile in profiles
    ]


def test_live_profile_names_is_scoped_to_passed_root(tmp_path):
    global_home = Path(tmp_path / "process-home")
    (global_home / "profiles" / "globalone").mkdir(parents=True)

    other = tmp_path / "service-home"
    (other / "profiles" / "ops").mkdir(parents=True)
    (other / "profiles" / ".deleted").mkdir()
    (other / "profiles" / "Not-A-Profile").mkdir()
    worker = other / "profiles" / "worker"
    worker.mkdir()
    mark_named_profile_deleted(worker)

    assert live_profile_names(other) == ["default", "ops"]
    assert "globalone" not in live_profile_names(other)
    assert live_profile_names(global_home) == ["default", "globalone"]


def test_group_chat_and_cron_hide_tombstones(profile_env):
    create_profile("ops", no_alias=True, no_skills=True)
    worker = create_profile("worker", no_alias=True, no_skills=True)
    service = HostedRoomService(_server(), db_path=profile_env / "state.db")

    before = service.create_room(
        room_id="room-before",
        name="Live members",
        members=_members("default", "ops"),
    )
    assert before["room_id"] == "room-before"
    assert service.local_profiles() == ("default", "ops", "worker")
    assert list_profile_names() == ["default", "ops", "worker"]

    _delete("worker")

    assert (profile_env / "profiles" / ".deleted").is_dir()
    assert "worker" not in list_profile_names()
    assert service.local_profiles() == ("default", "ops")
    after_delete = service.create_room(
        room_id="room-after-delete",
        name="Still live members",
        members=_members("default", "ops"),
    )
    assert after_delete["room_id"] == "room-after-delete"

    worker.mkdir()
    (worker / "state.db").write_bytes(b"")

    names = list_profile_names()
    assert names == ["default", "ops"]
    assert service.local_profiles() == ("default", "ops")
    target_ids = {target["id"] for target in cron_delivery_targets()}
    assert f"{BOT_CHAT_PLATFORM}:ops" in target_ids
    assert f"{BOT_CHAT_PLATFORM}:default" in target_ids
    assert f"{BOT_CHAT_PLATFORM}:worker" not in target_ids

    after_stale = service.create_room(
        room_id="room-after-stale",
        name="Stale shell still hidden",
        members=_members("default", "ops"),
    )
    assert after_stale["room_id"] == "room-after-stale"
