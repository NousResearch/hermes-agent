"""Hard-link-published files are still created on a Python without ``os.link`` (Termux).

Termux's bionic CPython omits ``os.link``, so the lookup raises ``AttributeError``, which the
``except OSError`` exclusive-create fallbacks never saw: the first inbound turn died creating
``log_hash_salt`` (#158513), and RoomLink secrets, cron cancel markers and update markers failed
the same way.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from cron.cancellation import _publish_first
from gateway import hosted_room_peer
from gateway.platforms.helpers import _read_or_create_log_hash_salt
from hermes_cli.update_cmd_zip import _link_or_copy_artifact
from hermes_cli.update_lock import _publish_exclusive


def _salt(home):
    return home / "log_hash_salt", _read_or_create_log_hash_salt(home)


def _room_secret(home):
    hosted_room_peer._gateway_room_grant_secret_for_home(str(home))
    path = home / hosted_room_peer._ROOM_GRANT_SECRET_FILE
    return path, path.read_bytes()


def _cancel_marker(home):
    _publish_first(home / "cancel.marker", b"stopped")
    return home / "cancel.marker", b"stopped"


def _update_marker(home):
    assert _publish_exclusive(home / "update.marker", b"pid 1") is True
    return home / "update.marker", b"pid 1"


def _artifact(home):
    (home / "release.bin").write_bytes(b"artifact")
    _link_or_copy_artifact(str(home / "release.bin"), str(home / "staged.bin"))
    return home / "staged.bin", b"artifact"


@pytest.mark.parametrize("publish", [_salt, _room_secret, _cancel_marker, _update_marker, _artifact])
def test_publish_without_os_link_uses_the_fallback(publish, tmp_path, monkeypatch):
    monkeypatch.delattr(os, "link", raising=False)

    path, expected = publish(tmp_path)

    assert expected and path.read_bytes() == expected


def _slowed(real):
    def call(*args, **kwargs):  # widen the open→write and check→publish windows a racer could hit
        time.sleep(0.05)  # health: allow HX013 -- the widened window IS the race under test
        return real(*args, **kwargs)

    return call


def _first_room_secret(home):
    mint = hosted_room_peer._gateway_room_grant_secret_for_home.__wrapped__
    return mint(str(home)), home / hosted_room_peer._ROOM_GRANT_SECRET_FILE


def _first_salt(home):
    return _read_or_create_log_hash_salt(home), home / "log_hash_salt"


@pytest.mark.platforms("any")  # Windows publishes with a no-clobber rename, POSIX under a flock
@pytest.mark.parametrize("first_use_of", [_first_room_secret, _first_salt])
def test_concurrent_first_use_without_os_link_agrees_on_one_secret(first_use_of, tmp_path, monkeypatch):
    """Racing first users never read a partial secret or salt and all end up with the same one."""
    monkeypatch.delattr(os, "link", raising=False)
    for name in ("fdopen", "replace", "rename"):
        monkeypatch.setattr(os, name, _slowed(getattr(os, name)))
    barrier = threading.Barrier(16)
    results, errors = [], []

    def first_use():
        barrier.wait()
        try:
            results.append(first_use_of(tmp_path)[0])
        except Exception as exc:  # a partial read: HostedRoomGrantError, or ValueError for the salt
            errors.append(exc)

    threads = [threading.Thread(target=first_use) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(results)) == 1
    assert len(first_use_of(tmp_path)[1].read_bytes()) == 32
