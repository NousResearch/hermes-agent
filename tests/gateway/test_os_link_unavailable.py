"""Regression for #140080: hard-link publish helpers must survive a Python without ``os.link``.

Termux's bionic CPython build omits ``os.link`` entirely, so the failure is an ``AttributeError``
raised at attribute lookup — not the ``OSError`` a filesystem without hard links raises. The
handlers used to catch only ``OSError``, so the documented exclusive-create fallbacks were dead
code on that interpreter.
"""

from __future__ import annotations

import hashlib
import hmac
import os

import pytest

from cron.cancellation import _publish_first
from gateway import hosted_room_peer
from gateway.platforms.helpers import _read_or_create_log_hash_salt


@pytest.fixture
def without_os_link(monkeypatch):
    monkeypatch.delattr(os, "link", raising=False)
    assert not hasattr(os, "link")


def test_log_hash_salt_falls_back_to_exclusive_create(tmp_path, without_os_link):
    salt = _read_or_create_log_hash_salt(tmp_path)

    assert len(salt) == 32
    assert (tmp_path / "log_hash_salt").read_bytes() == salt


def test_room_grant_secret_falls_back_to_exclusive_create(tmp_path, without_os_link):
    secret = hosted_room_peer._gateway_room_grant_secret_for_home(str(tmp_path))

    stored = (tmp_path / hosted_room_peer._ROOM_GRANT_SECRET_FILE).read_bytes()
    assert len(stored) == 32
    assert secret == hmac.new(
        stored, b"hermes-hosted-room-installation-grant-v1", hashlib.sha256
    ).digest()


def test_cancel_marker_falls_back_to_exclusive_create(tmp_path, without_os_link):
    marker = tmp_path / "cancel.marker"

    _publish_first(marker, b"operator stopped the run")

    assert marker.read_bytes() == b"operator stopped the run"
