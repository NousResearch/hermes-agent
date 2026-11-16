"""Gateway logs at INFO+ carry no verbatim user content (#46006): `hermes debug share` uploads them."""

import hashlib
import hmac
import logging
import os
import stat

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import helpers
from gateway.platforms.helpers import hash_for_log
from gateway.platforms.webhook import WebhookAdapter

_SECRET_TEXT = "my bank PIN is 4321, call me at alice@example.com"


def test_hash_for_log_is_stable_short_and_hides_the_value():
    digest = hash_for_log("room:!secret-room-id")
    assert digest == hash_for_log("room:!secret-room-id") != hash_for_log("room:!other")
    assert "secret" not in digest and len(digest) <= 12
    assert hash_for_log(None) == hash_for_log("") == "<none>"


@pytest.mark.asyncio
@pytest.mark.parametrize("deliver", ["_deliver_to", "_direct_deliver"])
async def test_webhook_log_delivery_keeps_the_body_out_of_info(caplog, deliver):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"port": 0, "routes": {}}))
    caplog.set_level(logging.DEBUG, logger="gateway.platforms.webhook")
    args = ({"deliver": "log"}, "webhook:x", _SECRET_TEXT) if deliver == "_deliver_to" else (
        _SECRET_TEXT, {"deliver": "log"})
    result = await getattr(adapter, deliver)(*args)
    assert result.success
    info = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert info and not any("PIN" in m or "alice@" in m for m in info), info
    assert any(_SECRET_TEXT in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


def _fresh_home(tmp_path, monkeypatch, name):
    home = tmp_path / name
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(helpers, "_LOG_HASH_SALTS", {})
    return home


def test_hash_for_log_is_salted_per_install(tmp_path, monkeypatch):
    phone = "+15551234567"
    home_a = _fresh_home(tmp_path, monkeypatch, "a")
    digest_a = hash_for_log(phone)
    assert digest_a == hash_for_log(phone)
    assert digest_a != "h:" + hashlib.sha256(phone.encode()).hexdigest()[:10]
    assert len((home_a / "log_hash_salt").read_bytes()) == 32
    monkeypatch.setattr(helpers, "_LOG_HASH_SALTS", {})  # a new process on the same install
    assert hash_for_log(phone) == digest_a
    _fresh_home(tmp_path, monkeypatch, "b")
    assert hash_for_log(phone) != digest_a


def test_hash_for_log_salt_persists_without_hard_links(tmp_path, monkeypatch):
    # Android (SELinux) and exFAT refuse link(): the salt must still be written once and reused.
    home = _fresh_home(tmp_path, monkeypatch, "nolink")

    def refuse_link(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "link", refuse_link)
    digest = hash_for_log("+155****1111")
    assert len((home / "log_hash_salt").read_bytes()) == 32
    monkeypatch.setattr(helpers, "_LOG_HASH_SALTS", {})  # a new process on the same install
    assert hash_for_log("+155****1111") == digest
    assert not list(home.glob(".log_hash_salt.*"))  # no temp file left behind


def test_hash_for_log_salt_failure_is_not_retried_per_call(tmp_path, monkeypatch):
    _fresh_home(tmp_path, monkeypatch, "broken")
    attempts = []

    def fail(home):
        attempts.append(home)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(helpers, "_read_or_create_log_hash_salt", fail)
    assert hash_for_log("a") == hash_for_log("a")
    hash_for_log("b")
    assert len(attempts) == 1


def test_hash_for_log_salt_without_hard_links_reads_the_first_creators_salt(tmp_path, monkeypatch):
    home = _fresh_home(tmp_path, monkeypatch, "nolink-race")
    winner = bytes(range(32))

    def lose_race(*_args, **_kwargs):  # another process publishes its salt before our exclusive create
        (home / "log_hash_salt").write_bytes(winner)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "link", lose_race)
    assert hash_for_log("x") == "h:" + hmac.new(winner, b"x", hashlib.sha256).hexdigest()[:10]


@pytest.mark.platforms("posix")  # POSIX permission bits
def test_hash_for_log_salt_without_hard_links_is_private(tmp_path, monkeypatch):
    home = _fresh_home(tmp_path, monkeypatch, "nolink-mode")

    def refuse_link(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "link", refuse_link)
    hash_for_log("x")
    assert stat.S_IMODE((home / "log_hash_salt").stat().st_mode) == 0o600


@pytest.mark.platforms("posix")  # POSIX permission bits
def test_hash_for_log_salt_is_private_and_unwritable_home_falls_back(tmp_path, monkeypatch):
    home = _fresh_home(tmp_path, monkeypatch, "posix")
    hash_for_log("x")
    assert stat.S_IMODE((home / "log_hash_salt").stat().st_mode) == 0o600
    if os.geteuid() == 0:
        return  # root ignores directory modes; the fallback half needs a real permission error
    home = _fresh_home(tmp_path, monkeypatch, "ro")
    home.chmod(0o500)
    try:
        digest = hash_for_log("alice@example.com")
        assert digest == hash_for_log("alice@example.com") and "alice" not in digest
        assert not (home / "log_hash_salt").exists()
    finally:
        home.chmod(0o700)
