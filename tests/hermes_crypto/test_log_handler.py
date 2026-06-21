"""Tests for encrypting log rotation and read-log helpers."""

from __future__ import annotations

import logging

from hermes_constants import get_hermes_home
from hermes_crypto import detect, logs_encryption_active, migrate
from hermes_crypto.log_handler import (
    EncryptingRotatingFileHandler,
    build_rotating_handler,
    read_log_text,
)

FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}


def _enable_with_log_encryption() -> None:
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    migrate._set_config("security.encryption.encrypt_logs", True)


def test_logs_encryption_active_respects_config():
    assert not logs_encryption_active()
    _enable_with_log_encryption()
    assert logs_encryption_active()


def test_build_rotating_handler_selects_encrypting_class():
    _enable_with_log_encryption()
    log_path = get_hermes_home() / "logs" / "agent.log"
    handler = build_rotating_handler(log_path, max_bytes=64, backup_count=2)
    assert isinstance(handler, EncryptingRotatingFileHandler)


def test_build_rotating_handler_plain_when_flag_off():
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    log_path = get_hermes_home() / "logs" / "agent.log"
    handler = build_rotating_handler(log_path, max_bytes=64, backup_count=2)
    from logging.handlers import RotatingFileHandler

    assert type(handler) is RotatingFileHandler


def test_rotation_encrypts_backup_leaves_live_plaintext():
    _enable_with_log_encryption()
    log_path = get_hermes_home() / "logs" / "agent.log"
    handler = EncryptingRotatingFileHandler(
        log_path,
        maxBytes=80,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("hermes.test.log_rotation")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        for idx in range(30):
            logger.info("line-%s with enough text to trigger rotation soon", idx)
        handler.flush()
        handler.close()

        live = log_path.read_bytes()
        backup1 = log_path.with_name("agent.log.1").read_bytes()
        backup2 = log_path.with_name("agent.log.2").read_bytes()
        assert not detect.is_encrypted(live)
        assert detect.is_encrypted(backup1)
        assert detect.is_encrypted(backup2)
        assert b"line-29" in live
        assert "line-28" in read_log_text(log_path.with_name("agent.log.1"))
        assert "line-27" in read_log_text(log_path.with_name("agent.log.2"))
    finally:
        logger.handlers.clear()


def test_read_log_text_passes_through_plaintext():
    log_path = get_hermes_home() / "logs" / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # AGENTS.md §6.1: write_bytes avoids the Windows text-mode CRLF
    # translation that would put "\r\n" on disk and break the equality
    # check against the "\n"-terminated literal below.
    log_path.write_bytes(b"plain log line\n")
    assert read_log_text(log_path) == "plain log line\n"
