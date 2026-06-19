from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import time
from pathlib import Path

import pytest

from plugins.context_engine.lcm import storage_security
from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine
from plugins.context_engine.lcm.store import MessageStore
from plugins.context_engine.lcm.tools import lcm_doctor, lcm_expand, lcm_status


def _joined(*parts: str) -> str:
    return "".join(parts)


def _secure_config(profile_dir: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(profile_dir / "lcm.db"),
        sensitive_patterns_enabled=True,
        sensitive_patterns=["all"],
        encryption_enabled=True,
        encryption_key_path=str(profile_dir / "lcm-row.key"),
        retention_ttl_days=14,
        retention_max_bytes=1024 * 1024 * 1024,
    )


def _secret_value() -> str:
    return _joined("sk", "-", "proj", "-", "e" * 48)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_encrypted_mode_hides_raw_secret_from_strings_and_expand_still_reads(tmp_path):
    if not storage_security.aead_available():
        pytest.skip("cryptography AESGCM is not available in this environment")
    profile_dir = tmp_path / "profile"
    cfg = _secure_config(profile_dir)
    engine = LCMEngine(config=cfg, hermes_home=str(profile_dir))
    engine.on_session_start("session-storage", conversation_id="conversation-storage")

    secret = _secret_value()
    store_id = engine._store.append(
        "session-storage",
        {"role": "user", "content": "please remember api token " + secret},
        token_estimate=12,
    )

    expanded = json.loads(lcm_expand({"store_id": store_id, "max_tokens": 1000}, engine=engine))
    assert expanded["source_type"] == "raw_message"
    assert "LCM sensitive redaction" in expanded["content"]
    assert secret not in expanded["content"]

    db_path = profile_dir / "lcm.db"
    conn = sqlite3.connect(str(db_path))
    try:
        content, search_content = conn.execute(
            "SELECT content, search_content FROM messages WHERE store_id = ?",
            (store_id,),
        ).fetchone()
        sqlite_dump = "\n".join(conn.iterdump())
    finally:
        conn.close()

    assert isinstance(content, str) and content.startswith(storage_security.AEAD_PREFIX)
    assert secret not in content
    assert secret not in (search_content or "")
    assert secret not in sqlite_dump

    strings_bin = shutil.which("strings")
    if strings_bin:
        strings_out = subprocess.run(
            [strings_bin, str(db_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert secret not in strings_out

    engine._close_storage()


def test_storage_permissions_are_private_for_db_and_sidecars(tmp_path):
    profile_dir = tmp_path / "profile"
    cfg = _secure_config(profile_dir)
    store = MessageStore(profile_dir / "lcm.db", ingest_protection_config=cfg, hermes_home=str(profile_dir))
    store.append("session-perms", {"role": "user", "content": "hello"}, token_estimate=1)

    assert _mode(profile_dir) == 0o700
    for path in [profile_dir / "lcm.db", profile_dir / "lcm.db-wal", profile_dir / "lcm.db-shm"]:
        if path.exists():
            assert _mode(path) == 0o600, path

    store.close()


def test_encryption_enabled_fails_loud_when_aead_dependency_missing(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profile"
    cfg = _secure_config(profile_dir)
    monkeypatch.setattr(storage_security, "AESGCM", None)
    with pytest.raises(RuntimeError, match="cryptography.*AESGCM.*LCM encryption"):
        MessageStore(profile_dir / "lcm.db", ingest_protection_config=cfg, hermes_home=str(profile_dir))


def test_plaintext_path_checker_requires_explicit_unsynced_marker(tmp_path):
    unmarked = tmp_path / "plain" / "lcm.db"
    marked_dir = tmp_path / "plain-ok"
    marked = marked_dir / "lcm.db"
    marked_dir.mkdir()
    (marked_dir / storage_security.UNSYNCED_PATH_MARKER).write_text("local only\n")
    synced_dir = tmp_path / "iCloud Drive" / "plain-ok"
    synced_dir.mkdir(parents=True)
    (synced_dir / storage_security.UNSYNCED_PATH_MARKER).write_text("local only\n")

    assert not storage_security.plaintext_path_policy(unmarked)["allowed"]
    assert storage_security.plaintext_path_policy(marked)["allowed"]
    synced_policy = storage_security.plaintext_path_policy(synced_dir / "lcm.db")
    assert not synced_policy["allowed"]
    assert synced_policy["synced_path_detected"]


def test_status_and_doctor_surface_retention_size_and_oldest_row(tmp_path):
    profile_dir = tmp_path / "profile"
    cfg = _secure_config(profile_dir)
    engine = LCMEngine(config=cfg, hermes_home=str(profile_dir))
    engine.on_session_start("session-retention", conversation_id="conversation-retention")
    engine._store.append("session-retention", {"role": "user", "content": "retention probe"}, token_estimate=2)
    old_ts = time.time() - (15 * 86400)
    engine._store._conn.execute("UPDATE messages SET timestamp = ?", (old_ts,))
    engine._store._conn.commit()

    status = json.loads(lcm_status({}, engine=engine))
    retention = status["storage_retention"]
    assert retention["ttl_days"] == 14
    assert retention["max_bytes"] == 1024 * 1024 * 1024
    assert retention["total_size_bytes"] > 0
    assert retention["oldest_row_at"] == pytest.approx(old_ts)
    assert retention["oldest_row_age_days"] >= 14

    doctor = json.loads(lcm_doctor({}, engine=engine))
    retention_check = next(check for check in doctor["checks"] if check["check"] == "retention_policy")
    assert retention_check["detail"]["total_size_bytes"] == retention["total_size_bytes"]
    assert retention_check["detail"]["oldest_row_at"] == pytest.approx(old_ts)

    engine._close_storage()
