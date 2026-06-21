"""Tests for the enable / disable / status migration orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from hermes_crypto import audit, detect, envelope, is_encryption_enabled, keystore, migrate
from hermes_crypto.errors import HermesCryptoError, KeystoreError
from hermes_crypto.fileio import harden_dir

FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}


def _seed_credentials() -> tuple:
    """Write a plaintext .env and auth.json into the test HERMES_HOME."""
    home = get_hermes_home()
    env = home / ".env"
    auth = home / "auth.json"
    env.write_text("OPENAI_API_KEY=sk-test-secret\nFOO=bar\n", encoding="utf-8")
    auth.write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}),
        encoding="utf-8",
    )
    return env, auth


def test_enable_encrypts_credential_files():
    env, auth = _seed_credentials()
    result = migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)

    assert ".env" in result.encrypted_files
    assert "auth.json" in result.encrypted_files
    assert detect.is_encrypted(env.read_bytes())
    assert detect.is_encrypted(auth.read_bytes())
    assert keystore.keystore_exists()


def test_enable_flips_config_flag_last():
    _seed_credentials()
    from hermes_crypto import is_encryption_enabled

    assert not is_encryption_enabled()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    assert is_encryption_enabled()


def test_enable_restores_backup_on_post_write_verify_failure(monkeypatch):
    """Post-write verify failure must restore the plaintext backup and abort."""
    home = get_hermes_home()
    auth = home / "auth.json"
    original = json.dumps({"version": 1, "providers": {}, "credential_pool": {}})
    auth.write_text(original, encoding="utf-8")

    real_decrypt = envelope.decrypt
    calls = [0]

    def fail_post_write_verify(data, dek):
        calls[0] += 1
        if calls[0] == 2:
            return b"tampered-roundtrip"
        return real_decrypt(data, dek)

    monkeypatch.setattr(migrate.envelope, "decrypt", fail_post_write_verify)

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    with pytest.raises(HermesCryptoError, match="post-write verification failed"):
        migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)

    assert auth.read_text(encoding="utf-8") == original
    assert not detect.is_encrypted(auth.read_bytes())
    assert not is_encryption_enabled()
    events = audit.read_recent(10)
    assert any(e.get("activity") == audit.KEYSTORE_CREATED for e in events)
    assert not any(e.get("activity") == audit.ENCRYPTION_ENABLED for e in events)


def test_enable_creates_plaintext_backups():
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    backups = list(keystore.backup_dir().glob("*.bak"))
    assert any(b.name.startswith(".env") for b in backups)
    assert any(b.name.startswith("auth.json") for b in backups)
    # Backups are the original plaintext, not envelopes.
    for backup in backups:
        assert not detect.is_encrypted(backup.read_bytes())


def test_enable_refuses_when_already_enabled():
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    with pytest.raises(HermesCryptoError):
        migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)


def test_enable_after_interrupted_enable_points_to_disable():
    """A keystore without the config flag = a crashed previous enable().

    The guard must say exactly how to recover (disable rolls back to
    plaintext), not claim encryption "looks enabled already".
    """
    _seed_credentials()
    # Simulate a crash between keystore creation and the config flip: run a
    # full enable, then force the enabled flag back off the way enable()
    # would have left it had it died before the final write.
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    migrate._set_config("security.encryption.enabled", False)

    with pytest.raises(HermesCryptoError, match="interrupted"):
        migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)

    # disable() recovers from this state.
    migrate.disable(passphrase="pw")
    assert not keystore.keystore_exists()


def test_disable_restores_plaintext():
    env, auth = _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    assert detect.is_encrypted(env.read_bytes())

    migrate.disable()
    assert not detect.is_encrypted(env.read_bytes())
    assert not detect.is_encrypted(auth.read_bytes())
    assert b"OPENAI_API_KEY=sk-test-secret" in env.read_bytes()
    assert json.loads(auth.read_text(encoding="utf-8"))["version"] == 1
    assert not keystore.keystore_exists()


def test_disable_keeps_config_when_destroy_fails(monkeypatch):
    env, _auth = _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    real_unlink = Path.unlink

    def fail_keystore_unlink(self, *args, **kwargs):
        if self == keystore.keystore_path():
            raise OSError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_keystore_unlink)
    with pytest.raises(KeystoreError):
        migrate.disable()

    assert is_encryption_enabled()
    assert keystore.keystore_exists()
    assert not detect.is_encrypted(env.read_bytes())


def test_status_reflects_state():
    _seed_credentials()
    before = migrate.status()
    assert before["enabled"] is False
    assert before["keystore_exists"] is False
    assert {"name": ".env", "state": "plaintext"} in before["files"]

    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    after = migrate.status()
    assert after["enabled"] is True
    assert after["primary_slot"] == "passphrase"
    assert {"name": ".env", "state": "encrypted"} in after["files"]


def test_enable_skips_missing_files():
    # Only .env exists — auth.json absent. enable() must not fail.
    (get_hermes_home() / ".env").write_text("KEY=value\n", encoding="utf-8")
    result = migrate.enable("keyfile", argon2_params=FAST_ARGON2)
    assert result.encrypted_files == [".env"]


def test_backup_summary_empty():
    summary = migrate.backup_summary()
    # Plaintext / ciphertext split added by hardening work.
    assert summary == {
        "count": 0,
        "total_bytes": 0,
        "oldest_days": None,
        "plaintext_count": 0,
        "ciphertext_count": 0,
    }


def test_backup_summary_after_enable():
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    summary = migrate.backup_summary()
    assert summary["count"] >= 2
    assert summary["total_bytes"] > 0
    assert summary["oldest_days"] == 0


def test_backup_names_do_not_collide_within_same_second():
    env, _auth = _seed_credentials()

    first = migrate._backup(env)
    second = migrate._backup(env)

    assert first != second
    assert first.is_file()
    assert second.is_file()
    assert migrate._parse_backup_stamp(first.name) is not None
    assert migrate._parse_backup_stamp(second.name) is not None


def test_clean_backups_deletes_all():
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    assert migrate.backup_summary()["count"] >= 2

    result = migrate.clean_backups()
    assert result["deleted_count"] >= 2
    assert result["kept_count"] == 0
    assert result["errors"] == []
    assert migrate.backup_summary()["count"] == 0


def test_clean_backups_respects_older_than():
    backup_dir = keystore.backup_dir()
    harden_dir(backup_dir)
    old = backup_dir / ".env.20200101T000000.bak"
    new = backup_dir / f".env.{datetime.now().strftime('%Y%m%dT%H%M%S')}.bak"
    old.write_text("old secret\n", encoding="utf-8")
    new.write_text("new secret\n", encoding="utf-8")

    result = migrate.clean_backups(older_than_days=30)
    assert result["deleted_count"] == 1
    assert old.name in result["deleted"]
    assert not old.exists()
    assert new.exists()
    assert migrate.backup_summary()["count"] == 1


def test_clean_backups_audits_success():
    backup_dir = keystore.backup_dir()
    harden_dir(backup_dir)
    path = backup_dir / ".env.20200101T000000.bak"
    path.write_text("secret\n", encoding="utf-8")
    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    migrate.clean_backups()
    events = audit.read_recent(5)
    assert any(e.get("activity") == audit.BACKUPS_CLEANED for e in events)


def test_status_includes_backup_count():
    _seed_credentials()
    before = migrate.status()
    # Plaintext / ciphertext split added by hardening work.
    assert before["backups"] == {
        "count": 0,
        "total_bytes": 0,
        "oldest_days": None,
        "plaintext_count": 0,
        "ciphertext_count": 0,
    }

    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    after = migrate.status()
    assert after["backups"]["count"] >= 2
    assert after["backups"]["total_bytes"] > 0
    # All backups made during enable are plaintext credentials.
    assert after["backups"]["plaintext_count"] >= 2
    assert after["backups"]["ciphertext_count"] == 0


def test_enable_aborts_when_concurrent_hermes_detected(monkeypatch):
    _seed_credentials()
    monkeypatch.setattr(
        migrate,
        "_detect_concurrent_hermes_instances",
        lambda: [{"pid": 4242, "cmdline": "hermes gateway"}],
    )
    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    with pytest.raises(HermesCryptoError, match="another Hermes process is still running"):
        migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)

    assert not keystore.keystore_exists()
    assert not is_encryption_enabled()
    events = audit.read_recent(5)
    assert any(
        e.get("activity") == audit.MIGRATION_BLOCKED and e.get("outcome") == audit.FAILURE
        for e in events
    )


def test_enable_databases_aborts_when_concurrent_hermes_detected(monkeypatch):
    """Regression guard: ``--databases`` flow must honour the concurrent-Hermes
    check before any ``dbcrypt.encrypt_database`` call. Locks in the
    caller-responsibility contract documented on the dbcrypt primitives.
    """
    from unittest.mock import Mock

    _seed_credentials()
    monkeypatch.setattr(
        migrate,
        "_detect_concurrent_hermes_instances",
        lambda: [{"pid": 12345, "cmdline": "hermes gateway"}],
    )
    encrypt_db_shim = Mock()
    monkeypatch.setattr(migrate.dbcrypt, "encrypt_database", encrypt_db_shim)

    with pytest.raises(HermesCryptoError, match="another Hermes process is still running"):
        migrate.enable(
            "passphrase",
            passphrase="pw",
            argon2_params=FAST_ARGON2,
            encrypt_databases=True,
        )

    encrypt_db_shim.assert_not_called()
    assert not keystore.keystore_exists()
    assert not is_encryption_enabled()


def test_enable_force_bypasses_concurrent_hermes_check(monkeypatch):
    _seed_credentials()
    monkeypatch.setattr(
        migrate,
        "_detect_concurrent_hermes_instances",
        lambda: [{"pid": 4242, "cmdline": "hermes gateway"}],
    )

    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        force=True,
    )
    assert keystore.keystore_exists()
    assert is_encryption_enabled()


def test_enable_proceeds_when_no_concurrent_hermes(monkeypatch):
    _seed_credentials()
    monkeypatch.setattr(migrate, "_detect_concurrent_hermes_instances", lambda: [])

    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    assert keystore.keystore_exists()


def test_disable_aborts_when_concurrent_hermes_detected(monkeypatch):
    _seed_credentials()
    monkeypatch.setattr(migrate, "_detect_concurrent_hermes_instances", lambda: [])
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    monkeypatch.setattr(
        migrate,
        "_detect_concurrent_hermes_instances",
        lambda: [{"pid": 4242, "cmdline": "python -m gateway.server"}],
    )

    with pytest.raises(HermesCryptoError, match="disable encryption"):
        migrate.disable()

    assert keystore.keystore_exists()
    assert is_encryption_enabled()


def test_is_hermes_runtime_cmdline_excludes_encrypt_cli():
    assert not migrate._is_hermes_runtime_cmdline(["hermes", "encrypt", "enable"])
    assert migrate._is_hermes_runtime_cmdline(["hermes", "gateway"])
    assert migrate._is_hermes_runtime_cmdline(["python", "-m", "gateway.server"])


def test_is_hermes_runtime_cmdline_rejects_path_false_positives():
    assert not migrate._is_hermes_runtime_cmdline(
        ["/home/gateway-user/bin/python", "worker.py"]
    )
    assert not migrate._is_hermes_runtime_cmdline(["/opt/agent-tools/run.sh"])
    assert migrate._is_hermes_runtime_cmdline(["python", "-m", "hermes", "gateway"])
    assert migrate._is_hermes_runtime_cmdline(["/usr/local/bin/hermes-agent"])


def test_is_hermes_runtime_cmdline_still_catches_real_hermes_gateway():
    """Regression: the fix must not regress real-runtime detection."""
    # Direct binary invocations.
    assert migrate._is_hermes_runtime_cmdline(["hermes", "gateway"])
    assert migrate._is_hermes_runtime_cmdline(["/usr/local/bin/hermes", "agent"])
    assert migrate._is_hermes_runtime_cmdline([r"C:\Hermes\hermes.exe", "gateway"])
    # Module invocations.
    assert migrate._is_hermes_runtime_cmdline(
        ["python", "-m", "gateway.run"]
    )
    assert migrate._is_hermes_runtime_cmdline(
        ["python", "-m", "agent.entry"]
    )
    # Standalone hermes-agent binary (if a fork ships one).
    assert migrate._is_hermes_runtime_cmdline(
        ["/opt/hermes/hermes-agent", "gateway"]
    )
    # Special runtimes.
    assert migrate._is_hermes_runtime_cmdline(
        ["python", "-m", "tui_gateway"]
    )


# ── clean-backups must classify plaintext vs ciphertext ──
#
# _iter_backup_files() returns every *.bak, including the rekey-run backups
# produced by ``full_rekey --keep-backups``. Those are ciphertext sealed
# under the old DEK, often intentionally kept for forensic recovery.
# clean_backups must NOT delete them unless --include-ciphertext is passed.


def test_clean_backups_skips_ciphertext_by_default():
    """Regression: ciphertext rekey-run backups stay put unless the
    operator opts in via include_ciphertext=True.
    """
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    backup_dir = keystore.backup_dir()

    # Drop a synthetic ciphertext backup next to the real plaintext ones.
    fake_ct = backup_dir / "fake-rekey-run.20200101T000000.bak"
    fake_ct.write_bytes(b"HRMSENC\x01" + b"\x00" * 32)  # envelope-shaped header

    summary = migrate.backup_summary()
    assert summary["ciphertext_count"] >= 1
    assert summary["plaintext_count"] >= 1

    result = migrate.clean_backups()  # include_ciphertext defaults to False
    assert fake_ct.is_file(), "ciphertext backup must NOT be deleted by default"
    assert result["skipped_ciphertext"] >= 1
    # Plaintext backups WERE deleted in the same run.
    assert result["deleted_count"] >= 1


def test_clean_backups_include_ciphertext_flag_deletes_both():
    """Regression: --include-ciphertext / include_ciphertext=True
    removes the rekey-run ciphertext backups alongside plaintext ones.
    """
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    backup_dir = keystore.backup_dir()

    fake_ct = backup_dir / "fake-rekey-run.20200101T000000.bak"
    fake_ct.write_bytes(b"HRMSENC\x01" + b"\x00" * 32)

    result = migrate.clean_backups(include_ciphertext=True)
    assert not fake_ct.exists(), "ciphertext backup should be deleted with the flag"
    assert result["skipped_ciphertext"] == 0


def test_is_hermes_runtime_cmdline_still_catches_real_runtime_from_checkout_path():
    """Regression: a real `hermes gateway` invocation running from a
    hermes-agent-* checkout MUST still be detected. The path component
    shouldn't matter; the command itself should.
    """
    assert migrate._is_hermes_runtime_cmdline(
        [
            r"C:\Hermes\hermes-agent-main\.venv\Scripts\python.exe",
            "-m",
            "gateway.run",
        ]
    )
    assert migrate._is_hermes_runtime_cmdline(
        [
            "/opt/hermes/hermes-agent-main/.venv/bin/hermes",
            "gateway",
        ]
    )


def test_parse_pid_command_lines_splits_tab_separated_output():
    my_pid = 1000
    text = (
        "1000\tpython -m gateway.server\n"
        "4242\tpython -m gateway.server\n"
        "4243\thermes gateway\n"
        "4244\thermes encrypt enable\n"
        "notab\n"
        "abc\tpython -m gateway.server\n"
        "4245\t\n"
        "\n"
    )
    result = migrate._parse_pid_command_lines(text, my_pid)
    assert result == [
        {"pid": 4242, "cmdline": "python -m gateway.server"},
        {"pid": 4243, "cmdline": "hermes gateway"},
    ]


def test_detect_concurrent_hermes_ps_audits_when_subprocess_fails(monkeypatch):
    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    def _raise(*_args, **_kwargs):
        raise OSError("ps unavailable")

    monkeypatch.setattr(migrate.subprocess, "run", _raise)
    result = migrate._detect_concurrent_hermes_ps(1000)
    assert result and result[0].get("enumeration_unavailable") is True

    events = audit.read_recent(5)
    assert any(
        e.get("activity") == audit.MIGRATION_ENUMERATION_UNAVAILABLE
        and e.get("outcome") == audit.INFO
        and e.get("detail", {}).get("reason") == "enumeration_unavailable"
        and e.get("detail", {}).get("method") == "ps"
        for e in events
    )


def test_detect_concurrent_hermes_ps_audits_when_subprocess_nonzero(monkeypatch):
    import subprocess

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    monkeypatch.setattr(
        migrate.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 1, "", ""),
    )
    result = migrate._detect_concurrent_hermes_ps(1000)
    assert result and result[0].get("enumeration_unavailable") is True

    events = audit.read_recent(5)
    assert any(
        e.get("activity") == audit.MIGRATION_ENUMERATION_UNAVAILABLE
        and e.get("detail", {}).get("method") == "ps"
        for e in events
    )


def test_detect_concurrent_hermes_procfs_audits_when_proc_missing(monkeypatch):
    from unittest.mock import MagicMock

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    mock_proc = MagicMock()
    mock_proc.is_dir.return_value = False

    real_path = migrate.Path

    def fake_path(value):
        if value == "/proc":
            return mock_proc
        return real_path(value)

    monkeypatch.setattr(migrate, "Path", fake_path)
    result = migrate._detect_concurrent_hermes_procfs(1000)
    assert result and result[0].get("enumeration_unavailable") is True

    events = audit.read_recent(5)
    assert any(
        e.get("activity") == audit.MIGRATION_ENUMERATION_UNAVAILABLE
        and e.get("detail", {}).get("method") == "procfs"
        for e in events
    )


def test_enable_requires_force_when_enumeration_unavailable(monkeypatch):
    _seed_credentials()
    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    def _unavailable(*_args, **_kwargs):
        return migrate._enumeration_unavailable("ps")

    monkeypatch.setattr(migrate, "_detect_concurrent_hermes_instances", _unavailable)
    with pytest.raises(HermesCryptoError, match="process enumeration is unavailable"):
        migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)

    assert not keystore.keystore_exists()
    events = audit.read_recent(10)
    assert any(e.get("activity") == audit.MIGRATION_ENUMERATION_UNAVAILABLE for e in events)
    assert any(
        e.get("activity") == audit.MIGRATION_BLOCKED
        and e.get("detail", {}).get("reason") == "enumeration_unavailable"
        for e in events
    )


def test_enable_force_proceeds_when_enumeration_unavailable(monkeypatch):
    _seed_credentials()

    def _unavailable(*_args, **_kwargs):
        return migrate._enumeration_unavailable("ps")

    monkeypatch.setattr(migrate, "_detect_concurrent_hermes_instances", _unavailable)
    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        force=True,
    )

    assert keystore.keystore_exists()


# ─── Session-transcript coverage (Phase 3) ───────────────────────────────────


def _seed_session(name: str = "alpha.jsonl", body: str = '{"line":1}\n') -> Path:
    sessions = get_hermes_home() / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / name
    path.write_text(body, encoding="utf-8")
    return path


def test_enable_with_encrypt_logs_encrypts_existing_sessions():
    _seed_credentials()
    path = _seed_session(body='{"role":"user","content":"hi"}\n')

    result = migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )

    assert any(label.endswith("alpha.jsonl") for label in result.encrypted_sessions)
    assert detect.is_encrypted(path.read_bytes())
    # The encrypt_logs flag is now persisted in config.
    from hermes_crypto import logs_encryption_active

    assert logs_encryption_active()


def test_enable_with_encrypt_logs_skips_live_sessions():
    _seed_credentials()
    from hermes_crypto.session_writer import _open_lockfile, _acquire_exclusive

    path = _seed_session(body='{"role":"user","content":"live"}\n')
    lock_path = path.with_name(path.name + ".lock")
    fd = _open_lockfile(lock_path)
    _acquire_exclusive(fd)
    try:
        result = migrate.enable(
            "passphrase",
            passphrase="pw",
            encrypt_logs=True,
            argon2_params=FAST_ARGON2,
            force=True,
        )
        assert any(label.endswith("alpha.jsonl") for label in result.skipped)
        assert not detect.is_encrypted(path.read_bytes())
    finally:
        from hermes_crypto.session_writer import _drop_session_lock

        _drop_session_lock(fd, path)
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass


def test_enable_with_encrypt_logs_encrypts_existing_rotated_log_segments():
    _seed_credentials()
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    live = log_dir / "agent.log"
    rotated = log_dir / "agent.log.1"
    live.write_bytes(b"live log stays plaintext\n")
    rotated.write_bytes(b"old rotated log\n")

    result = migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )

    assert any(label.endswith("agent.log.1") for label in result.encrypted_logs)
    assert detect.is_encrypted(rotated.read_bytes())
    assert not detect.is_encrypted(live.read_bytes())


def test_enable_without_encrypt_logs_leaves_sessions_plaintext():
    _seed_credentials()
    path = _seed_session()

    result = migrate.enable(
        "passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True
    )
    assert result.encrypted_sessions == []
    assert not detect.is_encrypted(path.read_bytes())


def test_disable_decrypts_sessions_regardless_of_flag_state():
    _seed_credentials()
    path = _seed_session(body='{"line":"secret"}\n')

    migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )
    assert detect.is_encrypted(path.read_bytes())

    migrate.disable(force=True)
    assert not detect.is_encrypted(path.read_bytes())
    assert '"line":"secret"' in path.read_text(encoding="utf-8")


def test_disable_restores_backup_on_post_write_verify_failure(monkeypatch):
    env, _auth = _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    assert detect.is_encrypted(env.read_bytes())

    real_atomic_write_private = migrate.atomic_write_private

    def _write_wrong_bytes(path, data, *args, **kwargs):
        if path == env:
            return real_atomic_write_private(path, b"tampered", *args, **kwargs)
        return real_atomic_write_private(path, data, *args, **kwargs)

    monkeypatch.setattr(migrate, "atomic_write_private", _write_wrong_bytes)

    with pytest.raises(HermesCryptoError, match="post-write verification failed"):
        migrate.disable(force=True)

    assert detect.is_encrypted(env.read_bytes())


def test_status_reports_session_counts():
    _seed_credentials()
    _seed_session("plain1.jsonl")
    encrypted = _seed_session("to-encrypt.jsonl", body='{"line":"will-be-sealed"}\n')

    migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )
    # Sanity: one was sealed at enable time.
    assert detect.is_encrypted(encrypted.read_bytes())

    snapshot = migrate.status()
    sessions = snapshot["sessions"]
    assert sessions["count"] == 2
    assert sessions["encrypted"] >= 1
    assert sessions["plaintext"] + sessions["encrypted"] == sessions["count"]


def test_sweep_sessions_no_op_when_directory_missing():
    _seed_credentials()
    migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )
    # sessions dir was never created — sweep returns empty result, no error.
    result = migrate.sweep_sessions()
    assert result["swept"] == []
    assert result["skipped_locked"] == []
    assert result["errors"] == []


def test_sweep_encrypts_orphan_plaintext_with_stale_lockfile():
    _seed_credentials()
    migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )

    # Plant a plaintext session with an orphaned lockfile (no live writer
    # holds it — the OS would have released it on the prior process's death).
    path = _seed_session("crashed.jsonl", body='{"line":"oops"}\n')
    lock_path = path.with_name(path.name + ".lock")
    from hermes_crypto.session_writer import _open_lockfile

    fd = _open_lockfile(lock_path)
    import os as _os

    _os.close(fd)  # simulates a dead writer — lockfile present, lock free
    assert lock_path.is_file()

    result = migrate.sweep_sessions()
    assert any(label.endswith("crashed.jsonl") for label in result["swept"])
    assert detect.is_encrypted(path.read_bytes())
    assert not lock_path.exists()


def test_sweep_skips_sessions_with_live_writer():
    _seed_credentials()
    migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )

    from hermes_crypto.session_writer import build_session_writer

    path = get_hermes_home() / "sessions" / "live.jsonl"
    writer = build_session_writer(path)
    try:
        writer.write("live line\n")
        writer.flush()

        result = migrate.sweep_sessions()
        assert any(label.endswith("live.jsonl") for label in result["skipped_locked"])
        # Live file is untouched.
        assert not detect.is_encrypted(path.read_bytes())
    finally:
        writer.close()


def test_sweep_is_idempotent_on_already_encrypted_files():
    _seed_credentials()
    path = _seed_session("sealed.jsonl", body='{"line":"sealed-content"}\n')
    migrate.enable(
        "passphrase",
        passphrase="pw",
        encrypt_logs=True,
        argon2_params=FAST_ARGON2,
        force=True,
    )
    assert detect.is_encrypted(path.read_bytes())
    sealed_bytes = path.read_bytes()

    # Second sweep finds nothing to do.
    result = migrate.sweep_sessions()
    assert result["swept"] == []
    assert path.read_bytes() == sealed_bytes


def test_session_targets_returns_empty_when_no_sessions_dir():
    assert migrate.session_targets() == []


def test_full_rekey_rotates_dek_and_reencrypts_credentials():
    env, auth = _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None
    auth_cipher_before = auth.read_bytes()

    result = migrate.full_rekey(passphrase="pw", force=True)

    new_dek = keystore.get_cached_dek()
    assert new_dek is not None
    assert new_dek != old_dek
    assert ".env" in result.rekeyed_files or any(
        label.endswith(".env") for label in result.rekeyed_files
    )
    assert any(label.endswith("auth.json") for label in result.rekeyed_files)
    assert detect.is_encrypted(auth.read_bytes())
    assert auth.read_bytes() != auth_cipher_before
    # Old DEK must no longer decrypt the re-sealed file.
    with pytest.raises(Exception):
        envelope.decrypt(auth.read_bytes(), old_dek)
    assert envelope.decrypt(auth.read_bytes(), new_dek).startswith(b"{")


def test_full_rekey_drops_recovery_slots():
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    code = keystore.add_recovery_slot()
    assert keystore.has_recovery_slot()

    result = migrate.full_rekey(passphrase="pw", force=True)
    assert result.recovery_slots_dropped == 1
    assert not keystore.has_recovery_slot()

    keystore.lock()
    with pytest.raises(Exception):
        keystore.unlock(recovery_code=code)
    assert keystore.unlock(passphrase="pw") is not None


def test_full_rekey_rollback_on_verify_failure(monkeypatch):
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    home = get_hermes_home()
    auth = home / "auth.json"
    original_cipher = auth.read_bytes()

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    real_encrypt = envelope.encrypt
    calls = [0]

    def fail_second_encrypt(data, dek):
        calls[0] += 1
        if calls[0] == 2:
            return b"HRMSENC" + b"\x00" * 8  # invalid blob that will fail verify
        return real_encrypt(data, dek)

    monkeypatch.setattr(migrate.envelope, "encrypt", fail_second_encrypt)

    with pytest.raises(HermesCryptoError):
        migrate.full_rekey(passphrase="pw", force=True)

    assert auth.read_bytes() == original_cipher
    # atomic restore must not leave a half-written .tmp.* artifact.
    assert not list(auth.parent.glob(f"{auth.name}.tmp.*")), (
        "atomic_write_private temp file leaked during rollback"
    )
    assert keystore.get_cached_dek() is not None
    events = audit.read_recent(5)
    assert any(
        e.get("activity") == audit.DATA_KEY_REKEY_FAILED
        and e.get("outcome") == audit.FAILURE
        and e.get("detail", {}).get("reason") == "DecryptionError"
        and e.get("detail", {}).get("restored_count") == 2
        for e in events
    )


def test_full_rekey_reencrypts_sessions_and_logs():
    home = get_hermes_home()
    _seed_credentials()
    sessions = home / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    session_path = sessions / "closed.jsonl"
    session_path.write_bytes(b'{"line":"secret"}\n')

    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "agent.log.1"
    log_body = b"2026-01-01 INFO agent: hello\n"

    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        encrypt_logs=True,
        force=True,
    )
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None
    assert detect.is_encrypted(session_path.read_bytes())
    # Log segments are encrypted at rotation time, not during enable — seal one manually.
    from hermes_crypto.fileio import atomic_write_private

    atomic_write_private(log_path, envelope.encrypt(log_body, old_dek))
    assert detect.is_encrypted(log_path.read_bytes())

    result = migrate.full_rekey(passphrase="pw", force=True)
    new_dek = keystore.get_cached_dek()
    assert new_dek is not None

    assert any(label.endswith("closed.jsonl") for label in result.rekeyed_sessions)
    assert len(result.rekeyed_logs) >= 1
    assert envelope.decrypt(session_path.read_bytes(), new_dek) == b'{"line":"secret"}\n'
    assert envelope.decrypt(log_path.read_bytes(), new_dek) == log_body
    with pytest.raises(Exception):
        envelope.decrypt(session_path.read_bytes(), old_dek)


def test_full_rekey_removes_run_backups_on_success():
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    result = migrate.full_rekey(passphrase="pw", force=True)

    assert result.backups_removed >= 2
    events = audit.read_recent(10)
    assert any(e.get("activity") == audit.BACKUPS_REMOVED_POST_REKEY for e in events)
    # No backup may still hold ciphertext sealed with the retired DEK.
    for bak in migrate._iter_backup_files():
        raw = bak.read_bytes()
        if detect.is_encrypted(raw):
            with pytest.raises(Exception):
                envelope.decrypt(raw, old_dek)


def test_full_rekey_keep_backups_retains_old_dek_ciphertext():
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    result = migrate.full_rekey(passphrase="pw", force=True, keep_backups=True)

    assert result.backups_removed == 0
    events = audit.read_recent(10)
    assert not any(e.get("activity") == audit.BACKUPS_REMOVED_POST_REKEY for e in events)

    retained_old_dek = False
    for bak in migrate._iter_backup_files():
        raw = bak.read_bytes()
        if not detect.is_encrypted(raw):
            continue
        try:
            envelope.decrypt(raw, old_dek)
            retained_old_dek = True
            break
        except Exception:
            continue
    assert retained_old_dek, "expected at least one old-DEK ciphertext backup"


def test_full_rekey_reencrypts_databases():
    pytest.importorskip("sqlcipher3")
    import hermes_state
    from hermes_crypto import dbcrypt

    home = get_hermes_home()
    _seed_credentials()

    # Seed a plaintext state.db. hermes_state rebinds SQLCipher at import
    # time so reading the encrypted DB from this process would not exercise
    # SQLCipher — connecting via dbcrypt.connect_encrypted below uses
    # sqlcipher3 directly, so no subprocess is needed here.
    state_path = home / "state.db"
    db = hermes_state.SessionDB(state_path)
    db.create_session("s1", "cli", model="m")
    db.close()
    assert dbcrypt.is_plaintext_sqlite(state_path)

    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        encrypt_databases=True,
        force=True,
    )
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None
    assert dbcrypt.is_not_plaintext_sqlite(state_path)
    dbcrypt.connect_encrypted(state_path, old_dek).close()

    result = migrate.full_rekey(passphrase="pw", force=True)
    new_dek = keystore.get_cached_dek()
    assert new_dek is not None and new_dek != old_dek

    assert "state.db" in result.rekeyed_databases
    dbcrypt.connect_encrypted(state_path, new_dek).close()
    with pytest.raises(Exception):
        dbcrypt.connect_encrypted(state_path, old_dek).close()

    # any DB backup that survives a successful rekey must be the
    # pre-encrypt plaintext snapshot from enable() — never ciphertext
    # still sealed with the retired DEK (that would mean
    # _remove_rekey_run_backups() failed to delete it).
    backup_dir = keystore.backup_dir()
    for bak in backup_dir.glob("state.db.*.bak"):
        assert dbcrypt.is_plaintext_sqlite(bak), (
            f"{bak.name} still holds ciphertext — rekey-run backup not cleaned"
        )

    # belt-and-suspenders: even if same-second timestamps collapse the
    # surviving backup count to zero, the rekey-run cleanup MUST count the
    # DB backup. With _seed_credentials() (.env + auth.json) plus state.db,
    # a successful full_rekey should remove at least three backups.
    assert result.backups_removed >= 3, (
        f"expected env + auth + state.db backups removed, "
        f"got {result.backups_removed}"
    )

    # cleanup: sidecar backups must also be removed post-rekey.
    assert not list(backup_dir.glob("state.db.*.bak-wal"))
    assert not list(backup_dir.glob("state.db.*.bak-shm"))


def test_full_rekey_aborts_when_concurrent_hermes_detected(monkeypatch):
    _seed_credentials()
    # force=True on the setup enable so it doesn't itself trip the detector.
    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        force=True,
    )
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None

    monkeypatch.setattr(
        migrate,
        "_detect_concurrent_hermes_instances",
        lambda: [{"pid": 4242, "cmdline": "hermes gateway"}],
    )
    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    with pytest.raises(HermesCryptoError, match="another Hermes process is still running"):
        migrate.full_rekey(passphrase="pw")

    # DEK unchanged — guard runs before any work, so no rotation happened.
    assert keystore.get_cached_dek() == old_dek
    events = audit.read_recent(5)
    assert any(
        e.get("activity") == audit.MIGRATION_BLOCKED and e.get("outcome") == audit.FAILURE
        for e in events
    )


def test_full_rekey_force_bypasses_concurrent_hermes_check(monkeypatch):
    _seed_credentials()
    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        force=True,
    )
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None

    monkeypatch.setattr(
        migrate,
        "_detect_concurrent_hermes_instances",
        lambda: [{"pid": 4242, "cmdline": "hermes gateway"}],
    )

    migrate.full_rekey(passphrase="pw", force=True)
    new_dek = keystore.get_cached_dek()
    assert new_dek is not None and new_dek != old_dek


def test_full_rekey_mid_db_rekey_failure_rolls_back(monkeypatch):
    """a failure after state.db is re-keyed on disk must roll back via restored."""
    pytest.importorskip("sqlcipher3")
    import sqlite3

    import hermes_state
    from hermes_crypto import dbcrypt

    home = get_hermes_home()
    _seed_credentials()

    state_path = home / "state.db"
    db = hermes_state.SessionDB(state_path)
    db.create_session("s1", "cli", model="m")
    db.close()

    kanban_path = home / "kanban.db"
    conn = sqlite3.connect(str(kanban_path))
    conn.execute("CREATE TABLE t(x INT)")
    conn.commit()
    conn.close()

    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        encrypt_databases=True,
        force=True,
    )
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    real_rekey = dbcrypt.rekey_database
    calls = [0]

    def fail_after_first_on_disk_rekey(path, old_dek_arg, new_dek):
        calls[0] += 1
        real_rekey(path, old_dek_arg, new_dek)
        if calls[0] == 1:
            raise HermesCryptoError("forced post-rekey failure")

    monkeypatch.setattr(migrate.dbcrypt, "rekey_database", fail_after_first_on_disk_rekey)

    with pytest.raises(HermesCryptoError):
        migrate.full_rekey(passphrase="pw", force=True)

    dbcrypt.connect_encrypted(state_path, old_dek).close()
    dbcrypt.connect_encrypted(kanban_path, old_dek).close()
    assert keystore.get_cached_dek() == old_dek

    events = audit.read_recent(5)
    assert any(
        e.get("activity") == audit.DATA_KEY_REKEY_FAILED
        and e.get("outcome") == audit.FAILURE
        and e.get("detail", {}).get("reason") == "HermesCryptoError"
        and e.get("detail", {}).get("restored_count") >= 3
        for e in events
    )


# ─── SQLCipher sidecar snapshot + rollback ──────────────────────────────


def test_backup_captures_sqlcipher_sidecars():
    """_backup must snapshot -wal / -shm sidecars alongside the main file."""
    home = get_hermes_home()
    state = home / "state.db"
    state.write_bytes(b"SQLite format 3\x00" + b"\x00" * 500)
    wal = home / "state.db-wal"
    shm = home / "state.db-shm"
    wal.write_bytes(b"WAL-PAYLOAD-XYZ")
    shm.write_bytes(b"SHM-PAYLOAD-ABC")

    backup = migrate._backup(state)

    assert backup.is_file()
    assert backup.read_bytes() == state.read_bytes()
    backup_wal = Path(str(backup) + "-wal")
    backup_shm = Path(str(backup) + "-shm")
    assert backup_wal.is_file() and backup_wal.read_bytes() == b"WAL-PAYLOAD-XYZ"
    assert backup_shm.is_file() and backup_shm.read_bytes() == b"SHM-PAYLOAD-ABC"


def test_full_rekey_rollback_restores_db_sidecars(monkeypatch):
    """rollback after a mid-rekey failure must restore the DB sidecars.

    Strategy: let the DB rekey succeed (so the DB gets backed up + sidecars
    snapshotted, then re-keyed), then make the FOLLOWING envelope rekey
    fail. The rollback path then runs against the database entry and must
    put back the snapshotted sidecars byte-for-byte.
    """
    pytest.importorskip("sqlcipher3")
    import hermes_state
    from hermes_crypto import dbcrypt

    home = get_hermes_home()
    _seed_credentials()

    state_path = home / "state.db"
    db = hermes_state.SessionDB(state_path)
    db.create_session("s1", "cli", model="m")
    db.close()

    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        encrypt_databases=True,
        force=True,
    )

    # Stub a "live WAL" alongside the encrypted DB after enable. dbcrypt's
    # _drop_sidecars cleans these after a successful conversion, so we
    # write them ourselves to simulate uncheckpointed WAL state.
    wal_bytes = b"WAL-pre-rekey-payload"
    shm_bytes = b"SHM-pre-rekey-payload"
    (home / "state.db-wal").write_bytes(wal_bytes)
    (home / "state.db-shm").write_bytes(shm_bytes)

    # Patch replace_data_key, not envelope.encrypt on a later target:
    # full_rekey iterates envelopes -> databases -> keystore flip in that
    # order, so the only call after the DB loop is replace_data_key. To
    # exercise rollback over an already-rekeyed DB the failure must land
    # at or after that call. See the ordering note in full_rekey().
    real_replace = keystore.replace_data_key

    def fail_keystore_update(*args, **kwargs):
        raise HermesCryptoError("forced post-rekey failure to trigger rollback")

    monkeypatch.setattr(migrate.keystore, "replace_data_key", fail_keystore_update)

    with pytest.raises(HermesCryptoError):
        migrate.full_rekey(passphrase="pw", force=True)

    # Sidecars must have been restored from the backup snapshot, not the
    # post-rekey state that dbcrypt._drop_sidecars would have removed.
    assert (home / "state.db-wal").read_bytes() == wal_bytes, (
        "WAL sidecar not restored from snapshot on rollback"
    )
    assert (home / "state.db-shm").read_bytes() == shm_bytes, (
        "SHM sidecar not restored from snapshot on rollback"
    )
    # Sanity: keystore was not touched (replace_data_key never succeeded).
    _ = real_replace  # silence unused-locals; keep reference for clarity


# ─── rollback_errors surfaced in DATA_KEY_REKEY_FAILED ─────────────────


def test_full_rekey_audits_rollback_errors(monkeypatch):
    """per-file rollback OSErrors must be counted in the audit event.

    Force a verify failure mid-rekey AND make ``atomic_write_private``
    raise ``OSError`` on the rollback path. The
    ``DATA_KEY_REKEY_FAILED`` event must surface a non-zero
    ``rollback_errors`` count so an operator who ran ``--full --force``
    after suspected DEK compromise can tell that the restore did not
    complete cleanly.
    """
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    # Force the second envelope.encrypt call to emit a malformed blob so
    # verify fails (mirrors test_full_rekey_rollback_on_verify_failure).
    real_encrypt = envelope.encrypt
    encrypt_calls = [0]

    def fail_second_encrypt(data, dek):
        encrypt_calls[0] += 1
        if encrypt_calls[0] == 2:
            return b"HRMSENC" + b"\x00" * 8  # invalid blob — verify will fail
        return real_encrypt(data, dek)

    monkeypatch.setattr(migrate.envelope, "encrypt", fail_second_encrypt)

    # Make the rollback restore path raise ``OSError`` so the rollback
    # counter ticks. The rollback uses ``atomic_copy(backup, path)`` per
    # ``_rekey_file`` still commits new ciphertext via
    # ``atomic_write_private`` (which we leave alone, otherwise the rekey
    # fails for a different reason and the rollback wouldn't even have a
    # populated ``restored`` list to fail over). The flag flips the first
    # time ``_rekey_file`` raises — that's exactly when ``_rollback``
    # starts iterating.
    real_atomic_copy = migrate.atomic_copy
    rollback_active = {"on": False}

    def fail_atomic_copy_on_rollback(src, dest, *args, **kwargs):
        if rollback_active["on"]:
            raise OSError(28, "No space left on device")
        return real_atomic_copy(src, dest, *args, **kwargs)

    monkeypatch.setattr(migrate, "atomic_copy", fail_atomic_copy_on_rollback)

    real_rekey_file = migrate._rekey_file

    def trip_flag_on_rekey_failure(target, raw, old_dek, new_dek):
        try:
            return real_rekey_file(target, raw, old_dek, new_dek)
        except Exception:
            rollback_active["on"] = True
            raise

    monkeypatch.setattr(migrate, "_rekey_file", trip_flag_on_rekey_failure)

    with pytest.raises(HermesCryptoError):
        migrate.full_rekey(passphrase="pw", force=True)

    events = audit.read_recent(10)
    matches = [
        e for e in events
        if e.get("activity") == audit.DATA_KEY_REKEY_FAILED
        and e.get("outcome") == audit.FAILURE
    ]
    assert matches, "no DATA_KEY_REKEY_FAILED event was emitted"
    assert any(
        e.get("detail", {}).get("rollback_errors", 0) >= 1
        for e in matches
    ), (
        "expected rollback_errors >= 1 in DATA_KEY_REKEY_FAILED detail; "
        f"got events: {matches}"
    )

    # Sanity: keystore was not flipped (failure happened well before
    # replace_data_key) and the DEK cache still holds the old DEK.
    assert keystore.get_cached_dek() is not None


# ─── categorisation routes by ``kind``, not by path sniffing ───────────


def test_full_rekey_categorisation_robust_to_logs_in_ancestor(monkeypatch, tmp_path):
    """a HERMES_HOME whose ancestor is literally named ``logs`` must
    not mis-bucket credential files.

    Pre-fix, ``full_rekey`` categorised every envelope target via
    ``"logs" in target.path.parts``. With this fix, the
    :class:`migrate.FileTarget` carries an explicit ``kind`` tag that
    drives the bucket, so even a HERMES_HOME beneath a directory named
    ``logs`` keeps credential files in ``result.rekeyed_files`` rather
    than leaking them into ``result.rekeyed_logs``.
    """
    # Build a HERMES_HOME with a ``logs`` ancestor segment. Pre-fix
    # path-sniffing would categorise every credential under this home as
    # a log archive.
    pathological_home = tmp_path / "logs" / "hermes_home"
    pathological_home.mkdir(parents=True, exist_ok=True)
    assert "logs" in pathological_home.parts, (
        "fixture setup error: expected 'logs' segment in HERMES_HOME path"
    )
    monkeypatch.setenv("HERMES_HOME", str(pathological_home))

    # Seed credentials inside the pathological HERMES_HOME.
    (pathological_home / ".env").write_text(
        "OPENAI_API_KEY=sk-test-secret\n", encoding="utf-8"
    )
    (pathological_home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}),
        encoding="utf-8",
    )

    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    result = migrate.full_rekey(passphrase="pw", force=True)

    # Every credential must land in ``rekeyed_files`` even though every
    # target path contains ``logs`` as a parts segment. Pre-fix this
    # assertion would fail — the entries would all be in
    # ``result.rekeyed_logs``.
    assert any(
        label.endswith(".env") for label in result.rekeyed_files
    ), (
        f"expected .env in rekeyed_files; "
        f"files={result.rekeyed_files} logs={result.rekeyed_logs}"
    )
    assert any(
        label.endswith("auth.json") for label in result.rekeyed_files
    ), (
        f"expected auth.json in rekeyed_files; "
        f"files={result.rekeyed_files} logs={result.rekeyed_logs}"
    )
    # No credential may leak into rekeyed_logs.
    for label in result.rekeyed_logs:
        assert not label.endswith(".env")
        assert not label.endswith("auth.json")


# ── config-write atomicity + audit-on-partial-failure ──
#
# These regressions pin two things:
#   1. ``enable`` and ``disable`` emit an ``ENCRYPTION_ENABLED`` /
#      ``ENCRYPTION_DISABLED`` event with ``outcome=failure`` and a
#      ``config_written`` / ``config_not_written`` breakdown when a
#      ``_set_config`` call raises mid-sequence. The original audit gap
#      meant operators could lose the audit trail entirely when a YAML
#      write hit ENOSPC or similar.
#   2. ``disable`` flips ``enabled=False`` FIRST so a downstream
#      ``encrypt_logs`` write failure leaves the runtime in a safe
#      "off" state (``logs_encryption_active()`` short-circuits on
#      ``enabled=False``). Reverse of ``enable``'s flipped-last pattern.


def test_enable_partial_config_write_fires_failure_audit(monkeypatch):
    """a mid-sequence ``_set_config`` failure must emit
    ENCRYPTION_ENABLED + outcome=failure with the partial-write breakdown,
    then re-raise so the operator sees the error.
    """
    _seed_credentials()
    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    real_set_config = migrate._set_config
    call_count = [0]

    def fail_on_third_call(dotted_key, value):
        call_count[0] += 1
        if call_count[0] == 3:
            raise OSError(28, "No space left on device")
        return real_set_config(dotted_key, value)

    monkeypatch.setattr(migrate, "_set_config", fail_on_third_call)

    with pytest.raises(OSError, match="No space left"):
        migrate.enable(
            "passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True
        )

    events = audit.read_recent(20)
    failure_events = [
        e
        for e in events
        if e.get("activity") == audit.ENCRYPTION_ENABLED
        and e.get("outcome") == audit.FAILURE
    ]
    assert len(failure_events) == 1, (
        f"expected exactly one ENCRYPTION_ENABLED+failure event, "
        f"got {[e.get('activity') + '+' + e.get('outcome', '') for e in events]}"
    )
    detail = failure_events[0]["detail"]
    assert detail["config_written"] == [
        "security.encryption.key_source",
        "security.encryption.encrypt_credentials",
    ], detail
    assert detail["config_not_written"] == [
        "security.encryption.encrypt_databases",
        "security.encryption.encrypt_logs",
        "security.encryption.enabled",
    ], detail
    assert detail["key_source"] == "passphrase"
    # No SUCCESS audit fired — the happy-path event is only emitted after the
    # full sequence completes.
    assert not any(
        e.get("activity") == audit.ENCRYPTION_ENABLED
        and e.get("outcome") == audit.SUCCESS
        for e in events
    )


def test_disable_partial_config_write_flips_enabled_first(monkeypatch):
    """when ``_set_config`` fails on disable's 2nd write, the
    ``enabled=False`` flip must have landed first so the runtime is in a
    safe "off" state, and the failure audit must record the gap.
    """
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    real_set_config = migrate._set_config
    call_count = [0]

    def fail_on_second_call(dotted_key, value):
        call_count[0] += 1
        if call_count[0] == 2:
            raise OSError(28, "No space left on device")
        return real_set_config(dotted_key, value)

    monkeypatch.setattr(migrate, "_set_config", fail_on_second_call)

    with pytest.raises(OSError, match="No space left"):
        migrate.disable(force=True)

    events = audit.read_recent(20)
    failure_events = [
        e
        for e in events
        if e.get("activity") == audit.ENCRYPTION_DISABLED
        and e.get("outcome") == audit.FAILURE
    ]
    assert len(failure_events) == 1, (
        f"expected ENCRYPTION_DISABLED+failure, got "
        f"{[e.get('activity') + '+' + e.get('outcome', '') for e in events]}"
    )
    detail = failure_events[0]["detail"]
    # The CRITICAL ordering invariant: enabled=False landed first.
    assert detail["config_written"] == [
        "security.encryption.enabled",
    ], detail
    assert detail["config_not_written"] == [
        "security.encryption.encrypt_logs",
    ], detail
    # Runtime is now in a safe state — is_encryption_enabled() should reflect
    # the enabled=False write even though encrypt_logs is stale.
    assert not is_encryption_enabled()
    # No SUCCESS audit fired.
    assert not any(
        e.get("activity") == audit.ENCRYPTION_DISABLED
        and e.get("outcome") == audit.SUCCESS
        for e in events
    )


def test_disable_happy_path_emits_success_audit_after_both_writes():
    """regression — the happy path still emits exactly one
    ENCRYPTION_DISABLED + outcome=success event after both `_set_config`
    calls land. No spurious dual emission, no failure event.
    """
    _seed_credentials()
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    migrate.disable(force=True)

    events = audit.read_recent(20)
    disable_events = [
        e for e in events if e.get("activity") == audit.ENCRYPTION_DISABLED
    ]
    assert len(disable_events) == 1, (
        f"expected exactly one ENCRYPTION_DISABLED event, "
        f"got {[e.get('outcome') for e in disable_events]}"
    )
    assert disable_events[0]["outcome"] == audit.SUCCESS
