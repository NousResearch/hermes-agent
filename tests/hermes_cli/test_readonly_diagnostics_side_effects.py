from __future__ import annotations

import importlib
import os
import sqlite3
import sys
from pathlib import Path


def test_readonly_argv_detection():
    from hermes_constants import is_readonly_diagnostic_argv

    assert is_readonly_diagnostic_argv(["hermes", "doctor"])
    assert is_readonly_diagnostic_argv(["hermes", "security", "audit"])
    assert is_readonly_diagnostic_argv(["hermes", "model", "--preflight"])
    assert is_readonly_diagnostic_argv(
        ["hermes", "--profile", "main-gateway-v2", "model", "--preflight", "--json"]
    )
    assert is_readonly_diagnostic_argv(
        ["hermes", "--profile", "main-gateway-v2", "logs", "gateway", "--triage-current-start", "--json"]
    )
    assert not is_readonly_diagnostic_argv(["hermes", "doctor", "--fix"])
    assert not is_readonly_diagnostic_argv(["hermes", "doctor", "--ack", "ADV-1"])


def test_readonly_diagnostic_skips_cli_log_file_creation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_READONLY_DIAGNOSTIC", "1")
    import hermes_logging

    log_dir = hermes_logging.setup_logging(hermes_home=tmp_path, mode="cli", force=True)
    assert log_dir == tmp_path / "logs"
    assert not log_dir.exists()


def test_readonly_diagnostic_does_not_backup_corrupt_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_READONLY_DIAGNOSTIC", "1")
    from hermes_cli.config import _backup_corrupt_config

    config = tmp_path / "config.yaml"
    config.write_text("model: [broken\n", encoding="utf-8")
    assert _backup_corrupt_config(config) is None
    assert list(tmp_path.glob("config.yaml.corrupt.*.bak")) == []


def test_readonly_db_probe_creates_no_sqlite_sidecars(tmp_path):
    from hermes_state import _db_opens_cleanly

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    assert _db_opens_cleanly(db, write_probe=False) is None
    assert not (tmp_path / "state.db-wal").exists()
    assert not (tmp_path / "state.db-shm").exists()
    assert not (tmp_path / "state.db-journal").exists()


def test_readonly_cache_writers_are_noops(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_READONLY_DIAGNOSTIC", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.model_catalog as model_catalog
    import hermes_cli.models as models
    import hermes_cli.security_advisories as advisories

    model_catalog._write_disk_cache({"models": ["x"]})
    models._write_nous_recommended_disk("https://example.invalid", {"models": ["x"]})
    models._save_provider_models_cache({"provider": {"models": ["x"]}})
    advisories._write_banner_cache({"ADV-1": 1.0})

    assert not (tmp_path / "cache/model_catalog.json").exists()
    assert not (tmp_path / "cache/nous_recommended_cache.json").exists()
    assert not (tmp_path / "provider_models_cache.json").exists()
    assert not (tmp_path / "cache/advisory_banner_seen").exists()
    assert not (tmp_path / "cache").exists()


def test_cli_sets_readonly_flag_before_startup_config(monkeypatch):
    monkeypatch.delenv("HERMES_READONLY_DIAGNOSTIC", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "security", "audit"])
    sys.modules.pop("cli", None)
    try:
        module = importlib.import_module("cli")
        assert module.os.environ.get("HERMES_READONLY_DIAGNOSTIC") == "1"
    finally:
        os.environ.pop("HERMES_READONLY_DIAGNOSTIC", None)
