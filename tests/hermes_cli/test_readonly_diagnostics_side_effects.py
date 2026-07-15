from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_readonly_argv_detection():
    from hermes_constants import is_readonly_diagnostic_argv

    assert is_readonly_diagnostic_argv(["hermes", "doctor"])
    assert is_readonly_diagnostic_argv(["hermes", "security", "audit"])
    assert is_readonly_diagnostic_argv(["hermes", "prompt-size", "--json"])
    assert is_readonly_diagnostic_argv(["hermes", "status"])
    assert is_readonly_diagnostic_argv(["hermes", "gateway", "status"])
    assert is_readonly_diagnostic_argv(["hermes", "model", "--preflight"])
    assert is_readonly_diagnostic_argv(
        ["hermes", "--profile", "main-gateway-v2", "model", "--preflight", "--json"]
    )
    assert is_readonly_diagnostic_argv(
        [
            "hermes", "--profile", "main-gateway-v2", "model",
            "--provider", "openai-codex", "--model", "gpt-next",
            "--confirm-profile", "main-gateway-v2", "--json",
        ]
    )
    assert not is_readonly_diagnostic_argv(
        [
            "hermes", "--profile", "main-gateway-v2", "model",
            "--provider", "openai-codex", "--model", "gpt-next",
            "--confirm-profile", "main-gateway-v2", "--apply-transaction",
        ]
    )
    assert is_readonly_diagnostic_argv(
        ["hermes", "--profile", "main-gateway-v2", "logs", "gateway", "--triage-current-start", "--json"]
    )
    assert not is_readonly_diagnostic_argv(["hermes", "doctor", "--fix"])
    assert not is_readonly_diagnostic_argv(["hermes", "doctor", "--ack", "ADV-1"])


def _tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | None]]:
    snapshot: dict[str, tuple[str, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            snapshot[relative] = ("dir", None)
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


def test_readonly_cli_diagnostics_leave_isolated_filesystem_unchanged(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    sandbox = tmp_path / "sandbox"
    process_home = sandbox / "home"
    hermes_home = sandbox / "hermes"
    xdg_cache = sandbox / "xdg-cache"
    xdg_state = sandbox / "xdg-state"
    for directory in (process_home, hermes_home, xdg_cache, xdg_state):
        directory.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: openai/gpt-4o-mini\n  provider: openrouter\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(process_home),
            "HERMES_HOME": str(hermes_home),
            "XDG_CACHE_HOME": str(xdg_cache),
            "XDG_STATE_HOME": str(xdg_state),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env.pop("HERMES_READONLY_DIAGNOSTIC", None)
    before = _tree_snapshot(sandbox)

    entrypoint = "from hermes_cli.main import main; raise SystemExit(main() or 0)"
    for args in (("prompt-size", "--json"), ("gateway", "status")):
        result = subprocess.run(
            [sys.executable, "-c", entrypoint, *args],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        if args[0] == "prompt-size":
            json_start = result.stdout.find("{")
            assert json_start >= 0, result.stdout
            payload = json.loads(result.stdout[json_start:])
            assert payload["system_prompt"]["bytes"] > 0
        else:
            assert "Gateway" in result.stdout
        assert _tree_snapshot(sandbox) == before
        assert not (hermes_home / "state.db-wal").exists()
        assert not (hermes_home / "state.db-shm").exists()
        assert not (hermes_home / "state.db-journal").exists()


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

    import agent.model_metadata as model_metadata
    import agent.models_dev as models_dev
    import agent.prompt_builder as prompt_builder
    import hermes_cli.model_catalog as model_catalog
    import hermes_cli.models as models
    import hermes_cli.security_advisories as advisories
    import tools.skill_usage as skill_usage

    model_metadata._save_model_metadata_disk_cache({"example/model": {"context_length": 1}})
    models_dev._save_disk_cache({"example": {"models": {}}})
    prompt_builder._write_skills_snapshot(tmp_path / "skills", {}, [], {})
    skill_usage.bump_use("readonly-diagnostic-test")
    model_catalog._write_disk_cache({"models": ["x"]})
    models._write_nous_recommended_disk("https://example.invalid", {"models": ["x"]})
    models._save_provider_models_cache({"provider": {"models": ["x"]}})
    advisories._write_banner_cache({"ADV-1": 1.0})

    assert not (tmp_path / ".skills_prompt_snapshot.json").exists()
    assert not (tmp_path / "skills/.usage.json").exists()
    assert not (tmp_path / "skills/.usage.json.lock").exists()
    assert not (tmp_path / "cache/openrouter_model_metadata.json").exists()
    assert not (tmp_path / "models_dev_cache.json").exists()
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
