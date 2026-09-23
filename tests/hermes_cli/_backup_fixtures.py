"""Shared isolated backup fixtures and fake Hermes home builders."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    """run_import() auto-installs the gateway service post-restore; tests must
    never touch the host's systemd/launchd. Individual tests re-patch these to
    assert the wiring."""
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def _advance_backup_clock(seconds: float = 1.1) -> None:
    """Skew hermes_cli.backup's datetime forward instead of sleeping.

    Snapshot ids have 1-second resolution; tests that need two distinct
    timestamps previously slept >1s. This installs (once) a datetime shim in
    the backup module whose now() adds a cumulative offset, then bumps it.
    """
    import datetime as _dt

    import hermes_cli.backup as _backup

    shim = getattr(_backup.datetime, "_hermes_test_shim", None)
    if shim is None:
        class _ShimDatetime(_dt.datetime):
            _hermes_test_shim = True
            _offset = _dt.timedelta(0)

            @classmethod
            def now(cls, tz=None):  # noqa: D102
                return _dt.datetime.now(tz) + cls._offset

        _backup.datetime = _ShimDatetime
        shim = _ShimDatetime
    else:
        shim = _backup.datetime
    shim._offset += _dt.timedelta(seconds=seconds)


def _make_hermes_tree(root: Path) -> None:
    """Create a realistic ~/.hermes directory structure for testing."""
    (root / "config.yaml").write_text("model:\n  provider: openrouter\n")
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-test-123\n")
    for db_name in ("memory_store.db", "hermes_state.db"):
        with sqlite3.connect(root / db_name) as conn:
            conn.execute("CREATE TABLE sample (value TEXT)")
            conn.execute("INSERT INTO sample VALUES ('test')")

    # Sessions
    (root / "sessions").mkdir(exist_ok=True)
    (root / "sessions" / "abc123.json").write_text("{}")

    # Skills
    (root / "skills").mkdir(exist_ok=True)
    (root / "skills" / "my-skill").mkdir()
    (root / "skills" / "my-skill" / "SKILL.md").write_text("# My Skill\n")

    # Skins
    (root / "skins").mkdir(exist_ok=True)
    (root / "skins" / "cyber.yaml").write_text("name: cyber\n")

    # Cron
    (root / "cron").mkdir(exist_ok=True)
    (root / "cron" / "jobs.json").write_text("[]")

    # Memories
    (root / "memories").mkdir(exist_ok=True)
    (root / "memories" / "notes.json").write_text("{}")

    # Profiles
    (root / "profiles").mkdir(exist_ok=True)
    (root / "profiles" / "coder").mkdir()
    (root / "profiles" / "coder" / "config.yaml").write_text("model:\n  provider: anthropic\n")
    (root / "profiles" / "coder" / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-123\n")

    # hermes-agent repo (should be EXCLUDED)
    (root / "hermes-agent").mkdir(exist_ok=True)
    (root / "hermes-agent" / "run_agent.py").write_text("# big file\n")
    (root / "hermes-agent" / ".git").mkdir()
    (root / "hermes-agent" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    # __pycache__ (should be EXCLUDED)
    (root / "plugins").mkdir(exist_ok=True)
    (root / "plugins" / "__pycache__").mkdir()
    (root / "plugins" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")

    # PID files (should be EXCLUDED)
    (root / "gateway.pid").write_text("12345")

    # Logs (should be included)
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "agent.log").write_text("log line\n")


def _symlink_file_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")
