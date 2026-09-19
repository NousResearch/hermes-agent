from __future__ import annotations

from pathlib import Path

import pytest

from plugins.agentops.control.events import EventSpool, EventValidationError
from plugins.agentops.control.store import open_store


def test_synthetic_secret_never_enters_spool_or_database(make_event, write_config):
    from plugins.agentops.control.config import load_agentops_config

    secret = "sk-test-canary-secret"
    config = load_agentops_config(write_config())
    spool = EventSpool(config.spool_dir)
    store = open_store(config)

    with pytest.raises(EventValidationError):
        event = make_event(payload={"cookie": secret})
        spool.write(event)

    assert secret.encode() not in config.sqlite_path.read_bytes()
    assert not spool.pending_paths()


def test_phase_one_source_contains_no_target_execution_primitives():
    root = Path(__file__).resolve().parents[4] / "plugins" / "agentops"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
    )

    assert "subprocess" not in source
    assert "os.system" not in source
    assert "launchctl" not in source
    assert "shell=True" not in source
