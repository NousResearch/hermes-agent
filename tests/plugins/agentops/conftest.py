"""Shared factories for AgentOps Phase 1 tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

import pytest


@pytest.fixture
def make_event():
    def _make_event(event_id: str = "evt-0001", *, payload: dict | None = None, schema_version: int = 1):
        from plugins.agentops.control.models import EventEnvelope

        return EventEnvelope.create(
            schema_version=schema_version,
            event_id=event_id,
            event_type="signal.observed",
            occurred_at=datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc),
            producer="test.collector.v1",
            target_id="hermes:profile:default:gateway",
            correlation_id="corr-0001",
            payload=payload or {"status": "ok"},
            redaction_version=1,
        )

    return _make_event


@pytest.fixture
def write_config(tmp_path):
    created_state_dirs: list[Path] = []

    def _write_config(*, state_dir=None, extra: str = ""):
        if state_dir is None:
            parent = Path(tempfile.mkdtemp(prefix="agentops-parent-", dir="/private/tmp"))
            state_dir = parent / "state"
            created_state_dirs.append(parent)
        path = tmp_path / "agentops.yaml"
        path.write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    "control_plane:",
                    "  socket_path: agentops.sock",
                    "  event_spool_max_mb: 1",
                    "storage:",
                    f"  state_dir: {state_dir}",
                    "  sqlite_path: state.db",
                    "  spool_dir: event-spool",
                    "safety:",
                    "  default_authority: observe_only",
                    "  global_write_enabled: false",
                    extra.strip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        from plugins.agentops.control.config import initialize_state_dir, load_agentops_config

        initialize_state_dir(load_agentops_config(path))
        return path

    yield _write_config
    for state_dir in created_state_dirs:
        shutil.rmtree(state_dir, ignore_errors=True)
