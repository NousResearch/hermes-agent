"""Native Relay ATOF capture shared by adapter and real-backend tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.fixture()
def relay_capture(tmp_path, monkeypatch):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")

    from agent import relay_runtime

    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    config = tmp_path / "plugins.toml"
    config.write_text(
        f"""version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = {json.dumps(str(atof_dir))}
filename = "events.jsonl"
mode = "overwrite"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(config))
    monkeypatch.setattr(
        "hermes_cli.config_effective.load_user_config_effective", lambda *_a, **_k: {}
    )
    relay_runtime._reset_for_tests()
    leases, turns = [], []
    coordinator = relay_runtime.SESSION_COORDINATOR

    def acquire(session_id="session-1", *, parent_session_id=""):
        lease = coordinator.acquire_conversation(
            profile_key=relay_runtime.current_profile_key(),
            session_id=session_id,
            platform="cli",
            parent_session_id=parent_session_id,
        )
        leases.append(lease)
        return lease

    def begin(lease, turn_id="turn-1"):
        turn = coordinator.begin_turn(lease, turn_id=turn_id, task_id=turn_id)
        turns.append(turn)
        return turn

    def events():
        relay.subscribers.flush()
        return [
            json.loads(line)
            for line in (atof_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    try:
        yield SimpleNamespace(
            relay=relay,
            coordinator=coordinator,
            acquire=acquire,
            begin=begin,
            events=events,
        )
    finally:
        for turn in reversed(turns):
            coordinator.end_turn(turn, outcome="success")
        for lease in reversed(leases):
            coordinator.release_conversation(lease)
        relay_runtime._reset_for_tests()
