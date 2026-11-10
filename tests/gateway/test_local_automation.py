"""Local automation belongs to the owner, never an attached client."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_local_automation_accepts_a_restored_desktop_source(monkeypatch):
    """A process watcher rebuilds its source from the durable route, not the live object."""
    from types import SimpleNamespace

    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_automation import snapshot_local_automation

    live_source = SessionSource(platform=Platform.LOCAL, chat_id="desktop-session", user_id="nolan", chat_type="dm")
    restored_source = SessionSource.from_dict(live_source.to_dict())
    route = "agent:local:desktop-session"
    entry = SimpleNamespace(session_key=route, session_id="desktop-session")
    adapter = SimpleNamespace(authorize_source=lambda source: source is live_source)
    runner = SimpleNamespace(_adapter_for_source=lambda source: adapter if source is live_source else None)
    authority = SimpleNamespace(runner=runner, sessions={"desktop-session": SimpleNamespace(
        source=live_source, route=route,
    )})
    monkeypatch.setattr("gateway.session_local_recovery.restore_local_session",
                        lambda _authority, _chat_id: SimpleNamespace(session_id="desktop-session"))
    event = MessageEvent(text="[background notification]", source=restored_source, internal=True,
                         metadata={"gateway_session_key": route, "gateway_session_id": "desktop-session"})

    payload, returned_entry = snapshot_local_automation(authority, adapter, event, "watch:desktop", entry)

    assert returned_entry is entry
    assert payload["local_automation_v1"]["owner"] == "desktop-session"


def _probe(tmp_path, source):
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env['PYTHONPATH'] = str(root)
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/local_automation_peer.py'),
        str(tmp_path), source], cwd=root, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


@pytest.mark.parametrize('source', ['cli', 'tui', 'gui'])
def test_local_completion_without_an_observer_is_a_note_the_next_turn_sees(tmp_path, source):
    _probe(tmp_path, source)  # #134889 LC-7: no unattended turn


def test_local_watch_match_without_an_observer_is_a_note(tmp_path):
    _probe(tmp_path, 'watch')


def test_local_completion_survives_owner_kill_without_duplicate(tmp_path):
    _probe(tmp_path, 'restart')
