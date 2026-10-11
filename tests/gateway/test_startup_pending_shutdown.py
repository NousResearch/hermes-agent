"""Shutdown retains deferred startup input under each owning profile."""

import json
import weakref

import pytest

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session_identity import RoutingIdentity
from gateway.shutdown_pending import flush_runner_pending
from hermes_constants import get_hermes_home
from tests.gateway.test_active_session_text_merge import _make_event, _make_initialized_adapter


@pytest.mark.parametrize("write_failure", [False, True])
def test_shutdown_preserves_startup_fifo_and_retries_in_the_owning_home(tmp_path, monkeypatch, write_failure):
    launch = tmp_path / "launch"
    launch.mkdir()
    homes = {profile: tmp_path / profile for profile in ("a", "b")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        adapter = _make_initialized_adapter()
        runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
        runner.adapters[adapter.platform] = adapter
        adapter.gateway_runner = runner
        older = _make_event("older-tail")
        older.source.profile = "a"
        setattr(older.source, "_identity", RoutingIdentity(
            "default", "a", launch, homes["a"], transport=weakref.ref(adapter),
        ))
        key = runner._session_key_for_source(older.source)
        tail = runner._session_state(key).conversation.queued_events
        tail.append(older)
        runner._startup_restore_in_progress = True
        events = []
        for index, profile in enumerate(("a", "b", "a")):
            event = _make_event(f"deferred-{index}")
            event.source.profile = profile
            setattr(event.source, "_identity", RoutingIdentity(
                "default", profile, launch, homes[profile], transport=weakref.ref(adapter),
            ))
            assert runner._defer_for_startup_restore(event)
            events.append(event)
        if write_failure:
            from gateway.shutdown_flush import _write_payload

            def write(directory, payload):
                if get_hermes_home() == homes["a"]:
                    raise OSError("controlled full disk")
                return _write_payload(directory, payload)

            monkeypatch.setattr("gateway.shutdown_flush._write_payload", write)

        flush_runner_pending(runner)

        def stored():
            return {
                profile: [record["event"]["text"]
                          for path in (home / "pending_messages").glob("*.json")
                          for record in json.loads(path.read_text())["events"]]
                for profile, home in homes.items()
            }

        assert (stored(), tail, runner._startup_restore_queue, get_hermes_home(),
                list((launch / "pending_messages").glob("*.json"))) == (
            {"a": [] if write_failure else ["older-tail", "deferred-0", "deferred-2"], "b": ["deferred-1"]},
            [older] if write_failure else [], [events[0], events[2]] if write_failure else [], launch, [],
        )
        if write_failure:
            monkeypatch.setattr("gateway.shutdown_flush._write_payload", _write_payload)
            flush_runner_pending(runner)
            assert (stored(), tail, runner._startup_restore_queue) == (
                {"a": ["older-tail", "deferred-0", "deferred-2"], "b": ["deferred-1"]}, [], [],
            )
    finally:
        set_multiplex_active(active)
