"""Regression coverage for #104671: a backlog of background-process completions
must not become one full agent turn per event in the interactive CLI.

``cli._drain_process_notifications`` used to ``_pending_input.put`` every
drained completion independently, so N stale/dead completions produced N
model calls. Completions drained together are now coalesced into a single
synthetic turn; non-completion events (async delegation results) keep their
exact one-by-one delivery.
"""

import queue

from cli import HermesCLI


def _completion_event(sid):
    return {
        "type": "completion",
        "session_id": sid,
        "session_key": "cli-session",
        "command": f"sleep {sid}",
        "exit_code": 0,
    }


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-session"
    cli._session_db = None
    cli._pending_input = queue.Queue()
    return cli


def _patch(monkeypatch, registry):
    claimed = []
    completed = []

    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        registry,
    )
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: claimed.append((evt, consumer)) or "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )
    return claimed, completed


class FakeRegistry:
    def __init__(self, pairs):
        self._pairs = list(pairs)

    def drain_notifications(self, *, session_key="", owns_event=None):
        return list(self._pairs)


def test_backlog_coalesces_into_single_pending_input(monkeypatch):
    """10 ready completions -> exactly ONE queued turn containing every payload."""
    cli = _make_cli()
    events = [_completion_event(f"proc_{i}") for i in range(10)]
    pairs = [(evt, f"process {evt['session_id']} done") for evt in events]
    _, completed = _patch(monkeypatch, FakeRegistry(pairs))

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.qsize() == 1
    batched = cli._pending_input.get_nowait()
    for evt in events:
        assert evt["session_id"] in batched
    # Every claimed completion is still acknowledged exactly once.
    assert sorted(evt["session_id"] for evt, _token in completed) == sorted(
        evt["session_id"] for evt in events
    )


def test_single_completion_delivery_unchanged(monkeypatch):
    """One genuine completion still reaches the model as its raw message."""
    cli = _make_cli()
    event = _completion_event("proc_only")
    claimed, completed = _patch(monkeypatch, FakeRegistry([(event, "solo payload")]))

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.qsize() == 1
    assert cli._pending_input.get_nowait() == "solo payload"
    assert claimed == [(event, "cli-idle")]
    assert completed == [(event, "claim-token")]


def test_async_delegation_results_keep_individual_delivery(monkeypatch):
    """Delegation results are never folded into the completion batch."""
    cli = _make_cli()
    deleg = {"type": "async_delegation", "delegation_id": "d1", "session_key": "cli-session"}
    completions = [_completion_event("proc_a"), _completion_event("proc_b")]
    pairs = [(deleg, "delegation result")] + [
        (evt, f"payload {evt['session_id']}") for evt in completions
    ]
    _, completed = _patch(monkeypatch, FakeRegistry(pairs))

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.qsize() == 2
    first = cli._pending_input.get_nowait()
    second = cli._pending_input.get_nowait()
    assert first == "delegation result"
    assert "proc_a" in second and "proc_b" in second
    assert len(completed) == 3


def test_unclaimed_event_produces_no_turn(monkeypatch):
    """A lost claim race drops the event without queueing or acking it."""
    cli = _make_cli()
    event = _completion_event("proc_race")
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        FakeRegistry([(event, "racy payload")]),
    )
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery", lambda evt, consumer: None
    )
    completed = []
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.qsize() == 0
    assert completed == []


def test_consumed_after_enqueue_stays_suppressed_with_real_registry(tmp_path):
    """Kill/wait consumption AFTER the event is queued must still suppress it.

    Uses the real ProcessRegistry: enqueue via _move_to_finished, then mark
    consumed (exactly what kill_process/wait do), then drain. The drain-time
    skip is what closes the enqueue/drain race (#104671 requirement 1).
    """
    import time
    from unittest.mock import patch

    from tools.process_registry import ProcessRegistry, ProcessSession

    registry = ProcessRegistry()
    with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
        for i in range(3):
            session = ProcessSession(
                id=f"proc_race_{i}",
                command="echo hi",
                task_id="t1",
                started_at=time.time(),
                exited=True,
                exit_code=0,
                output_buffer="hi",
                notify_on_complete=True,
            )
            registry._running[session.id] = session
            registry._move_to_finished(session)
        assert registry.completion_queue.qsize() == 3

    # Consumed after enqueue (kill/wait landed between queueing and drain).
    registry._completion_consumed.add("proc_race_1")

    drained = registry.drain_notifications(session_key="", owns_event=None)
    assert sorted(evt["session_id"] for evt, _text in drained) == [
        "proc_race_0",
        "proc_race_2",
    ]
