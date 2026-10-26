"""Process-level shared-metrics marks from background work that outlives the turn it was spawned in.

A thread started with the turn's contextvars (a turn-end compaction pass, title generation) still
holds that turn in ``relay_runtime._CURRENT_TURN`` after the turn closes. A process mark has no
owning task and opens no scope, so the closed turn must not swallow it; a turn Relay skipped
(a concurrent turn on the same session) keeps gating.
"""

from __future__ import annotations

import contextvars
import threading

import pytest

from agent import relay_runtime
from hermes_cli import plugins
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_events as events
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.plugins import PluginManager

_AUX_USAGE = {"input_tokens": 1000, "output_tokens": 10}


@pytest.fixture
def relay_home(tmp_path, monkeypatch):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    manager = PluginManager()
    manager._discovered = True  # a runtime with zero plugins loaded (#64178)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    yield tmp_path / "hermes-home"
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def _emit_process_marks() -> None:
    events.record_slash_command(command="help", surface="cli")
    relay_shared_metrics.observe_lifecycle(
        "post_auxiliary_call", aux_task="title_generation", model="google/gemini-2.5-flash",
        provider="openrouter", usage=_AUX_USAGE,
    )


def _recorded(home) -> dict[str, int]:
    """Slash-command count and title-generation input tokens persisted in the store."""
    relay_shared_metrics.shutdown_runtimes()
    root = home / "telemetry" / "shared_metrics"
    found = {"slash": 0, "aux_input": 0}
    for row in SharedMetricsStore(root / "metrics.sqlite3", root / "outbox").counter_snapshot():
        dims = row["dimensions"]
        if row["metric_name"] == "hermes.slash_command.count" and dims.get("command") == "help":
            found["slash"] += row["value"]
        elif row["metric_name"] == "hermes.model_tokens.sum" and dims.get("token_type") == "input":
            found["aux_input"] += row["value"]
    return found


def _run_after(context: contextvars.Context, before_emit) -> None:
    """Run the marks on a thread holding ``context``, after ``before_emit`` ran on this thread."""
    go = threading.Event()
    worker = threading.Thread(target=context.run, args=(lambda: (go.wait(5), _emit_process_marks()),))
    worker.start()
    before_emit()
    go.set()
    worker.join(10)
    assert not worker.is_alive()


def test_marks_from_a_thread_that_outlives_its_turn_are_recorded(relay_home):
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="s1", platform="cli",
    )
    turn = coordinator.begin_turn(lease, turn_id="turn-1", task_id="t1")
    relay_shared_metrics.start_task_run(session_id="s1", task_id="t1", platform="cli")
    spawned_in_turn = contextvars.copy_context()

    def close_turn() -> None:
        relay_shared_metrics.finish_task_run(
            session_id="s1", task_id="t1", platform="cli", result={"completed": True},
        )
        coordinator.end_turn(turn, outcome="success")
        coordinator.release_conversation(lease)

    _run_after(spawned_in_turn, close_turn)

    assert spawned_in_turn.run(relay_runtime.current_turn) is turn and turn.closed
    assert _recorded(relay_home) == {"slash": 1, "aux_input": 1000}


def test_a_turn_relay_skipped_still_gates_marks(relay_home):
    coordinator = relay_runtime.SESSION_COORDINATOR
    lease = coordinator.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="s1", platform="cli",
    )
    first = coordinator.begin_turn(lease, turn_id="turn-1", task_id="t1")
    second = coordinator.begin_turn(lease, turn_id="turn-2", task_id="t2")
    assert second.relay_enabled is False
    inside_skipped = contextvars.copy_context()

    def close_turns() -> None:
        coordinator.end_turn(second, outcome="success")
        coordinator.end_turn(first, outcome="success")
        coordinator.release_conversation(lease)

    _run_after(inside_skipped, close_turns)

    assert _recorded(relay_home) == {"slash": 0, "aux_input": 0}
