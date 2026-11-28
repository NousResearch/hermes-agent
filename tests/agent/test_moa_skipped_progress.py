"""Regression for #71243: a guarded MoA slot still finishes its progress step."""

import threading
from types import SimpleNamespace

from agent import moa_loop


def test_skipped_moa_reference_emits_terminal_progress(monkeypatch):
    slots = [
        {"provider": "synthetic", "model": "a"},
        {"provider": "moa", "model": "nested"},
        {"provider": "synthetic", "model": "b"},
    ]

    def fake_reference(slot, *_args, **_kwargs):
        return (f"{slot['provider']}:{slot['model']}", "advice", None)

    monkeypatch.setattr(moa_loop, "_run_reference", fake_reference)
    progress = []
    heartbeats = []

    outputs = moa_loop._run_references_parallel(
        slots,
        [{"role": "user", "content": "plan"}],
        progress_callback=lambda done, total, label: progress.append(
            (done, total, label)
        ),
        heartbeat_callback=lambda done, total, pending, elapsed: heartbeats.append(
            (done, total, pending)
        ),
    )

    assert [done for done, _, _ in progress] == [1, 2, 3]
    assert all(total == len(slots) for _, total, _ in progress)
    assert progress[0][2] == "moa:nested"
    assert {label for _, _, label in progress} == {
        "synthetic:a", "moa:nested", "synthetic:b"
    }
    assert heartbeats and heartbeats[0][:2] == (1, 3)
    assert [label for label, _, _ in outputs] == [
        "synthetic:a", "moa:nested", "synthetic:b"
    ]
    assert outputs[1][1].startswith("[skipped:")


def test_interrupted_reference_slot_emits_terminal_progress(monkeypatch):
    """A slot abandoned by an interrupt still emits its one terminal event."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    slots = [{"provider": "synthetic", "model": "slow"}]
    agent = SimpleNamespace(_interrupt_requested=False)
    progress = []
    outputs = []

    def slow_reference(slot, *_args, **_kwargs):
        started.set()
        release.wait(timeout=2)
        finished.set()
        return (f"{slot['provider']}:{slot['model']}", "advice", None)

    monkeypatch.setattr(moa_loop, "_run_reference", slow_reference)
    monkeypatch.setattr(moa_loop, "_REFERENCE_POLL_INTERVAL_S", 0.01)

    def run_fanout():
        outputs.extend(
            moa_loop._run_references_parallel(
                slots,
                [{"role": "user", "content": "stop"}],
                progress_callback=lambda done, total, label: progress.append(
                    (done, total, label)
                ),
                agent=agent,
            )
        )

    worker = threading.Thread(target=run_fanout)
    worker.start()
    try:
        assert started.wait(timeout=1), "the reference worker should start before interruption"
        agent._interrupt_requested = True
        worker.join(timeout=1)
        assert not worker.is_alive(), "the fan-out should return after settling the interrupted slot"
    finally:
        release.set()
        worker.join(timeout=1)

    assert finished.wait(timeout=1)
    assert progress == [(1, 1, "synthetic:slow")]
    assert len(outputs) == 1
    assert outputs[0][1] == moa_loop._INTERRUPTED_REFERENCE_NOTE
