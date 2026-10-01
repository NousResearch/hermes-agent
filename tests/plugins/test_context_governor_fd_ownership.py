"""Concurrency regression tests for operation-local governed descriptors."""

import threading
from types import SimpleNamespace

import pytest

from plugins.context_engine._context_governor import ContextGovernorEngine




def _failure_capabilities() -> dict:
    return {
        "failure_envelope": {
            "schema": "ContextGovernorFailureV1",
            "flag": "--failure-envelope-v1",
            "stream": "stderr",
        }
    }

class _Binding:
    def __init__(self, marker: int) -> None:
        self.marker = marker
        self.pass_fds = (marker, marker + 100)
        self.close_count = 0

    def command_args(self) -> list[str]:
        return ["--governed-key-fd", str(self.marker), "--governed-snapshot-fd", str(self.marker + 100)]

    def close(self) -> None:
        self.close_count += 1


def test_certified_subprocess_owns_its_binding_across_overlap() -> None:
    engine = ContextGovernorEngine.__new__(ContextGovernorEngine)
    bindings = [_Binding(11), _Binding(22)]
    binding_lock = threading.Lock()

    def active_binding() -> _Binding:
        with binding_lock:
            return bindings.pop(0)

    engine._key_state = SimpleNamespace(active_binding=active_binding)
    engine._capabilities = _failure_capabilities()
    a_entered = threading.Event()
    release_a = threading.Event()
    calls: list[tuple[list[str], tuple[int, ...]]] = []

    def run_json(args, payload, *, pass_fds=()):
        del payload
        marker = int(args[args.index("--governed-key-fd") + 1])
        calls.append((list(args), tuple(pass_fds)))
        if marker == 11:
            a_entered.set()
            assert release_a.wait(2)
        return {"marker": marker}

    engine._run_json = run_json
    results: dict[str, dict] = {}
    thread_a = threading.Thread(
        target=lambda: results.setdefault("a", engine._run_certified_json(["compact-v2"], {}))
    )
    thread_a.start()
    assert a_entered.wait(2)
    results["b"] = engine._run_certified_json(["search"], {})
    assert bindings == []
    # B may close only its own binding while A remains in flight.
    assert calls[0][1] == (11, 111)
    assert calls[1][1] == (22, 122)
    release_a.set()
    thread_a.join(2)
    assert not thread_a.is_alive()
    assert results == {"a": {"marker": 11}, "b": {"marker": 22}}
    for args, pass_fds in calls:
        argv_fds = {
            int(args[args.index("--governed-key-fd") + 1]),
            int(args[args.index("--governed-snapshot-fd") + 1]),
        }
        assert argv_fds == set(pass_fds)


def test_certified_binding_closes_once_on_timeout_without_touching_peer() -> None:
    engine = ContextGovernorEngine.__new__(ContextGovernorEngine)
    timed_out = _Binding(31)
    peer = _Binding(42)
    bindings = iter((timed_out, peer))
    engine._key_state = SimpleNamespace(active_binding=lambda: next(bindings))
    engine._capabilities = _failure_capabilities()

    def run_json(args, payload, *, pass_fds=()):
        del payload, pass_fds
        marker = int(args[args.index("--governed-key-fd") + 1])
        if marker == 31:
            raise TimeoutError("child timeout")
        assert timed_out.close_count == 1
        assert peer.close_count == 0
        return {"ok": True}

    engine._run_json = run_json
    with pytest.raises(TimeoutError, match="child timeout"):
        engine._run_certified_json(["compact-v2"], {})
    assert engine._run_certified_json(["search"], {}) == {"ok": True}
    assert timed_out.close_count == 1
    assert peer.close_count == 1


def test_detached_worker_binding_does_not_close_new_operation_binding() -> None:
    engine = ContextGovernorEngine.__new__(ContextGovernorEngine)
    old_worker = _Binding(51)
    new_worker = _Binding(61)
    bindings = iter((old_worker, new_worker))
    setattr(engine, "_key_state", SimpleNamespace(active_binding=lambda: next(bindings)))
    engine._capabilities = _failure_capabilities()
    old_started = threading.Event()
    release_old = threading.Event()

    def run_json(args, payload, *, pass_fds=()):
        del payload
        marker = int(args[args.index("--governed-key-fd") + 1])
        assert tuple(pass_fds) == (marker, marker + 100)
        if marker == old_worker.marker:
            old_started.set()
            assert release_old.wait(2)
        return {"marker": marker}

    engine._run_json = run_json
    results: dict[str, dict] = {}
    old_thread = threading.Thread(
        target=lambda: results.setdefault("old", engine._run_certified_json(["compact-v2"], {}))
    )
    old_thread.start()
    assert old_started.wait(2)

    # The host may fence-cancel the old worker while it winds down. A new
    # operation must remain independently paired with its own descriptors.
    results["new"] = engine._run_certified_json(["prepare-v2"], {})
    assert old_worker.close_count == 0
    assert new_worker.close_count == 1
    release_old.set()
    old_thread.join(2)
    assert not old_thread.is_alive()
    assert old_worker.close_count == 1
    assert results == {"old": {"marker": 51}, "new": {"marker": 61}}


def test_duplicate_compressions_do_not_share_pending_settlement() -> None:
    engine = ContextGovernorEngine.__new__(ContextGovernorEngine)
    engine.session_id = "serialized-session"
    engine._lineage_session_id = "serialized-session"
    engine.last_receipt_id = "ctxr_parent"
    engine._pending_admission = None
    engine.last_outcome = None
    engine.last_error = None
    engine._last_compress_aborted = False
    engine._last_summary_error = None
    engine._last_summary_fallback_used = False
    engine._last_compression_made_progress = False
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def compress_once(messages, current_tokens, focus_topic):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(2)
        engine._pending_admission = {"receipt_id": "ctxr_pending"}
        return [{"role": "user", "content": "bounded result"}]

    setattr(engine, "_compress_once", compress_once)
    engine.validate_pending_compression = lambda _messages: True
    messages = [{"role": "user", "content": "same request"}]
    results = {}
    owner = threading.Thread(
        target=lambda: results.setdefault("owner", engine.compress(messages, 100, "focus"))
    )
    follower = threading.Thread(
        target=lambda: results.setdefault("follower", engine.compress(messages, 100, "focus"))
    )
    owner.start()
    assert started.wait(2)
    follower.start()
    threading.Event().wait(0.05)
    assert calls == 1
    release.set()
    owner.join(2)
    follower.join(2)
    assert not owner.is_alive() and not follower.is_alive()
    assert calls == 1
    assert results["owner"] == [{"role": "user", "content": "bounded result"}]
    assert results["follower"] == messages
    assert engine._pending_admission == {"receipt_id": "ctxr_pending"}
    assert engine.last_outcome == {
        "kind": "pending_host_commit",
        "receipt_id": "ctxr_pending",
    }
