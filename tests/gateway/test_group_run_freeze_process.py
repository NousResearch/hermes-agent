"""Separate SQLite writers establish the admission/freeze/control ordering."""

import multiprocessing
from contextlib import closing, contextmanager

import pytest

from gateway.platforms.api_server_run_idempotency import GroupRunFreezeError, RunIdempotencyStore
from gateway.platforms.api_server_run_scope import room_run_scope_key
from tests.gateway.test_group_run_scope import IDENTITY


def _actor(path, identity, operation, pause, channel):
    store = None
    try:
        store = RunIdempotencyStore(path)
        paused = False

        def hold_writer(sql):
            nonlocal paused
            prefix = "INSERT INTO run_idempotency(" if operation == "reserve" else "INSERT INTO group_run_freezes"
            if pause and not paused and sql.startswith(prefix):
                paused = True
                channel.send({"state": "holding_writer"})
                if channel.poll(10):
                    channel.recv()

        store._conn.set_trace_callback(hold_writer)
        channel.send({"state": "ready"})
        if not channel.poll(10):
            raise RuntimeError("start signal timed out")
        channel.recv()
        channel.send({"state": "attempting"})
        if operation == "reserve":
            outcome, record = store.reserve(room_run_scope_key(identity), "late-key", "late-fp", "late-run", {"status": "queued"})
            result = {"outcome": outcome, "run_id": record["run_id"]}
        elif operation == "freeze":
            result = store.freeze_room_scope(identity, "command")
        else:
            with store.group_control_open(room_run_scope_key(identity)) as allowed:
                # Test-only scheduling barrier; production permits no I/O/store re-entry here.
                channel.send({"state": "holding_writer"})
                if not channel.poll(10):
                    raise RuntimeError("release signal timed out")
                channel.recv()
                result = {"allowed": allowed}
        channel.send({"state": "done", "result": result})
    except GroupRunFreezeError as exc:
        channel.send({"state": "done", "error": exc.code})
    except Exception as exc:
        channel.send({"state": "unexpected", "error": type(exc).__name__})
    finally:
        if store is not None:
            store.close()
        channel.close()


def _receive(channel, state):
    assert channel.poll(12), f"worker did not report {state}"
    result = channel.recv()
    assert result["state"] == state, result
    return result


@contextmanager
def _workers(path, specs):
    context = multiprocessing.get_context("spawn")
    children = []
    try:
        for identity, operation, pause in specs:
            parent, child = context.Pipe()
            process = context.Process(target=_actor, args=(str(path), identity, operation, pause, child))
            process.start()
            child.close()
            children.append((process, parent))
            _receive(parent, "ready")
        yield [channel for _, channel in children]
    finally:
        for process, channel in children:
            try:
                channel.send("release")
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            channel.close()
            assert not process.is_alive(), "owned test worker did not stop"


@pytest.mark.parametrize("first", ["reserve", "freeze", "control"])
def test_cross_process_barrier_orders_admission_and_control(tmp_path, first):
    path = tmp_path / "runs.db"
    scope = room_run_scope_key(IDENTITY)
    with closing(RunIdempotencyStore(str(path))) as store:
        store.reserve(scope, "seed-key", "seed-fp", "seed", {"status": "queued"})
    second = "reserve" if first == "freeze" else "freeze"
    with _workers(path, [(IDENTITY, first, True), (IDENTITY, second, False)]) as channels:
        channels[0].send("go")
        _receive(channels[0], "attempting")
        _receive(channels[0], "holding_writer")
        channels[1].send("go")
        _receive(channels[1], "attempting")
        channels[0].send("release")
        results = [_receive(channel, "done") for channel in channels]
    if first == "freeze":
        assert results[1].get("error") == "group_work_frozen"
    elif first == "reserve":
        assert results[0]["result"]["outcome"] == "created"
    else:
        assert results[0]["result"]["allowed"] is True
    with closing(RunIdempotencyStore(str(path))) as restarted:
        snapshot = restarted.room_stop_snapshot("command")
        assert {row["run_id"] for row in snapshot["runs"]} == ({"seed", "late-run"} if first == "reserve" else {"seed"})
        assert restarted.is_scope_frozen(scope)
        with restarted.group_control_open(scope) as allowed:
            assert allowed is False


def test_two_processes_cannot_bind_one_command_to_different_scopes(tmp_path):
    path = tmp_path / "runs.db"
    other = {**IDENTITY, "member_id": "other-member"}
    with closing(RunIdempotencyStore(str(path))) as store:
        for number, identity in enumerate((IDENTITY, other)):
            store.reserve(room_run_scope_key(identity), "seed", "fp", f"seed-{number}", {"status": "queued"})
    with _workers(path, [(IDENTITY, "freeze", False), (other, "freeze", False)]) as channels:
        for channel in channels:
            channel.send("go")
        for channel in channels:
            _receive(channel, "attempting")
        results = [_receive(channel, "done") for channel in channels]
    assert sum("result" in result for result in results) == 1
    assert [result["error"] for result in results if "error" in result] == ["group_stop_command_conflict"]
    with closing(RunIdempotencyStore(str(path))) as restarted:
        assert sum(restarted.is_scope_frozen(room_run_scope_key(value)) for value in (IDENTITY, other)) == 1
