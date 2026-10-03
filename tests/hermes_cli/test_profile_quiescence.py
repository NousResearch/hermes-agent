"""Profile mutation waits for a different process's real SessionDB sweep.

Regression for #127923: a best-effort rescan is not a completed writer barrier.
The child owns the real gateway control socket and an opener paused after SQLite
connect; only service/process shutdown unrelated to this protocol is substituted.
"""

import asyncio
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest


def _wait_for(predicate, *, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("child/operation did not reach its synchronization point")
        time.sleep(0.02)


async def _gateway_child(root, control):
    from gateway.config import GatewayConfig
    from gateway import host_rendezvous as hr
    from gateway.run import GatewayRunner, _start_gateway_start_control_socket
    from gateway.status import flush_runtime_status
    import hermes_state_registry as registry

    alpha, beta = root / "profiles/alpha", root / "profiles/beta"
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._primary_profile_name = "alpha"
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._failed_platforms = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner.pairing_store = MagicMock()
    runner.pairing_stores = {}
    runner._adapter_disconnect_timeout_secs = lambda: 2.0
    assert hr.claim_host_lock(hr.ROLE_GATEWAY)[0] is hr.HostLockOutcome.ACQUIRED
    runner._record_served_profiles("alpha", [("default", root), ("alpha", alpha), ("beta", beta)])
    flush_runtime_status()

    # A -> B -> A uses real imports and separate homes in the gateway process.
    alpha_db = registry.acquire(alpha / "state.db")
    alpha_db.create_session("alpha-before", source="cli")
    opened, resume = [], threading.Event()
    real_open, real_sweep = registry._open_session_db, registry.close_all_under

    def paused_open(path):
        db = real_open(path)
        if Path(path).parent == beta:
            real_close = db.close

            def close():
                if (control / "fail-close").exists():
                    from hermes_state_dbfile import RetiredGenerationCaptureError
                    (control / "close-failed.json").write_text(json.dumps({
                        "beta_live": db._conn is not None,
                        "alpha_live": alpha_db._conn is not None,
                    }))
                    raise RetiredGenerationCaptureError("fixture refuses destructive close")
                return real_close()

            db.close = close
            opened.append(db)
            (control / "connected").touch()
            if not resume.wait(45.0):
                raise RuntimeError("parent did not resume the connected opener")
        return db

    def observed_sweep(directory, **kwargs):
        if Path(directory) != beta:
            return real_sweep(directory, **kwargs)
        (control / "sweep-entered").touch()
        result = real_sweep(directory, **kwargs)
        (control / "swept.json").write_text(json.dumps({
            "closed": bool(opened) and all(db._conn is None for db in opened),
            "alpha_live": alpha_db._conn is not None,
        }))
        return result

    registry._open_session_db, registry.close_all_under = paused_open, observed_sweep

    def open_writer():
        try:
            registry.acquire(beta / "state.db")
        except Exception as exc:
            (control / "opener-error").write_text(repr(exc))

    writer = threading.Thread(target=open_writer, daemon=True)
    writer.start()
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    (control / "ready").touch()
    try:
        while not (control / "stop").exists():
            if (control / "resume").exists():
                resume.set()
            if (control / "probe-late").exists() and not (control / "late.json").exists():
                try:
                    opened[0].create_session("beta-late", source="cli")
                    refused = False
                except Exception:
                    refused = True
                alpha_db.create_session("alpha-after", source="cli")
                (control / "late.json").write_text(json.dumps({
                    "refused": refused, "beta_closed": opened[0]._conn is None,
                    "alpha_live": alpha_db._conn is not None,
                    "alpha_after": alpha_db.get_session("alpha-after") is not None,
                }))
            await asyncio.sleep(0.02)
    finally:
        resume.set()
        await server.stop()
        await asyncio.to_thread(writer.join, 10.0)
        await asyncio.to_thread(registry.close_all)
        hr.clear_record(hr.ROLE_GATEWAY)
        hr.release_host_lock(hr.ROLE_GATEWAY)


@contextlib.contextmanager
def _remote_gateway(tmp_path, monkeypatch):
    from gateway.host_attach import invalidate_host_gateway_cache
    from hermes_cli import profiles

    root, control = tmp_path / ".hermes", tmp_path / "control"
    control.mkdir()
    root.mkdir()
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n")
    for name in ("alpha", "beta"):
        directory = root / "profiles" / name
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(control / "host-locks"))
    # The protocol, discovery record and remote sweep are real. These operations
    # would touch the host's supervisors or unrelated backends and are not under test.
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_profile_backends", "_stop_bot_desktop"):
        monkeypatch.setattr(profiles, name, lambda *args: False)
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda *_: False)
    monkeypatch.setattr(profiles, "_purge_identity", lambda *_: True)
    monkeypatch.setattr(profiles, "check_alias_collision", lambda *_: "test skips shell alias")
    monkeypatch.setattr(profiles, "remove_wrapper_script", lambda *_: False)
    invalidate_host_gateway_cache()
    env = dict(os.environ, HERMES_HOME=str(root / "profiles/alpha"))
    with (control / "child.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, __file__, "--gateway", str(root), str(control)],
            cwd=Path(__file__).resolve().parents[2], env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            _wait_for(lambda: child.poll() is not None or
                      ((control / "ready").exists() and (control / "connected").exists()))
            assert child.poll() is None, (control / "child.log").read_text()
            yield root, control
        finally:
            (control / "fail-close").unlink(missing_ok=True)
            (control / "resume").touch()
            (control / "stop").touch()
            try:
                child.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5.0)
            invalidate_host_gateway_cache()


def _start_mutation(root, operation):
    from hermes_cli import profiles

    result, done = [], threading.Event()

    def run():
        try:
            result.append(profiles.delete_profile("beta", yes=True) if operation == "delete"
                          else profiles.rename_profile("beta", "renamed"))
        except Exception as exc:
            result.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return result, done, worker


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_remote_ack_precedes_profile_mutation(tmp_path, monkeypatch, operation):
    from hermes_cli import profiles

    with _remote_gateway(tmp_path, monkeypatch) as (root, control):
        mutations = []
        real_remove, real_rename = profiles._rmtree_with_retry, Path.rename

        def observe():
            mutations.append(True)
            assert (control / "swept.json").exists(), "filesystem mutation preceded the remote sweep"
            assert json.loads((control / "swept.json").read_text()) == {"closed": True, "alpha_live": True}

        def remove(path, *args):
            observe()
            return real_remove(path, *args)

        def rename(path, target):
            if path == root / "profiles/beta":
                observe()
            return real_rename(path, target)

        monkeypatch.setattr(profiles, "_rmtree_with_retry", remove)
        monkeypatch.setattr(Path, "rename", rename)
        result, done, worker = _start_mutation(root, operation)
        try:
            _wait_for(lambda: (control / "sweep-entered").exists() or done.is_set())
            assert not done.is_set(), result
            assert (root / "profiles/beta/config.yaml").exists()
            assert mutations == []
        finally:
            (control / "resume").touch()
        assert done.wait(15.0), "mutation did not finish after the remote opener resumed"
        worker.join(5.0)
        assert result and isinstance(result[0], Path), result
        assert len(mutations) == 1
        assert not (root / "profiles/beta").exists()
        assert (root / "profiles/alpha/config.yaml").exists()
        (control / "probe-late").touch()
        _wait_for(lambda: (control / "late.json").exists())
        assert json.loads((control / "late.json").read_text()) == {
            "refused": True, "beta_closed": True, "alpha_live": True, "alpha_after": True,
        }
        assert not (root / "profiles/beta").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("operation", ["delete", "rename"])
@pytest.mark.parametrize("failure_mode", ["timeout", "close-failure"])
def test_pending_remote_sweep_preserves_profile_directory(tmp_path, monkeypatch, operation, failure_mode):
    from gateway import control_socket

    with _remote_gateway(tmp_path, monkeypatch) as (root, control):
        # Forward to the real socket with a bounded client deadline; the paused
        # SQLite open still outlasts that deadline, so there can be no positive ACK.
        real_request = control_socket.request_unserve_profile

        def request(home, name, **kwargs):
            if failure_mode == "timeout":
                kwargs["timeout"] = 2.0
            return real_request(home, name, **kwargs)

        monkeypatch.setattr(control_socket, "request_unserve_profile", request)
        if failure_mode == "close-failure":
            (control / "fail-close").touch()
        result, done, worker = _start_mutation(root, operation)
        if failure_mode == "close-failure":
            _wait_for(lambda: (control / "sweep-entered").exists() or done.is_set())
            (control / "resume").touch()
        assert done.wait(15.0), "a pending remote barrier did not fail within its deadline"
        worker.join(5.0)
        assert len(result) == 1 and isinstance(result[0], RuntimeError), result
        assert (root / "profiles/beta/config.yaml").exists()
        assert not (root / "profiles/renamed").exists()
        assert not (control / "swept.json").exists()
        assert (root / "profiles/alpha/config.yaml").exists()
        if failure_mode == "close-failure":
            assert json.loads((control / "close-failed.json").read_text()) == {
                "beta_live": True, "alpha_live": True,
            }


if __name__ == "__main__" and sys.argv[1:2] == ["--gateway"]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    asyncio.run(_gateway_child(Path(sys.argv[2]), Path(sys.argv[3])))
