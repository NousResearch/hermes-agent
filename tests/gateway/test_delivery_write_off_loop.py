"""Cron artifact writes must not park every adapter on the gateway event loop."""
import asyncio
from datetime import datetime
from pathlib import Path
import threading

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget


@pytest.mark.parametrize("local", [True, False])
def test_real_delivery_write_does_not_stall_event_loop(tmp_path, monkeypatch, local):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    write_threads = []
    count_lock = threading.Lock()
    real_write = Path.write_text

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 29, 12, 0, 0)

    monkeypatch.setattr("gateway.delivery.datetime", FixedClock)

    def parked_write(path, text, *args, **kwargs):
        if "output" not in path.parts:
            return real_write(path, text, *args, **kwargs)
        with count_lock:
            write_threads.append(threading.get_ident())
            first = len(write_threads) == 1
        if not first:
            return real_write(path, text, *args, **kwargs)
        # Actual partially completed file I/O: a concurrent open/truncate/write
        # on this same path can mix the two payloads after this writer resumes.
        with path.open("w", encoding=kwargs.get("encoding", "utf-8")) as output:
            half = len(text) // 2
            output.write(text[:half])
            output.flush()
            entered.set()
            release.wait(timeout=5)
            output.write(text[half:])
        return len(text)

    monkeypatch.setattr(Path, "write_text", parked_write)

    class Adapter:
        splits_long_messages = True

        def __init__(self):
            self.received = []

        async def send(self, chat_id, content, metadata=None):
            self.received.append(content)
            return {"success": True}

    adapter = Adapter()
    # Separate routers share the same destination, as independent turn/cron
    # callers can. The old loop serialized their writes across instances too.
    routers = [DeliveryRouter(GatewayConfig(), {Platform.DISCORD: adapter}) for _ in range(2)]
    target = DeliveryTarget.parse("local" if local else "discord:123")
    contents = [f"{marker} 数据" * 1000 for marker in ("FIRST", "SECOND")]

    async def scenario():
        first = asyncio.create_task(routers[0].deliver(contents[0], [target], job_id="job1", metadata={"job_id": "job1"}))
        second = None
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            assert not first.done(), "disk I/O parked the event loop until the write completed"
            second = asyncio.create_task(routers[1].deliver(contents[1], [target], job_id="job1", metadata={"job_id": "job1"}))
            # Give a second worker the chance to enter the real writer. It must
            # wait off-loop while the first holds a partially written artifact.
            await asyncio.sleep(0.1)
            assert len(write_threads) == 1, "concurrent writers opened the same artifact"
            assert not second.done()
            assert write_threads[0] != threading.get_ident()
        finally:
            release.set()
            results = await asyncio.gather(first, *([second] if second is not None else []))
        assert all(result[target.to_string()]["success"] for result in results)
        if local:
            path = Path(results[-1]["local"]["result"]["path"])
            assert path.read_text(encoding="utf-8").endswith(contents[-1])
        else:
            assert sorted(adapter.received) == sorted(contents)
            path = next((tmp_path / "cron" / "output").glob("job1_*.txt"))
            assert path.read_text(encoding="utf-8") == contents[-1]

    asyncio.run(scenario())


@pytest.mark.parametrize("local", [True, False])
def test_worker_delivery_preserves_a_b_a_profile_scope(tmp_path, monkeypatch, local):
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home

    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    homes = [tmp_path / "profiles" / name for name in ("a", "b")]
    for home in homes:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    real_write = Path.write_text
    writes = []

    def record_write(path, text, *args, **kwargs):
        if "output" in path.parts:
            writes.append((path, get_hermes_home(), threading.get_ident()))
        return real_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_write)

    class Adapter:
        splits_long_messages = True

        async def send(self, chat_id, content, metadata=None):
            return {"success": True}

    async def scenario():
        for index, home in enumerate((homes[0], homes[1], homes[0])):
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                router = DeliveryRouter(GatewayConfig(multiplex_profiles=True), {Platform.DISCORD: Adapter()})
                target = DeliveryTarget.parse("local" if local else "discord:123")
                content = f"profile-{index} 数据" * 1000
                result = await router.deliver(content, [target], job_id=f"job{index}", metadata={"job_id": f"job{index}"})
                assert result[target.to_string()]["success"]
                path, worker_home, worker_thread = writes[-1]
                assert path.is_relative_to(home)
                assert worker_home == home
                assert worker_thread != threading.get_ident()
                assert content in path.read_text(encoding="utf-8")
        assert not (launch / "cron" / "output").exists()
        assert get_hermes_home() == launch

    set_multiplex_active(True)
    try:
        asyncio.run(scenario())
    finally:
        set_multiplex_active(False)
