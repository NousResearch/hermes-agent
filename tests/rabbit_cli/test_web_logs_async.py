import asyncio
import threading

from rabbit_cli.web_routers import status


def test_get_logs_yields_while_reading_and_filtering(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "agent.log").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(status, "get_rabbit_home", lambda: tmp_path)

    loop_ran = threading.Event()

    def blocking_read(*args, **kwargs):
        assert loop_ran.wait(2), "log reading blocked the event loop"
        return ["fixture"]

    import rabbit_cli.logs
    monkeypatch.setattr(rabbit_cli.logs, "_read_tail", blocking_read)

    async def exercise():
        asyncio.get_running_loop().call_soon(loop_ran.set)
        return await status.get_logs(file="agent", lines=100)

    assert asyncio.run(exercise()) == {"file": "agent", "lines": ["fixture"]}
