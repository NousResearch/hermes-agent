import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest

from plugins.memory.openviking import local_server, reaper


def _payload(**overrides):
    payload = {
        "endpoint": "http://127.0.0.1:1933",
        "pid": 1234,
        "create_time": 100.0,
        "task_ids": ["task-1"],
        "headers": {"X-OpenViking-Actor-Peer": "hermes"},
        "wait_timeout": 300.0,
        "poll_interval": 0.5,
        "stop_timeout": 10.0,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com",
        "https://127.0.0.1:1933",
        "ftp://localhost/resource",
        "http://192.168.1.10:1933",
        "not-a-url",
    ],
)
def test_payload_rejects_non_loopback_or_non_http_endpoint(endpoint):
    with pytest.raises(ValueError, match="endpoint must be local HTTP"):
        reaper._validated_payload(_payload(endpoint=endpoint))


def test_reaper_handoff_uses_windowless_python_and_shared_windows_detach_flags(
    monkeypatch,
    tmp_path,
):
    reaper_stdin = MagicMock()
    reaper_process = SimpleNamespace(stdin=reaper_stdin)
    popen = MagicMock(return_value=reaper_process)
    expected_flags = 0x10
    monkeypatch.setattr(
        local_server,
        "is_windows",
        lambda: True,
    )
    monkeypatch.setattr(
        "psutil.Process",
        MagicMock(return_value=SimpleNamespace(create_time=lambda: 123.5)),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway_windows._resolve_detached_python",
        lambda executable: ("C:/Python/pythonw.exe", Path("C:/venv"), []),
    )
    monkeypatch.setattr(
        local_server,
        "windows_detach_flags",
        lambda: expected_flags,
    )
    monkeypatch.setattr(local_server.subprocess, "Popen", popen)

    assert local_server.defer_owned_shutdown(
        SimpleNamespace(pid=4321),
        hermes_home=tmp_path,
        endpoint="http://127.0.0.1:1933",
        headers={},
        task_ids=set(),
    )

    argv = popen.call_args.args[0]
    assert argv[0] == "C:/Python/pythonw.exe"
    assert argv[1].endswith("plugins/memory/openviking/reaper.py")
    assert popen.call_args.kwargs["creationflags"] == expected_flags
    assert "start_new_session" not in popen.call_args.kwargs


def test_run_waits_for_terminal_task_then_drains_queues_before_stopping(monkeypatch):
    statuses = iter(["pending", "running", "completed"])
    events = []
    stop = MagicMock(return_value=True)
    monkeypatch.setattr(reaper, "_same_process", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        reaper,
        "_task_status",
        lambda *_args: events.append("task") or next(statuses),
    )
    monkeypatch.setattr(
        reaper,
        "_wait_for_processing",
        lambda *_args: events.append("queues"),
    )
    monkeypatch.setattr(
        reaper,
        "_stop_exact_process",
        lambda *args: events.append("stop") or stop(*args),
    )
    monkeypatch.setattr(reaper.time, "sleep", lambda _seconds: None)

    assert reaper.run(_payload()) == 0
    assert stop.call_args.args == (1234, 100.0, 10.0)
    assert events == ["task", "task", "task", "queues", "stop"]


def test_run_drains_queues_without_task_ids_before_stopping(monkeypatch):
    wait_for_processing = MagicMock()
    stop = MagicMock(return_value=True)
    monkeypatch.setattr(reaper, "_same_process", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(reaper, "_task_status", MagicMock())
    monkeypatch.setattr(reaper, "_wait_for_processing", wait_for_processing)
    monkeypatch.setattr(reaper, "_stop_exact_process", stop)

    assert reaper.run(_payload(task_ids=[])) == 0

    reaper._task_status.assert_not_called()
    wait_for_processing.assert_called_once()
    stop.assert_called_once_with(1234, 100.0, 10.0)


def test_run_leaves_server_when_queue_drain_cannot_be_verified(monkeypatch, caplog):
    stop = MagicMock()
    monkeypatch.setattr(reaper, "_same_process", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        reaper,
        "_wait_for_processing",
        MagicMock(side_effect=RuntimeError("queue wait failed")),
    )
    monkeypatch.setattr(reaper, "_stop_exact_process", stop)

    with caplog.at_level("ERROR", logger=reaper.__name__):
        assert reaper.run(_payload(task_ids=[])) == 2

    stop.assert_not_called()
    assert "Could not verify managed OpenViking queue drain" in caplog.text
    assert "leaving PID 1234 running" in caplog.text


def test_wait_for_processing_posts_bounded_system_wait(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"status":"ok","result":{"Embedding":{"processed":1}}}'

    class Opener:
        def open(self, request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    monkeypatch.setattr(reaper, "build_opener", lambda *_args: Opener())

    reaper._wait_for_processing(
        "http://127.0.0.1:1933",
        {"Authorization": "Bearer secret"},
        12.5,
    )

    request = captured["request"]
    assert request.full_url == "http://127.0.0.1:1933/api/v1/system/wait"
    assert request.method == "POST"
    assert json.loads(request.data) == {"timeout": 12.5}
    assert request.get_header("Authorization") == "Bearer secret"
    assert captured["timeout"] == 13.5


def test_wait_for_processing_rejects_queue_errors(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"status":"ok","result":{"Embedding":'
                b'{"processed":0,"error_count":1,"errors":["failed"]}}}'
            )

    opener = MagicMock()
    opener.open.return_value = Response()
    monkeypatch.setattr(reaper, "build_opener", lambda *_args: opener)

    with pytest.raises(RuntimeError, match="reported processing errors"):
        reaper._wait_for_processing(
            "http://127.0.0.1:1933",
            {},
            12.5,
        )


def test_task_status_encodes_id_and_validates_status(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"status":"ok","result":{"status":"running"}}'

    class Opener:
        def open(self, request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    monkeypatch.setattr(reaper, "build_opener", lambda *_args: Opener())

    assert (
        reaper._task_status(
            "http://127.0.0.1:1933",
            "task/with spaces",
            {"Authorization": "Bearer secret"},
        )
        == "running"
    )

    request = captured["request"]
    assert request.full_url.endswith("/api/v1/tasks/task%2Fwith%20spaces")
    assert request.get_header("Authorization") == "Bearer secret"
    assert captured["timeout"] == 5.0


def test_task_status_rejects_unknown_status(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"status":"ok","result":{"status":"mystery"}}'

    opener = MagicMock()
    opener.open.return_value = Response()
    monkeypatch.setattr(reaper, "build_opener", lambda *_args: opener)

    with pytest.raises(ValueError, match="unknown status"):
        reaper._task_status("http://127.0.0.1:1933", "task-1", {})


def test_run_leaves_server_when_task_status_cannot_be_verified(monkeypatch, caplog):
    stop = MagicMock()
    monkeypatch.setattr(reaper, "_same_process", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        reaper,
        "_task_status",
        MagicMock(side_effect=RuntimeError("connection failed")),
    )
    monkeypatch.setattr(reaper, "_stop_exact_process", stop)

    with caplog.at_level("ERROR", logger=reaper.__name__):
        assert reaper.run(_payload()) == 2

    stop.assert_not_called()
    assert "leaving PID 1234 running" in caplog.text


def test_run_is_bounded_when_task_remains_pending(monkeypatch, caplog):
    stop = MagicMock()
    monotonic = iter([10.0, 11.0])
    monkeypatch.setattr(reaper, "_same_process", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(reaper, "_task_status", MagicMock(return_value="pending"))
    monkeypatch.setattr(reaper, "_stop_exact_process", stop)
    monkeypatch.setattr(reaper.time, "monotonic", lambda: next(monotonic))

    with caplog.at_level("ERROR", logger=reaper.__name__):
        assert reaper.run(_payload(wait_timeout=0.5)) == 3

    stop.assert_not_called()
    assert "did not finish within 0.5 seconds" in caplog.text


def test_run_does_nothing_if_owned_process_already_exited(monkeypatch):
    stop = MagicMock()
    monkeypatch.setattr(reaper, "_same_process", MagicMock(return_value=None))
    monkeypatch.setattr(reaper, "_task_status", MagicMock())
    monkeypatch.setattr(reaper, "_stop_exact_process", stop)

    assert reaper.run(_payload()) == 0
    stop.assert_not_called()


def test_same_process_treats_zombie_as_exited(monkeypatch):
    process = SimpleNamespace(
        create_time=lambda: 100.0,
        status=lambda: psutil.STATUS_ZOMBIE,
    )
    monkeypatch.setattr(reaper.psutil, "Process", lambda _pid: process)

    assert reaper._same_process(1234, 100.0) is None


def test_stop_exact_process_revalidates_identity_before_force_kill(monkeypatch):
    process = MagicMock()
    process.wait.side_effect = [psutil.TimeoutExpired(10, pid=1234), None]
    same_process = MagicMock(side_effect=[process, process])
    monkeypatch.setattr(reaper, "_same_process", same_process)

    assert reaper._stop_exact_process(1234, 100.0, 10.0) is True

    process.send_signal.assert_called_once()
    process.kill.assert_called_once_with()
    assert same_process.call_count == 2


def test_stop_exact_process_never_force_kills_reused_pid(monkeypatch):
    process = MagicMock()
    process.wait.side_effect = psutil.TimeoutExpired(10, pid=1234)
    monkeypatch.setattr(
        reaper,
        "_same_process",
        MagicMock(side_effect=[process, None]),
    )

    assert reaper._stop_exact_process(1234, 100.0, 10.0) is True
    process.kill.assert_not_called()
