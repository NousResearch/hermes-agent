"""ComfyUI must not replay submissions with an unknown outcome (#146034)."""

import importlib
import socket
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def transport(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "optional-skills/creative/comfyui/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    for name in ("_common", "run_workflow", "extract_schema"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    common = importlib.import_module("_common")
    runner = importlib.import_module("run_workflow")
    backoff = Mock()
    monkeypatch.setattr(common, "_sleep_backoff", backoff)
    yield common, runner, backoff
    for name in ("_common", "run_workflow", "extract_schema"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("failure,replayed", [
    # Unknown outcome: the request may have been queued.
    (TimeoutError, False), (ConnectionError, False), (OSError, False),
    (500, False), (502, False), (503, False), (504, False), (522, False), (524, False),
    # Provably not processed: safe to send again.
    ("RequestNotSent", True), (408, True), (429, True),
])
def test_submission_replays_only_when_not_processed(transport, monkeypatch, failure, replayed):
    common, runner, backoff = transport
    if failure == "RequestNotSent":
        failure = common.RequestNotSent
    response = common.HTTPResponse(200, {}, b'{"prompt_id":"once"}', "http://localhost/prompt")
    first = (failure("response lost") if isinstance(failure, type)
             else common.HTTPResponse(failure, {}, b'{}', response.url))
    send = Mock(side_effect=[first, response])
    monkeypatch.setattr(common, "_http_once", send)
    client = runner.ComfyRunner(host="http://localhost", client_id="test-client")
    workflow = {"1": {"class_type": "Example", "inputs": {}}}

    if replayed:
        assert client.submit(workflow)["prompt_id"] == "once"
    elif isinstance(failure, type):
        with pytest.raises(failure, match="response lost"):
            client.submit(workflow)
    else:
        assert client.submit(workflow)["_http_error"] == failure

    assert send.call_count == (2 if replayed else 1)
    assert send.call_args.kwargs["json_body"] == {"prompt": workflow, "client_id": "test-client"}
    assert backoff.call_count == (1 if replayed else 0)


@pytest.mark.parametrize("method,body", [
    ("GET", {}), ("head", {}), ("OPTIONS", {}),
    ("GET", {"json_body": {}}), ("GET", {"data": b""}),
    ("GET", {"files": {}}), ("GET", {"form": {}}),
    ("POST", {}), ("PUT", {}), ("DELETE", {}),
])
@pytest.mark.parametrize("failure", [TimeoutError, 502])
def test_only_bodyless_safe_requests_retry(transport, monkeypatch, method, body, failure):
    common, _runner, backoff = transport
    # Multipart validation is independent of the retry loop tested here.
    monkeypatch.setattr(common, "HAS_REQUESTS", True)
    response = common.HTTPResponse(200, {}, b'{}', "http://localhost/resource")
    first = (failure("response lost") if isinstance(failure, type)
             else common.HTTPResponse(failure, {}, b'{}', response.url))
    send = Mock(side_effect=[first, response])
    monkeypatch.setattr(common, "_http_once", send)
    safe = method.upper() in {"GET", "HEAD", "OPTIONS"} and not body

    if not safe and isinstance(failure, type):
        with pytest.raises(failure, match="response lost"):
            common.http_request(method, response.url, **body)
    else:
        result = common.http_request(method, response.url, **body)
        assert result.status == (200 if safe else failure)

    assert send.call_count == (2 if safe else 1)
    assert backoff.call_count == (1 if safe else 0)


def _dead_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def submission_server():
    """Counts POSTs it fully reads; 'drop' then closes, 'redirect' answers 303 to a dead port."""
    listener = socket.create_server(("127.0.0.1", 0))
    seen = {"posts": 0, "mode": "drop", "dead": _dead_port()}

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                request = b""
                while b"\r\n\r\n" not in request:
                    request += conn.recv(65536)
                head, _, body = request.partition(b"\r\n\r\n")
                length = int(next(line.split(b":")[1] for line in head.split(b"\r\n")
                                  if line.lower().startswith(b"content-length")))
                while len(body) < length:
                    body += conn.recv(65536)
                seen["posts"] += 1
                if seen["mode"] == "redirect":
                    conn.sendall(b"HTTP/1.1 303 See Other\r\nLocation: http://127.0.0.1:%d/receipt"
                                 b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" % seen["dead"])

    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}", seen
    listener.close()


@pytest.mark.parametrize("client", ["requests", "stdlib"])
@pytest.mark.parametrize("scenario", ["drop", "redirect", "refused", "proxy_refused"])
def test_real_clients_replay_only_unsent_submissions(transport, monkeypatch, submission_server,
                                                     client, scenario):
    common, _runner, _backoff = transport
    if client == "stdlib":
        monkeypatch.setattr(common, "HAS_REQUESTS", False)
    elif not common.HAS_REQUESTS:
        pytest.skip("requests is not installed")
    base, seen = submission_server
    attempts = Mock(wraps=common._http_once)
    monkeypatch.setattr(common, "_http_once", attempts)
    for key in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    url = f"{base}/prompt"
    if scenario == "redirect":
        seen["mode"] = "redirect"
    elif scenario == "refused":
        url = f"http://127.0.0.1:{_dead_port()}/prompt"
    elif scenario == "proxy_refused":
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{_dead_port()}")

    with pytest.raises(OSError) as info:
        common.http_post(url, json_body={"prompt": {}, "client_id": "c"}, timeout=5)

    never_sent = scenario in {"refused", "proxy_refused"}
    assert isinstance(info.value, common.RequestNotSent) is never_sent
    assert attempts.call_count == (common.DEFAULT_RETRIES if never_sent else 1)
    assert seen["posts"] == (0 if never_sent else 1)
