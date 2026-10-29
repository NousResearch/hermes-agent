"""ComfyUI must not replay submissions with an unknown outcome (#146034)."""

import importlib
import sys
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


@pytest.mark.parametrize("failure", [TimeoutError, ConnectionError, OSError, 429, 503])
def test_submission_does_not_replay_after_response_failure(transport, monkeypatch, failure):
    common, runner, backoff = transport
    response = common.HTTPResponse(200, {}, b'{"prompt_id":"duplicate"}', "http://localhost/prompt")
    first = (failure("response lost") if isinstance(failure, type)
             else common.HTTPResponse(failure, {}, b'{}', response.url))
    send = Mock(side_effect=[first, response])
    monkeypatch.setattr(common, "_http_once", send)
    client = runner.ComfyRunner(host="http://localhost", client_id="test-client")
    workflow = {"1": {"class_type": "Example", "inputs": {}}}

    if isinstance(failure, type):
        with pytest.raises(failure, match="response lost"):
            client.submit(workflow)
    else:
        assert client.submit(workflow)["_http_error"] == failure

    assert send.call_count == 1
    assert send.call_args.kwargs["json_body"] == {"prompt": workflow, "client_id": "test-client"}
    backoff.assert_not_called()


@pytest.mark.parametrize("method,body", [
    ("GET", {}), ("head", {}), ("OPTIONS", {}),
    ("GET", {"json_body": {}}), ("GET", {"data": b""}),
    ("GET", {"files": {}}), ("GET", {"form": {}}),
    ("POST", {}), ("PUT", {}), ("DELETE", {}),
])
@pytest.mark.parametrize("failure", [TimeoutError, 503])
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
