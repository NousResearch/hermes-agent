"""The script-facing completion shares Hermes routing without an agent turn."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli.infer_cmd import cmd_infer


def _args(**overrides):
    values = dict(prompt="Classify this", provider=None, model=None, schema=None,
                  json=True, max_tokens=None, timeout=None)
    return argparse.Namespace(**(values | overrides))


def _response(text: str, model: str = "selected-model"):
    return SimpleNamespace(
        model=model, choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14),
    )


@pytest.fixture
def configured_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  default: selected-model\n"
        "  base_url: http://127.0.0.1:9876/v1\n  api_key: test-key\n"
        "  aliases:\n    triage: custom/triage-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_infer_uses_profile_model_and_prints_route_usage(configured_home, monkeypatch, capsys):
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        kwargs["route_info"].update(provider="custom", model="selected-model")
        return _response("answer")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    assert cmd_infer(_args()) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["error"] is None
    assert output["text"] == "answer"
    assert output["provider"] == "custom"
    assert output["model"] == "selected-model"
    assert output["usage"]["total_tokens"] == 14
    assert seen["messages"] == [{"role": "user", "content": "Classify this"}]
    assert "tools" not in seen
    assert not list(configured_home.glob("**/state.db"))


def test_infer_calls_configured_local_endpoint_without_agent_context(
    configured_home, capsys,
):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, json.loads(body)))
            response = json.dumps({
                "id": "fixture", "object": "chat.completion", "model": "selected-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        (configured_home / "config.yaml").write_text(
            "model:\n  provider: custom\n  default: selected-model\n"
            f"  base_url: http://127.0.0.1:{server.server_port}/v1\n"
            "  api_key: test-key\n", encoding="utf-8")
        assert cmd_infer(_args()) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    output = json.loads(capsys.readouterr().out)
    assert output["text"] == "answer"
    assert output["usage"]["total_tokens"] == 4
    assert len(requests) == 1
    assert requests[0][0] == "/v1/chat/completions"
    assert requests[0][1]["messages"] == [{"role": "user", "content": "Classify this"}]
    assert "tools" not in requests[0][1]
    assert not list(configured_home.glob("**/state.db"))


def test_cli_rehydrates_profile_key_for_sanitized_script(configured_home):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.headers.get("Authorization"),
                             json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            response = json.dumps({
                "id": "fixture", "object": "chat.completion", "model": "fixture-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        (configured_home / "config.yaml").write_text(
            "model:\n  provider: custom:fixture\n  default: fixture-model\n"
            "custom_providers:\n  - name: fixture\n"
            f"    base_url: http://127.0.0.1:{server.server_port}/v1\n"
            "    key_env: INFER_TEST_KEY\n", encoding="utf-8")
        (configured_home / ".env").write_text("INFER_TEST_KEY=profile-test-key\n", encoding="utf-8")
        env = os.environ.copy()
        env.pop("INFER_TEST_KEY", None)
        env["HERMES_HOME"] = str(configured_home)
        env["PYTHONIOENCODING"] = "utf-8"
        run = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "infer", "from script", "--json"],
            env=env, capture_output=True, text=True, timeout=90, check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["text"] == "done"
    assert len(requests) == 1
    assert requests[0][0] == "/v1/chat/completions"
    assert requests[0][1] == "Bearer profile-test-key"
    assert requests[0][2]["messages"] == [{"role": "user", "content": "from script"}]


def test_infer_resolves_configured_model_alias(configured_home, monkeypatch, capsys):
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        kwargs["route_info"].update(provider="custom", model="triage-model")
        return _response("ok", "triage-model")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    assert cmd_infer(_args(model="triage", json=False)) == 0
    assert capsys.readouterr().out == "ok\n"
    assert seen["model"] == "triage-model"
    assert seen["provider"] == "custom"


def test_infer_schema_validates_and_returns_parsed_json(configured_home, tmp_path, monkeypatch, capsys):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"type": "object", "properties": {
        "label": {"type": "string"}}, "required": ["label"]}), encoding="utf-8")

    def fake_call_llm(**kwargs):
        assert kwargs["extra_body"]["response_format"]["type"] == "json_schema"
        return _response('{"label":"urgent"}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    assert cmd_infer(_args(schema=str(schema_path))) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["parsed"] == {"label": "urgent"}


def test_infer_refuses_invalid_schema_before_model_call(configured_home, tmp_path, monkeypatch, capsys):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"not-a-type"}', encoding="utf-8")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: pytest.fail("model called"))
    assert cmd_infer(_args(schema=str(schema_path))) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["text"] is None
    assert output["error"]["code"] == "request_failed"


def test_infer_reports_invalid_json_as_failure(configured_home, tmp_path, monkeypatch, capsys):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("not JSON")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    assert cmd_infer(_args(schema=str(schema_path))) == 1
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert captured.err == ""
    assert output["ok"] is False
    assert output["text"] == "not JSON"
    assert output["parsed"] is None
    assert output["error"]["code"] == "invalid_json"
    assert output["usage"]["total_tokens"] == 14
    assert output["model"] == "selected-model"
    assert len(calls) == 1


def test_infer_returns_raw_schema_mismatch_for_local_recovery(configured_home, tmp_path, monkeypatch, capsys):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object","properties":{"label":{"type":"string"}},'
                           '"required":["label"]}', encoding="utf-8")
    raw = '{"label":7,"other":"keep this"}'
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: _response(raw))
    assert cmd_infer(_args(schema=str(schema_path))) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["text"] == raw
    assert output["parsed"] is None
    assert output["error"]["code"] == "schema_mismatch"
    assert "string" in output["error"]["message"]
    assert output["usage"]["total_tokens"] == 14


def test_infer_plain_mode_keeps_invalid_output_off_stdout(configured_home, tmp_path, monkeypatch, capsys):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: _response("not JSON"))
    assert cmd_infer(_args(schema=str(schema_path), json=False)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "did not return valid JSON" in captured.err


def test_infer_returns_error_envelope_on_provider_failure(configured_home, monkeypatch, capsys):
    class ProviderError(Exception):
        pass

    def fail_call(**kwargs):
        raise ProviderError("endpoint unavailable")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fail_call)
    assert cmd_infer(_args()) == 1
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert captured.err == ""
    assert output["ok"] is False
    assert output["text"] is None
    assert output["error"] == {"code": "request_failed", "message": "endpoint unavailable"}


@pytest.mark.parametrize("overrides", [
    {"provider": "openrouter"}, {"max_tokens": 0}, {"timeout": -1}, {"prompt": ""},
])
def test_infer_rejects_invalid_request_without_model_call(configured_home, monkeypatch, capsys, overrides):
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **kw: pytest.fail("model called"))
    assert cmd_infer(_args(**overrides)) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["text"] is None
    assert output["error"]["code"] == "request_failed"
