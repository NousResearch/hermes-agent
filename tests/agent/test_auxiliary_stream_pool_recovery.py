"""Streaming auxiliary calls recover in the provider pool before delivery (#132284)."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openai import APIStatusError

import pytest

from agent import auxiliary_client as aux
from agent.credential_pool import load_pool
from agent.secret_scope import (
    build_profile_secret_scope,
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def bind_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    multiplex_token = set_multiplex_context(True)
    tokens = []

    def bind(home):
        home_token = set_hermes_home_override(home)
        secret_token = set_secret_scope(
            build_profile_secret_scope(home), profile_home=str(home)
        )
        tokens.append((home_token, secret_token))

    try:
        yield bind
    finally:
        for home_token, secret_token in reversed(tokens):
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
        reset_multiplex_context(multiplex_token)


@contextmanager
def _provider(status, reject_all=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = self.headers.get("Authorization", "")
            requests.append((key, body, self.headers.get("X-Probe")))
            if reject_all or key.endswith("-a"):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({
                        "error": {
                            "message": "quota exhausted",
                            "type": "rate_limit_error",
                            "code": "insufficient_quota",
                        }
                    }).encode()
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "probe",
                "object": "chat.completion.chunk",
                "model": "probe-model",
                "created": 0,
                "choices": [
                    {"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}
                ],
            }
            self.wfile.write(
                ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def _home(tmp_path, monkeypatch, bind_profile, endpoint, label):
    home = tmp_path / label
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "credential_pool_strategies: {openrouter: fill_first}\n"
    )
    if not (home / "auth.json").exists():
        (home / "auth.json").write_text(
            json.dumps({
                "version": 1,
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": label + suffix,
                            "label": suffix,
                            "priority": index,
                            "auth_type": "api_key",
                            "source": "manual",
                            "access_token": f"sk-probe-{label}-{suffix}",
                            "base_url": endpoint,
                        }
                        for index, suffix in enumerate(("a", "b"))
                    ]
                },
            })
        )
    bind_profile(home)
    aux._evict_cached_clients("openrouter")
    return home


def _request(endpoint, key):
    return aux.call_llm(
        task="moa_aggregator",
        provider="openrouter",
        model="probe-model",
        base_url=endpoint,
        api_key=key,
        messages=[{"role": "user", "content": "probe"}],
        stream=True,
        stream_options={"include_usage": True},
        extra_headers={"X-Probe": "preserved"},
        main_runtime={},
        timeout=3,
    )


@pytest.mark.parametrize("status", [402, 429])
def test_stream_rotates_the_failed_provider_credential_in_its_profile(
    tmp_path, monkeypatch, bind_profile, status
):
    """Use the real SDK wire, auth store, resolver and pool across A -> B -> A."""
    with _provider(status) as (endpoint, requests):
        first = _home(tmp_path, monkeypatch, bind_profile, endpoint, "first")
        entry = load_pool("openrouter").select()
        assert entry is not None
        key = entry.runtime_api_key
        chunks = list(_request(endpoint, key))
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "OK"
        assert [request[0] for request in requests] == [
            "Bearer sk-probe-first-a",
            "Bearer sk-probe-first-b",
        ]
        assert all(
            body["messages"] == [{"role": "user", "content": "probe"}]
            and body["stream_options"] == {"include_usage": True}
            and header == "preserved"
            for _, body, header in requests
        )
        statuses = {
            entry.id: entry.last_status for entry in load_pool("openrouter").entries()
        }
        assert statuses["firsta"] == "exhausted"
        second = _home(tmp_path, monkeypatch, bind_profile, endpoint, "second")
        assert all(
            entry.last_status != "exhausted"
            for entry in load_pool("openrouter").entries()
        )
        assert first != second
        _home(tmp_path, monkeypatch, bind_profile, endpoint, "first")
        entry = load_pool("openrouter").select()
        assert entry is not None
        assert entry.runtime_api_key == "sk-probe-first-b"


@pytest.mark.parametrize("standalone", [False, True])
def test_stream_preserves_terminal_errors_without_consuming_unrelated_credentials(
    tmp_path, monkeypatch, bind_profile, standalone
):
    with _provider(429, reject_all=True) as (endpoint, requests):
        _home(tmp_path, monkeypatch, bind_profile, endpoint, "first")
        entry = load_pool("openrouter").select()
        assert entry is not None
        key = entry.runtime_api_key
        if standalone:
            key = "sk-standalone"
        with pytest.raises(APIStatusError) as caught:
            list(_request(endpoint, key))
        assert getattr(caught.value, "status_code", None) == 429
        if standalone:
            assert [request[0] for request in requests] == ["Bearer sk-standalone"]
            assert all(
                entry.last_status != "exhausted"
                for entry in load_pool("openrouter").entries()
            )
        else:
            assert [request[0] for request in requests] == [
                "Bearer sk-probe-first-a",
                "Bearer sk-probe-first-b",
            ]
            assert not load_pool("openrouter").has_available()
