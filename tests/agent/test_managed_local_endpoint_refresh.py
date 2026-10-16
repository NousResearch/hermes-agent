"""Managed alias conversations follow the live router lease (#132727)."""

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest

from agent.secret_scope import (
    build_profile_secret_scope,
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.local_runtime.supervisor import state_path
from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import AIAgent


_SERVER = r"""
import json, os, socket, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
log, move, registry = map(Path, sys.argv[1:])
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    def do_GET(self):
        payload = {'data': [{'id': 'local-probe', 'object': 'model', 'context_length': 262144}],
                   'default_generation_settings': {'n_ctx': 262144}, 'context_length': 262144}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with log.open('a') as f:
            f.write(json.dumps({'body': body, 'key': self.headers.get('Authorization')}) + '\n')
        if move.exists():
            registry.write_text(move.read_text())
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            os._exit(0)
        if body.get('stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            chunk = {'id': 'probe', 'object': 'chat.completion.chunk', 'model': 'local-probe',
                     'created': 0, 'choices': [{'index': 0, 'delta': {'content': 'OK'}, 'finish_reason': 'stop'}]}
            self.wfile.write(('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode())
        else:
            payload = {'id': 'probe', 'object': 'chat.completion', 'created': 0, 'model': 'local-probe',
                       'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'OK'},
                                    'finish_reason': 'stop'}],
                       'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(f'http://127.0.0.1:{server.server_port}/v1', flush=True)
server.serve_forever()
"""


@contextmanager
def _router(tmp_path, label):
    log = tmp_path / f"{label}-requests.jsonl"
    move = tmp_path / f"{label}-move.json"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _SERVER, str(log), str(move), str(state_path())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        url = proc.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:")
        child, owner = psutil.Process(proc.pid), psutil.Process()
        state = {
            "pid": proc.pid,
            "create_time": child.create_time(),
            "executable": child.exe(),
            "owner_pid": owner.pid,
            "owner_create_time": owner.create_time(),
            "base_url": url,
            "api_key": f"local-{label}",
        }
        yield proc, state, log, move
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


@contextmanager
def _scope(home):
    home_token = set_hermes_home_override(home)
    mode_token = set_multiplex_context(True)
    secret_token = set_secret_scope(
        build_profile_secret_scope(home), profile_home=str(home)
    )
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_multiplex_context(mode_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "agent: {api_max_retries: 1, auto_recovery_cycles: 0, environment_probe: false}\n"
        "memory: {provider: builtin}\n"
    )
    with _scope(home):
        yield home


def _publish(state):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def _agent(state, requested="llamacpp"):
    runtime = (
        resolve_runtime_provider(requested=requested, target_model="local-probe")
        if requested == "llamacpp"
        else {
            "provider": "custom",
            "requested_provider": requested,
            "base_url": state["base_url"],
            "api_key": state["api_key"],
            "api_mode": "chat_completions",
        }
    )
    return AIAgent(
        **{
            key: runtime[key]
            for key in (
                "provider",
                "requested_provider",
                "base_url",
                "api_key",
                "api_mode",
            )
        },
        model="local-probe",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        max_iterations=2,
    )


@pytest.mark.parametrize("phase", ["next_turn", "in_flight"])
def test_managed_conversation_follows_a_restarted_router(tmp_path, profile_home, phase):
    with (
        _router(tmp_path, "a") as (first, state_a, log_a, move_a),
        _router(tmp_path, "b") as (_, state_b, log_b, _),
    ):
        _publish(state_a)
        agent = _agent(state_a)
        try:
            warm = agent.run_conversation("first turn")
            assert warm["completed"] and warm["final_response"] == "OK"
            prompt = agent._cached_system_prompt
            second_home = tmp_path / "second-profile"
            second_home.mkdir()
            (second_home / "config.yaml").write_text(
                "agent: {environment_probe: false}\n"
            )
            with _scope(second_home):
                other = _agent(state_b, requested="custom")
                try:
                    assert other.run_conversation("other profile")["completed"]
                finally:
                    other.close()
            before = log_a.read_text().splitlines()
            if phase == "next_turn":
                first.terminate()
                first.wait(timeout=5)
                _publish(state_b)
            else:
                move_a.write_text(json.dumps(state_b))
            result = agent.run_conversation(
                "second turn", conversation_history=warm["messages"]
            )
            assert result["completed"] and result["final_response"] == "OK"
            assert agent.base_url == state_b["base_url"]
            assert agent._primary_runtime["base_url"] == state_b["base_url"]
            assert agent._client_kwargs["base_url"] == state_b["base_url"]
            assert agent._cached_system_prompt == prompt
            assert agent.context_compressor.base_url == state_b["base_url"]
            assert agent.context_compressor.api_key == state_b["api_key"]
            assert len(log_a.read_text().splitlines()) == len(before) + (
                phase == "in_flight"
            )
            requests = [json.loads(line) for line in log_b.read_text().splitlines()]
            assert requests[-1]["key"] == "Bearer local-b"
            assert requests[-1]["body"]["messages"][-1]["content"] == "second turn"
        finally:
            agent.close()


@pytest.mark.parametrize("requested", ["custom", "llama.cpp"])
def test_explicit_local_conversations_keep_their_endpoint(
    tmp_path, profile_home, requested
):
    with (
        _router(tmp_path, "managed") as (_, managed, _, _),
        _router(tmp_path, "explicit") as (_, explicit, log, _),
    ):
        _publish(managed)
        agent = _agent(explicit, requested)
        try:
            result = agent.run_conversation("probe")
            assert result["completed"] and result["final_response"] == "OK"
            assert agent.base_url == explicit["base_url"]
            assert (
                json.loads(log.read_text().splitlines()[-1])["key"]
                == "Bearer local-explicit"
            )
        finally:
            agent.close()
