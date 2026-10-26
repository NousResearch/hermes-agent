"""Real CLI-subprocess integration evidence for managed delegation/MoA
authority inheritance: a genuinely spawned Hermes process — going through
``hermes_cli.main`` argv parsing and its real first inference/tool loop —
that delegates a nested task (or runs a managed MoA aggregator) must never
escape the managed routing authority to a configured-but-unapproved
"forbidden" target, while ordinary unmanaged delegation retains its
configured route.

The worker-side kanban routing carrier (HERMES_KANBAN_ROUTING_RECEIPT env
fence at the single-query bootstrap) has been removed pending a redesign
around the gateway kanban.run contract; the delegation/MoA routing
enforcement exercised here is independent of it.

Isolated: everything under tmp_path; the fake servers bind 127.0.0.1:0
(ephemeral loopback port only); no external network, no paid inference.
"""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import hermes_yaml as yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class _CapturingHandler(http.server.BaseHTTPRequestHandler):
    """Records every chat-completions request body it receives; always
    answers a minimal valid response so the turn completes. Non-chat
    GET/POST paths get a harmless canned reply."""

    requests: list  # class-level, set per-server by the fixture

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        try:
            parsed = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            parsed = {"_raw": body.decode("utf-8", "replace")}
        if not self.path.rstrip("/").endswith("chat/completions"):
            resp = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        type(self).requests.append(parsed)
        model = parsed.get("model", "fake-model")
        stream = bool(parsed.get("stream"))
        if stream:
            chunk1 = json.dumps({
                "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ack"}, "finish_reason": None}],
            })
            chunk2 = json.dumps({
                "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
            body_out = (f"data: {chunk1}\n\n" f"data: {chunk2}\n\n" "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body_out)))
            self.end_headers()
            self.wfile.write(body_out)
            return
        resp = json.dumps({
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 0,
            "model": model,
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ack"},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        # Model-list/show discovery probes (/v1/models, /api/v1/models, ...) —
        # harmless canned reply.
        resp = json.dumps({"data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


def _start_fake_server():
    """A fresh handler subclass per server so ``requests`` never leaks
    between tests / profiles."""
    handler_cls = type("Handler", (_CapturingHandler,), {"requests": []})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler_cls


def _write_profile_home(home: Path, base_url: str, model: str, provider_name: str):
    """A minimal real Hermes profile root: config.yaml with a custom_providers
    entry pointed at the fake server, plus the on-disk shape resolve_profile_env
    expects for a named profile (parent dir literally named ``profiles``)."""
    home.mkdir(parents=True, exist_ok=True)
    config = {
        "model": {"default": model, "provider": provider_name},
        "security": {"tirith_enabled": False},
        "custom_providers": [{
            "name": provider_name,
            "base_url": base_url,
            "api_key": "fake-test-key-not-real",
            "api_mode": "chat_completions",
            "models": [model],
        }],
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _publish_route_policy(hermes_home: Path, *, endpoint: str, reasoning: str = "medium"):
    """Publish + activate the kanban-default policy with the fake route, so
    managed delegation/MoA resolution inside the child process can actually
    select an approved route (no receipt is persisted here — the child
    resolves its own decision at launch/bootstrap time)."""
    from agent.model_selection_store import activate_policy, publish_policy

    policy = {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": "fake-route", "route_revision": 1, "provider": "custom-fake",
            "model": "fake-model", "endpoint": endpoint, "maker": "test",
            "model_family": "fake-model", "status": "approved", "allowed_roles": ["builder", "wider"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {
            "builder": {"deep": ["fake-route"], "shallow": ["fake-route"]},
            "wider": {"deep": ["fake-route"], "shallow": ["fake-route"]},
        },
    }
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])


def _worker_python() -> str:
    """The real interpreter this worktree's tests actually run under (mirrors
    ``hermes_cli.kanban_db_dispatch._module_hermes_argv``'s ``sys.executable``
    fallback for shim-less launch environments) — never the bare system
    ``python3``, which may be an unrelated interpreter without this repo's
    deps installed (see ``.hermes/pinned_python.py``)."""
    venv_python = REPO_ROOT.parent.parent / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_worker(*, profile_home: Path, provider: str, model: str,
                query: str, reasoning: str = "high", timeout: float = 60.0,
                bootstrap=None):
    """Launches the REAL CLI entry point in module form (no ``hermes``
    console script is guaranteed on PATH inside a test sandbox).
    ``HERMES_GATEWAY_LOCK_DIR`` is preserved through the scrub: the child
    starts its own gateway for the isolated profile home, and the host
    rendezvous (the developer's live one-gateway-per-host) must stay out of
    its discovery, exactly like conftest isolates the pytest process."""
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("HERMES_") and k not in ("HERMES_HOME", "HERMES_GATEWAY_LOCK_DIR"):
            env.pop(k, None)
    env["HERMES_HOME"] = str(profile_home)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["HERMES_SINGLE_QUERY_SESSION"] = "1"
    argv = [
        _worker_python(), "-m", "hermes_cli.main",
        "--provider", provider, "-m", model, "--reasoning", reasoning,
    ]
    argv.extend(["chat", "-q", query, "-Q"])
    if bootstrap is not None:
        argv[1:3] = [str(bootstrap)]
    proc = subprocess.run(
        argv, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    return proc


from typing import Callable


class _DelegatingHandler(_CapturingHandler):
    task_overrides: dict
    before_delegate: Callable | None = None

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        type(self).requests.append(req)
        messages = req.get("messages", [])
        child = any(m.get("role") == "user" and m.get("content") == "NESTED_TASK" for m in messages)
        finished = any(m.get("role") == "tool" for m in messages)
        message = {"role": "assistant", "content": "done"}
        reason = "stop"
        if not child and not finished:
            if self.before_delegate is not None and any(
                tool.get("function", {}).get("name") == "delegate_task" for tool in req.get("tools", [])
            ):
                self.before_delegate()
                type(self).before_delegate = None
            task = {"goal": "NESTED_TASK", "routing_requirements": {
                "input_tokens": 1000, "reserve_tokens": 8192,
            }, **self.task_overrides}
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "delegate-1", "type": "function", "function": {
                    "name": "delegate_task", "arguments": json.dumps({"tasks": [task]}),
                },
            }]}
            reason = "tool_calls"
        if req.get("stream"):
            payload = {"id": "nested", "model": req.get("model"), "choices": [{
                "index": 0, "delta": message, "finish_reason": reason,
            }]}
            for call in message.get("tool_calls", []):
                call["index"] = 0
            body = (f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n").encode()
            content_type = "text/event-stream"
        else:
            body = json.dumps({"id": "nested", "model": req.get("model"), "choices": [{
                "index": 0, "message": message, "finish_reason": reason,
            }]}).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.parametrize("surface", ["moa", "unmanaged", "moa-lifecycle"])
@pytest.mark.parametrize("overrides", [{}, {"routing_role": "wider"}, {"routing_policy_id": "other"},
                                      {"model": "forbidden-model"}, {"refresh_policy": True}])
def test_real_worker_tool_round_inherits_managed_authority(tmp_path, surface, overrides):
    """A real CLI turn that delegates (or runs a managed MoA aggregator)
    must inherit managed routing authority: a nested task carrying
    routing fields may only launch on a policy-approved route — never the
    profile's configured "forbidden" delegation target — while ordinary
    unmanaged delegation keeps its configured route."""
    from agent.model_selection_store import activate_policy, get_active_policy, publish_policy

    handler = type("DelegatingHandler", (_DelegatingHandler,), {
        "requests": [], "task_overrides": {} if "refresh_policy" in overrides else overrides})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    forbidden, forbidden_handler = _start_fake_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        forbidden_url = f"http://127.0.0.1:{forbidden.server_port}/v1"
        home = tmp_path / "home"
        _write_profile_home(home, url, "fake-model", "custom-fake")
        config = yaml.safe_load((home / "config.yaml").read_text())
        # The nested fixture explicitly supports medium on the wire, including
        # lifecycle children whose default routing requirement is medium.
        config["custom_providers"][0]["extra_body"] = {"reasoning": {"effort": "medium"}}
        config.update({"toolsets": ["delegation"], "delegation": {
            "provider": "custom-forbidden", "model": "forbidden-model", "max_iterations": 2,
        }, "agent": {"max_iterations": 3}})
        config["custom_providers"].append({
            "name": "custom-forbidden", "base_url": forbidden_url, "api_key": "test-only",
            "api_mode": "chat_completions", "models": ["forbidden-model"],
        })
        config["moa"] = {"default_preset": "nested", "presets": {"nested": {
            "enabled": False, "reference_models": [], "aggregator": {
                "provider": "custom-fake", "model": "fake-model", "routing_role": "builder",
                "reasoning_effort": "medium", "routing_requirements": {
                    "input_tokens": 1000, "reserve_tokens": 8192,
                },
            },
        }}}
        (home / "config.yaml").write_text(yaml.safe_dump(config))
        _publish_route_policy(home, endpoint=url, reasoning="medium")
        if "refresh_policy" in overrides and surface.startswith("moa"):
            def update_policy():
                policy = get_active_policy(home, "kanban-default")
                assert policy is not None
                policy["revision"] += 1
                policy["routes"][0].update(provider="custom-forbidden", model="forbidden-model", endpoint=forbidden_url)
                publish_policy(home, policy, approval_ref="local-test-update")
                activate_policy(home, "kanban-default", policy["revision"])
            handler.before_delegate = staticmethod(update_policy)
        bootstrap = None
        if surface.endswith("-lifecycle"):
            bootstrap = tmp_path / "lifecycle_worker.py"
            bootstrap.write_text('''import json
import runpy
import tools.delegate_tool
from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService, get_active_subagent_parent

def launch(**kwargs):
    service = SubagentLifecycleService(get_active_subagent_parent)
    task = kwargs["tasks"][0]
    handle = service.launch(SubagentLaunchRequest(
        goal="NESTED_TASK", model=task.get("model"), routing_role=task.get("routing_role"),
        routing_policy_id=task.get("routing_policy_id"), routing_requirements=task.get("routing_requirements")))
    service.wait(handle, timeout_seconds=20)
    return json.dumps({"summary": service.result(handle).summary})

tools.delegate_tool.delegate_task = launch
runpy.run_module("hermes_cli.main", run_name="__main__")
''')
        is_moa = surface.startswith("moa")
        proc = _run_worker(profile_home=home, provider="moa" if is_moa else "custom-fake",
                           model="nested" if is_moa else "fake-model",
                           query="PARENT_TASK", bootstrap=bootstrap, reasoning="medium")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert any(m.get("role") == "tool" for r in handler.requests for m in r.get("messages", [])), proc.stdout + proc.stderr
        if surface == "unmanaged" and "routing_role" not in overrides:
            assert forbidden_handler.requests, "ordinary unmanaged delegation must retain its configured route"
        else:
            assert not forbidden_handler.requests, "managed authority escaped to the configured forbidden target"
        if surface != "unmanaged" and (not overrides or "refresh_policy" in overrides):
            assert any(m.get("content") == "NESTED_TASK" for r in handler.requests for m in r.get("messages", [])), (
                "must construct and run an authorized child, not merely block every launch",
                [m.get("content") for r in handler.requests for m in r.get("messages", []) if m.get("role") == "tool"],
            )
        if bootstrap is not None:
            assert not any(r.get("model") == "forbidden-model" for r in handler.requests), "model-only lifecycle calls must be constrained preferences, not raw overrides"
    finally:
        server.shutdown()
        forbidden.shutdown()
        server.server_close()
        forbidden.server_close()
