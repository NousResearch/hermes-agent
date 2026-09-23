"""Real Kanban worker launch integration evidence (parent-flagged acceptance
gap): the guard tests in ``test_kanban_worker_route_enforcement.py`` call
``enforce_worker_route`` directly with hand-supplied ``actual_*`` fields.
That proves the guard function's own logic, but NOT that a genuinely
spawned Kanban worker process — going through ``hermes_cli.main`` argv
parsing, ``cli.py``'s single-query bootstrap, and its real first
inference call — actually reaches (or is actually blocked by)
``_enforce_kanban_routing_receipt`` at the location cli.py wires it in.

This test launches a REAL ``python -m hermes_cli.main -p <profile> --provider
custom-fake --model fake-model chat -q ... -Q`` subprocess (the exact argv
shape ``hermes_cli/kanban_db_dispatch.py::_worker_argv`` builds), pointed at
a local fake OpenAI-compatible HTTP server via a ``custom_providers`` config
entry, with ``HERMES_KANBAN_ROUTING_RECEIPT`` set in the child's env exactly
as ``_default_spawn`` sets it. No guard helper is called directly; only the
receipt is pre-persisted (equivalent of the dispatcher's claim-time
``resolve_task_route``) and the child process's own inference call (or
non-call) is observed.

Cases:
  * matched route -> worker's first inference call actually reaches the
    fake endpoint (non-zero requests recorded, request content is exactly
    the expected task text — no other content leaked).
  * mismatched/missing receipt -> worker exits non-zero BEFORE any request
    reaches the fake endpoint (zero requests recorded) -- proves fail-closed
    at the real subprocess boundary, not just in the guard's return value.
  * profile A / profile B scope (A-B-A): each profile's own
    model_routing.db receipt is found by that profile's own worker; a
    worker launched under the wrong profile's HERMES_HOME never
    reads or matches the other profile's receipt.

Isolated: everything under tmp_path; the fake server binds 127.0.0.1:0
(ephemeral loopback port only); no external network, no paid inference.
"""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class _CapturingHandler(http.server.BaseHTTPRequestHandler):
    """Records every chat-completions request body it receives (the actual
    first-inference call); always answers a minimal valid response so the
    worker's turn completes. Non-chat GET/POST paths (model-list/model-show
    probes some transports issue before the real inference call) get a
    harmless canned reply and are NOT counted as "requests" for the
    fail-closed assertions — only a real chat/completions call counts as
    task content actually reaching the endpoint."""

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
        # harmless canned reply; never counted toward the fail-closed assertions.
        resp = json.dumps({"data": [{"id": "fake-model"}, {"id": "fake-model-a"}, {"id": "fake-model-b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


def _start_fake_server():
    """A fresh handler subclass per server so ``requests`` never leaks
    between tests / profiles (A-B-A isolation of what the FIXTURE
    records, independent of the routing isolation under test)."""
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


def _persist_receipt(hermes_home: Path, *, provider: str, model: str, endpoint: str,
                      reasoning: str = "high", execution_id: str = "t_worker_cli_it", capacity: int = 200000):
    from agent.model_selection import select
    from agent.model_selection_store import activate_policy, persist_receipt, publish_policy

    policy = {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": "fake-route", "route_revision": 1, "provider": provider,
            "model": model, "endpoint": endpoint, "maker": "test",
            "model_family": model, "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": capacity,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {"builder": {"deep": ["fake-route"], "shallow": ["fake-route"]}},
    }
    requirements = {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": execution_id, "attempt_id": "1", "slot_id": "",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 8192, "reasoning": reasoning,
        "provenance": {"frozen_sha": "deadbeef", "verified_by": "test",
                       "complete": True, "contributors": []},
    }
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    board = hermes_home / "worker-test.db"
    kb.init_db(board)
    with connect(board) as conn:
        tid = kb.create_task(conn, title=execution_id, assignee="test", routing_role="builder")
        claimed = kb.claim_task(conn, tid, claimer="fixture-dispatcher")
        assert claimed is not None and claimed.current_run_id is not None
        requirements.update(execution_id=tid, attempt_id=str(claimed.current_run_id))
        decision = select(requirements, policy, {}, now=1000)
        receipt = persist_receipt(hermes_home, decision)
        assert kb.set_routing_receipt(conn, tid, receipt, expected_run_id=claimed.current_run_id)
    return receipt


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


def _run_worker(*, profile_home: Path, provider: str, model: str, receipt_id: str,
                 query: str, reasoning: str = "high", timeout: float = 60.0,
                 base_url: str = None, bootstrap=None, canonical_receipt_id=None):
    """Launches the REAL CLI entry point exactly as
    ``hermes_cli.kanban_db_dispatch._worker_argv``/``_resolve_hermes_argv``
    would (module form, since no ``hermes`` console script is guaranteed on
    PATH inside a test sandbox) with the SAME env vars
    ``_default_spawn`` sets for a managed task."""
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("HERMES_") and k not in ("HERMES_HOME",):
            env.pop(k, None)
    env["HERMES_HOME"] = str(profile_home)
    env["HERMES_KANBAN_ROUTING_RECEIPT"] = receipt_id
    if receipt_id:
        from agent.model_selection_store import get_receipt
        receipt = get_receipt(profile_home, canonical_receipt_id or receipt_id)
        env["HERMES_KANBAN_DB"] = str(profile_home / "worker-test.db")
        env["HERMES_KANBAN_ROUTING_ORIGIN_HOME"] = str(profile_home)
        if receipt:
            env["HERMES_KANBAN_TASK"] = receipt["requirements"]["execution_id"]
            env["HERMES_KANBAN_RUN_ID"] = receipt["requirements"]["attempt_id"]
    env["HERMES_SESSION_SOURCE"] = "kanban"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["HERMES_SINGLE_QUERY_SESSION"] = "1"
    argv = [
        _worker_python(), "-m", "hermes_cli.main",
        "--provider", provider, "-m", model, "--reasoning", reasoning,
    ]
    if base_url:
        # Mirrors _default_spawn's ``--base-url`` propagation of the
        # receipted route's endpoint (kanban_db_dispatch.py::_worker_argv).
        argv.extend(["--base-url", base_url])
    argv.extend(["chat", "-q", query, "-Q"])
    if bootstrap is not None:
        argv[1:3] = [str(bootstrap)]
    proc = subprocess.run(
        argv, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    return proc


@pytest.mark.parametrize("substitution", [None, "task", "attempt", "role", "board_link"])
def test_intact_receipt_must_match_live_canonical_claim(tmp_path, fake_server, substitution):
    from agent.model_selection import select
    from agent.model_selection_store import get_active_policy, get_receipt, persist_receipt
    from hermes_cli.kanban_db_connect import connect

    server, handler = fake_server
    url = f"http://127.0.0.1:{server.server_port}/v1"
    home = tmp_path / "home"
    _write_profile_home(home, url, "fake-model", "custom-fake")
    canonical_id = _persist_receipt(home, provider="custom-fake", model="fake-model", endpoint=url)
    canonical = get_receipt(home, canonical_id)
    from agent.model_selection_types import REQUIRED_REQUIREMENT_FIELDS
    requirements = {key: canonical["requirements"][key]
                    for key in REQUIRED_REQUIREMENT_FIELDS if key != "schema_version"}
    requirements["schema_version"] = 1
    supplied_id = canonical_id
    if substitution in ("task", "attempt", "board_link"):
        if substitution in ("task", "board_link"):
            requirements["execution_id"] = "another-task"
        else:
            requirements["attempt_id"] = "another-attempt"
        policy = get_active_policy(home, "kanban-default")
        supplied_id = persist_receipt(home, select(requirements, policy, {}, now=1000))
        assert supplied_id != canonical_id
        assert get_receipt(home, supplied_id)["selected"] == canonical["selected"]
        if substitution == "board_link":
            with connect(home / "worker-test.db") as conn:
                conn.execute("UPDATE tasks SET routing_receipt_id=? WHERE id=?",
                             (supplied_id, canonical["requirements"]["execution_id"]))
                conn.commit()
            supplied_id = canonical_id
    elif substitution == "role":
        with connect(home / "worker-test.db") as conn:
            conn.execute("UPDATE tasks SET routing_role=? WHERE id=?",
                         ("reviewquality", requirements["execution_id"]))
            conn.commit()
    proc = _run_worker(
        profile_home=home, provider="custom-fake", model="fake-model",
        receipt_id=supplied_id, canonical_receipt_id=canonical_id, query="canonical claim probe",
    )
    if substitution is None:
        assert handler.requests, proc.stdout + proc.stderr
        assert "canonical claim probe" in json.dumps(handler.requests)
    else:
        assert handler.requests == [], proc.stdout + proc.stderr
        assert proc.returncode != 0


def test_worker_checks_assembled_input_not_claimed_estimate(tmp_path, fake_server):
    server, handler = fake_server
    url = f"http://127.0.0.1:{server.server_port}/v1"
    home = tmp_path / "home"
    _write_profile_home(home, url, "fake-model", "custom-fake")
    receipt = _persist_receipt(home, provider="custom-fake", model="fake-model", endpoint=url, capacity=10000)
    proc = _run_worker(profile_home=home, provider="custom-fake", model="fake-model",
                       receipt_id=receipt, query="oversized task " * 10000)
    assert not handler.requests, "a small caller estimate cannot authorize larger assembled content"
    assert "input_too_large" in proc.stdout + proc.stderr


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


@pytest.mark.parametrize("surface", ["kanban", "moa", "unmanaged", "kanban-lifecycle", "moa-lifecycle"])
@pytest.mark.parametrize("overrides", [{}, {"routing_role": "wider"}, {"routing_policy_id": "other"},
                                      {"model": "forbidden-model"}, {"refresh_policy": True}])
def test_real_worker_tool_round_inherits_managed_authority(tmp_path, surface, overrides):
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
        receipt = _persist_receipt(home, provider="custom-fake", model="fake-model", endpoint=url, reasoning="medium")
        if "refresh_policy" in overrides and surface.startswith("kanban"):
            policy = get_active_policy(home, "kanban-default")
            assert policy is not None
            policy["revision"] += 1
            policy["routes"][0].update(provider="custom-forbidden", model="forbidden-model", endpoint=forbidden_url)
            publish_policy(home, policy, approval_ref="local-test-update")
            activate_policy(home, "kanban-default", policy["revision"])
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
                           receipt_id=receipt if surface.startswith("kanban") else "", query="PARENT_TASK",
                           bootstrap=bootstrap, reasoning="medium")
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


@pytest.fixture
def fake_server():
    server, handler_cls = _start_fake_server()
    try:
        yield server, handler_cls
    finally:
        server.shutdown()


# ── matched route: the real worker's first inference actually lands ─────────

def test_matched_route_reaches_fake_endpoint_with_only_task_content(tmp_path, fake_server, monkeypatch):
    server, handler_cls = fake_server
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    home = tmp_path / "profileA" / ".hermes"
    _write_profile_home(home, base_url, "fake-model", "custom-fake")
    receipt_id = _persist_receipt(home, provider="custom-fake", model="fake-model", endpoint=base_url)

    # The dispatcher never publishes a PID, modelling death just after spawn.
    # Observe durable ownership at HTTP arrival, not after worker completion.
    import sqlite3
    ownership_at_send = []
    original_post = handler_cls.do_POST

    def observe_post(handler):
        if handler.path.rstrip("/").endswith("chat/completions"):
            with sqlite3.connect(home / "worker-test.db") as conn:
                ownership_at_send.append(conn.execute(
                    "SELECT t.worker_pid, t.worker_started_at, r.worker_pid "
                    "FROM tasks t JOIN task_runs r ON r.id=t.current_run_id "
                    "WHERE t.routing_receipt_id=?", (receipt_id,),
                ).fetchone())
        original_post(handler)

    monkeypatch.setattr(handler_cls, "do_POST", observe_post)
    proc = _run_worker(
        profile_home=home, provider="custom-fake", model="fake-model",
        receipt_id=receipt_id, query="what is 2+2, answer in one word",
    )

    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert len(handler_cls.requests) >= 1, (
        "matched route must actually reach the fake endpoint's first inference call "
        f"(stdout={proc.stdout!r} stderr={proc.stderr!r})"
    )
    sent = handler_cls.requests[0]
    assert sent.get("model") == "fake-model"
    assert ownership_at_send
    assert all(pid and started and pid == run_pid for pid, started, run_pid in ownership_at_send), ownership_at_send
    all_text = json.dumps(sent)
    assert "2+2" in all_text
    # only the one allowed task's content — no stray leaked content from another route
    assert "guided-routing enforcement blocked" not in (proc.stdout + proc.stderr)


# ── mismatched/missing receipt: ZERO content ever reaches the endpoint ──────

def test_missing_receipt_sends_zero_requests_and_fails_closed(tmp_path, fake_server):
    server, handler_cls = fake_server
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    home = tmp_path / "profileA" / ".hermes"
    _write_profile_home(home, base_url, "fake-model", "custom-fake")
    # No receipt persisted at all -> get_receipt() returns None -> RoutingBlocked.
    proc = _run_worker(
        profile_home=home, provider="custom-fake", model="fake-model",
        receipt_id="rr_never_persisted", query="do not send this content",
    )

    assert proc.returncode != 0, "a worker with no valid receipt must exit non-zero, not silently succeed"
    assert handler_cls.requests == [], (
        "a worker whose route enforcement fails must send ZERO requests to the endpoint — "
        f"got {handler_cls.requests!r}"
    )
    assert "do not send this content" not in json.dumps(handler_cls.requests)


def test_actual_model_diverging_from_receipt_sends_zero_requests(tmp_path, fake_server):
    """The receipted decision says fake-model-a; the worker is actually
    launched (argv-level) with a different model -- must block before any
    request is sent, exactly like a code path that silently substituted a
    different route."""
    server, handler_cls = fake_server
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    home = tmp_path / "profileA" / ".hermes"
    _write_profile_home(home, base_url, "fake-model-a", "custom-fake")
    # also register the divergent model so the CLI can construct it at all
    cfg_path = home / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["custom_providers"][0]["models"] = ["fake-model-a", "fake-model-b"]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    receipt_id = _persist_receipt(
        home, provider="custom-fake", model="fake-model-a", endpoint=base_url,
    )
    proc = _run_worker(
        profile_home=home, provider="custom-fake", model="fake-model-b",  # diverges
        receipt_id=receipt_id, query="secret task content",
    )

    assert proc.returncode != 0
    assert handler_cls.requests == [], (
        f"a divergent actual model must block before sending — got {handler_cls.requests!r}"
    )


# ── selected endpoint propagation: --base-url must reach the actual client ──

def test_selected_endpoint_reaches_actual_client_via_base_url_override(tmp_path):
    """The receipted route's endpoint may differ from the worker profile's
    OWN configured default base_url (a real guided-routing scenario: the
    policy picked a route whose endpoint isn't the profile's config.yaml
    default). Propagating ``--base-url`` (mirrors kanban_db_dispatch.py's
    ``_worker_argv`` under ``routing_endpoint`` set from
    ``managed_child_kwargs()["endpoint"]``) must make the ACTUAL constructed
    client hit the route's real endpoint (own fake server, own request
    count), and must make ``enforce_worker_route``'s ``actual_endpoint``
    check compare correctly against that same nondefault endpoint --
    without ``--base-url`` the worker would silently start the OTHER
    server's endpoint (the profile default) instead."""
    default_server, default_handler_cls = _start_fake_server()
    routed_server, routed_handler_cls = _start_fake_server()
    try:
        default_port = default_server.server_address[1]
        routed_port = routed_server.server_address[1]
        default_base_url = f"http://127.0.0.1:{default_port}/v1"
        routed_base_url = f"http://127.0.0.1:{routed_port}/v1"
        assert default_base_url != routed_base_url

        home = tmp_path / "profileA" / ".hermes"
        # Profile's OWN config.yaml default endpoint is the "default" server —
        # NOT the one the receipted route actually selected.
        _write_profile_home(home, default_base_url, "fake-model", "custom-fake")
        receipt_id = _persist_receipt(
            home, provider="custom-fake", model="fake-model", endpoint=routed_base_url,
        )

        proc = _run_worker(
            profile_home=home, provider="custom-fake", model="fake-model",
            receipt_id=receipt_id, query="routed endpoint task",
            base_url=routed_base_url,
        )

        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        assert default_handler_cls.requests == [], (
            "the worker must never fall back to the profile's own default "
            f"endpoint when a route endpoint was selected: got "
            f"{default_handler_cls.requests!r}"
        )
        assert len(routed_handler_cls.requests) >= 1, (
            "the actual constructed client must reach the SELECTED route's "
            f"endpoint (stdout={proc.stdout!r} stderr={proc.stderr!r})"
        )
        assert any(
            "routed endpoint task" in json.dumps(r) for r in routed_handler_cls.requests
        )
    finally:
        default_server.shutdown()
        routed_server.shutdown()


def test_selected_endpoint_not_propagated_blocks_before_any_request(tmp_path):
    """Same setup as above, but WITHOUT ``--base-url`` -- the worker
    constructs its client against its own profile-default endpoint, which
    diverges from the receipted route's endpoint. enforce_worker_route's
    actual_endpoint check must catch this BEFORE any request reaches either
    server (fail-closed), proving the endpoint check has real teeth and is
    not vacuously satisfied by two servers that happen to both work."""
    default_server, default_handler_cls = _start_fake_server()
    routed_server, routed_handler_cls = _start_fake_server()
    try:
        default_port = default_server.server_address[1]
        routed_port = routed_server.server_address[1]
        default_base_url = f"http://127.0.0.1:{default_port}/v1"
        routed_base_url = f"http://127.0.0.1:{routed_port}/v1"

        home = tmp_path / "profileA" / ".hermes"
        _write_profile_home(home, default_base_url, "fake-model", "custom-fake")
        receipt_id = _persist_receipt(
            home, provider="custom-fake", model="fake-model", endpoint=routed_base_url,
        )

        # No base_url passed: worker boots against its own profile default.
        proc = _run_worker(
            profile_home=home, provider="custom-fake", model="fake-model",
            receipt_id=receipt_id, query="must not be sent anywhere",
        )

        assert proc.returncode != 0, (
            "an endpoint mismatch (selected route's endpoint never propagated) "
            "must fail closed, not silently launch against the profile default"
        )
        assert default_handler_cls.requests == []
        assert routed_handler_cls.requests == []
    finally:
        default_server.shutdown()
        routed_server.shutdown()

def test_profile_a_and_b_workers_only_find_their_own_receipt(tmp_path, fake_server):
    server, handler_cls = fake_server
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1"

    home_a = tmp_path / "profileA" / ".hermes"
    home_b = tmp_path / "profileB" / ".hermes"
    _write_profile_home(home_a, base_url, "fake-model", "custom-fake")
    _write_profile_home(home_b, base_url, "fake-model", "custom-fake")

    receipt_a = _persist_receipt(home_a, provider="custom-fake", model="fake-model", endpoint=base_url,
                                  execution_id="t_worker_cli_it_a")
    receipt_b = _persist_receipt(home_b, provider="custom-fake", model="fake-model", endpoint=base_url,
                                  execution_id="t_worker_cli_it_b")
    assert receipt_a != receipt_b

    # A's own worker, A's own receipt -> succeeds (origin-owned receipt found).
    proc_a1 = _run_worker(
        profile_home=home_a, provider="custom-fake", model="fake-model",
        receipt_id=receipt_a, query="A origin task",
    )
    assert proc_a1.returncode == 0, f"stdout={proc_a1.stdout!r} stderr={proc_a1.stderr!r}"
    n_after_a1 = len(handler_cls.requests)
    assert n_after_a1 >= 1
    assert any("A origin task" in json.dumps(r) for r in handler_cls.requests)

    # B's worker handed A's receipt id -> B's model_routing.db has no such row
    # -> must fail closed, never consult / reuse A's DB.
    proc_b_with_a_receipt = _run_worker(
        profile_home=home_b, provider="custom-fake", model="fake-model",
        receipt_id=receipt_a, query="B must not use A receipt",
    )
    assert proc_b_with_a_receipt.returncode != 0
    assert len(handler_cls.requests) == n_after_a1, (
        "profile B worker must not find/consult profile A's receipt store"
    )

    # Back to A again with a fresh, still-A-owned receipt -> succeeds again
    # (A-B-A: A's own scope is unaffected by the intervening B attempt). A worker
    # turn may also fire an auxiliary session-title-generation call, so assert on
    # the task content actually landing rather than an exact request count.
    receipt_a = _persist_receipt(home_a, provider="custom-fake", model="fake-model", endpoint=base_url,
                                 execution_id="t_worker_cli_it_a_again")
    proc_a2 = _run_worker(
        profile_home=home_a, provider="custom-fake", model="fake-model",
        receipt_id=receipt_a, query="A origin task again",
    )
    assert proc_a2.returncode == 0, f"stdout={proc_a2.stdout!r} stderr={proc_a2.stderr!r}"
    assert len(handler_cls.requests) > n_after_a1
    assert any("A origin task again" in json.dumps(r) for r in handler_cls.requests[n_after_a1:])
    n_after_a2 = len(handler_cls.requests)

    # And B's own receipt still resolves fine under B.
    proc_b_own = _run_worker(
        profile_home=home_b, provider="custom-fake", model="fake-model",
        receipt_id=receipt_b, query="B origin task",
    )
    assert proc_b_own.returncode == 0, f"stdout={proc_b_own.stdout!r} stderr={proc_b_own.stderr!r}"
    assert any("B origin task" in json.dumps(r) for r in handler_cls.requests[n_after_a2:])
