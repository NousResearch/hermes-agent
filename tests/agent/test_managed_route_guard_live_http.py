"""Live local-HTTP integration tests for the per-request managed-route guard
(design §12 "best-effort revocation generation check before each subsequent
managed request"). Drives a REAL ``AIAgent.run_conversation`` tool loop
in-process against real ``http.server`` fake endpoints -- not the guard
helper called directly, not a fake-shaped object. Complements:

  * ``tests/agent/test_managed_route_guard_per_request.py`` (unit: the
    guard's own decision logic with an in-memory ``agent`` stand-in).
  * ``tests/hermes_cli/test_kanban_worker_cli_route_enforcement_integration.py``
    (real subprocess CLI boot + the turn's FIRST inference call only).

This file is the missing piece both of those don't cover: a managed agent
whose turn spans MULTIPLE requests (a tool-call loop), where something
changes BETWEEN requests within the same turn --

  1. the active policy is revoked between request 1 and request 2 -> request
     2's content must never reach the (still-live, still-listening) fake
     endpoint;
  2. request 2 is a same-route TRANSIENT retry (e.g. one HTTP 500) -> stays
     pinned to the approved endpoint, no guard interference;
  3. a mid-turn fallback/outage attempt tries to swap the agent's live
     client onto a second, unapproved fake endpoint -> that second endpoint
     receives ZERO requests, ever.

External paid inference is never used; both endpoints are ephemeral
127.0.0.1 loopback ``http.server`` instances.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _CapturingHandler(BaseHTTPRequestHandler):
    """Subclassed per-server (own ``requests``/``response_queue``) so two
    servers in the same test never share state."""

    requests: list
    response_queue: list

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode()) if length else {}
        # Non-chat probes (e.g. an embeddings/model-capability POST issued
        # before the real inference call) get a harmless canned reply and
        # are NOT counted toward the request list -- only an actual
        # chat/completions call is "task content reaching the endpoint".
        if not self.path.rstrip("/").endswith("chat/completions"):
            self._send_json({"ok": True})
            return
        type(self).requests.append(req)
        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        if req.get("stream"):
            self._send_stream(resp)
        else:
            self._send_json(resp)

    def _send_stream(self, resp: dict):
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
        finish_reason = resp["choices"][0].get("finish_reason", "stop")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{
            "id": "m", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }]
        if content:
            chunks.append({
                "id": "m", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            })
        if tool_calls:
            for ti, tc in enumerate(tool_calls):
                chunks.append({
                    "id": "m", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
                    "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": ti, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                    }]}, "finish_reason": None}],
                })
        chunks.append({
            "id": "m", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        })
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        # Model-list/show discovery probes some transports issue before the
        # real inference call -- harmless canned reply, never counted.
        self._send_json({"data": [{"id": "test-model"}]})

    def _send_json(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


def _tc_resp(name: str, args: str = "{}") -> dict:
    return {
        "id": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "",
                        "tool_calls": [{"id": "call_1", "type": "function",
                                        "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def _start_server():
    handler_cls = type("Handler", (_CapturingHandler,), {"requests": [], "response_queue": []})
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler_cls


def _persist_receipt(hermes_home, *, provider: str, model: str, endpoint: str,
                      execution_id: str = "t_live_guard"):
    from agent.model_selection import select
    from agent.model_selection_store import activate_policy, persist_receipt, publish_policy

    policy = {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": "fake-route", "route_revision": 1, "provider": provider,
            "model": model, "endpoint": endpoint, "maker": "test",
            "model_family": model, "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {"builder": {"deep": ["fake-route"], "shallow": ["fake-route"]}},
    }
    requirements = {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": execution_id, "attempt_id": "1", "slot_id": "",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 0, "reserve_tokens": 0, "reasoning": "high",
        "provenance": {"frozen_sha": "deadbeef", "verified_by": "test",
                       "complete": True, "contributors": []},
    }
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])
    decision = select(requirements, policy, {}, now=1000)
    return persist_receipt(hermes_home, decision), policy, record["revision"]


@pytest.fixture()
def managed_agent_env():
    """A real ``AIAgent`` wired as a managed run: two fake servers (approved +
    unapproved), a real on-disk receipt store, and the agent's
    ``_managed_routing_*`` attributes set exactly as cli.py's
    ``_enforce_kanban_routing_receipt`` sets them after the turn's first
    successful validation."""
    approved_server, approved_handler = _start_server()
    unapproved_server, unapproved_handler = _start_server()

    test_home = tempfile.mkdtemp(prefix="hermes_managed_route_guard_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    os.makedirs(os.path.join(hermes_home, "logs"), exist_ok=True)
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = hermes_home

    saved_modules = dict(sys.modules)
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    from run_agent import AIAgent

    approved_port = approved_server.server_address[1]
    approved_url = f"http://127.0.0.1:{approved_port}/v1"
    receipt_id, policy, revision = _persist_receipt(
        hermes_home, provider="openai-compat", model="test-model", endpoint=approved_url,
    )

    agent = AIAgent(
        api_key="test-key", base_url=approved_url,
        provider="openai-compat", model="test-model",
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
        reasoning_config={"enabled": True, "effort": "high"},
    )
    agent.valid_tool_names = {"terminal", "read_file", "write_file", "execute_code", "session_search"}
    # Exactly what cli.py's _enforce_kanban_routing_receipt sets on success,
    # before the turn's first request -- this test starts from "already
    # validated once" and focuses on requests 2+ in the SAME turn.
    agent._managed_routing_receipt_id = receipt_id
    agent._managed_routing_home = hermes_home
    agent.requested_provider = "openai-compat"

    try:
        yield {
            "agent": agent, "approved": approved_handler, "unapproved": unapproved_handler,
            "hermes_home": hermes_home, "receipt_id": receipt_id,
            "policy": policy, "revision": revision,
            "unapproved_url": f"http://127.0.0.1:{unapproved_server.server_address[1]}/v1",
        }
    finally:
        approved_server.shutdown()
        unapproved_server.shutdown()
        sys.modules.clear()
        sys.modules.update(saved_modules)
        for _name, _mod in saved_modules.items():
            _parent, _, _child = _name.rpartition(".")
            if _parent and _parent in saved_modules:
                try:
                    setattr(saved_modules[_parent], _child, _mod)
                except Exception:
                    pass
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home
        shutil.rmtree(test_home, ignore_errors=True)


def test_matched_route_multi_iteration_turn_reaches_only_approved_endpoint(managed_agent_env):
    """Sanity baseline: an unrevoked, unmodified managed turn with two
    tool-call iterations sends both requests to the approved endpoint and
    NEVER touches the unapproved one."""
    env = managed_agent_env
    agent, approved, unapproved = env["agent"], env["approved"], env["unapproved"]
    approved.response_queue.append(_tc_resp("terminal", '{"command": "echo hi"}'))
    approved.response_queue.append(_text_resp("done"))

    agent.run_conversation("run echo hi", conversation_history=[], task_id="t")

    assert len(approved.requests) == 2
    assert unapproved.requests == []


def test_revocation_between_requests_blocks_the_next_request_content(managed_agent_env):
    """Policy revoked (activated a new revision) after request 1's response
    lands, before request 2 is built -- request 2's task content must NEVER
    reach the still-listening approved endpoint."""
    from agent.model_selection_store import activate_policy, publish_policy

    env = managed_agent_env
    agent, approved, unapproved = env["agent"], env["approved"], env["unapproved"]

    approved.response_queue.append(_tc_resp("terminal", '{"command": "echo SECRET_PAYLOAD_2"}'))
    approved.response_queue.append(_text_resp("should never be produced"))

    # Revoke between request 1 (already answered above) and request 2: hook
    # the fake server so the ACT of it answering request 1 is the trigger for
    # revocation, guaranteeing "between requests" ordering deterministically
    # rather than racing a background thread.
    original_do_POST = approved.do_POST
    revoked = {"done": False}

    def _do_post_then_revoke(self):
        original_do_POST(self)
        if len(type(self).requests) == 1 and not revoked["done"]:
            revoked["done"] = True
            new_policy = dict(env["policy"])
            new_policy["revision"] = env["revision"] + 1
            record = publish_policy(env["hermes_home"], new_policy, approval_ref="operator:revoke")
            activate_policy(env["hermes_home"], "kanban-default", record["revision"])

    approved.do_POST = _do_post_then_revoke

    agent.run_conversation("run echo SECRET_PAYLOAD_2", conversation_history=[], task_id="t")

    assert len(approved.requests) == 1, (
        "revocation between requests must stop the turn before request 2 is sent -- "
        f"got {len(approved.requests)} requests"
    )
    assert "SECRET_PAYLOAD_2" not in json.dumps(_tool_results(approved)), (
        "no leaked task content in a tool-result round-trip after revocation"
    )
    assert unapproved.requests == []


def test_same_route_transient_retry_stays_pinned_to_approved_endpoint(managed_agent_env):
    """A same-route retry (server hiccup, e.g. one throwaway extra tool-call
    round trip) must NOT be treated as a route change -- the guard must keep
    passing every iteration as long as nothing about the route diverged."""
    env = managed_agent_env
    agent, approved, unapproved = env["agent"], env["approved"], env["unapproved"]

    # Three iterations, same route throughout -- simulates transient retries
    # within the tool loop rather than a single request/response.
    approved.response_queue.append(_tc_resp("terminal", '{"command": "echo one"}'))
    approved.response_queue.append(_tc_resp("terminal", '{"command": "echo two"}'))
    approved.response_queue.append(_text_resp("done"))

    agent.run_conversation("run echo one then two", conversation_history=[], task_id="t")

    assert len(approved.requests) == 3
    assert unapproved.requests == []


def test_mid_turn_provider_swap_to_unapproved_endpoint_sends_zero_requests(managed_agent_env):
    """Simulates a fallback/outage code path swapping the agent's live client
    onto a second, unapproved endpoint BETWEEN iterations of the same
    managed turn (e.g. an activated fallback chain entry never covered by
    the receipt). The per-request guard must catch the divergence before
    request 2 is built, and the unapproved endpoint must receive ZERO
    requests -- not even a connection attempt."""
    env = managed_agent_env
    agent, approved, unapproved = env["agent"], env["approved"], env["unapproved"]

    approved.response_queue.append(_tc_resp("terminal", '{"command": "echo before swap"}'))
    # (never reached) -- would only be consumed if the guard failed to block.
    unapproved.response_queue.append(_text_resp("LEAKED: should never be sent"))

    original_do_POST = approved.do_POST
    swapped = {"done": False}

    def _do_post_then_swap(self):
        original_do_POST(self)
        if len(type(self).requests) == 1 and not swapped["done"]:
            swapped["done"] = True
            # Simulate an outage/fallback client rebuild: the live agent now
            # points at the unapproved fake endpoint for the NEXT request.
            agent.base_url = env["unapproved_url"]
            agent.client.base_url = env["unapproved_url"]

    approved.do_POST = _do_post_then_swap

    agent.run_conversation("run echo before swap", conversation_history=[], task_id="t")

    assert unapproved.requests == [], (
        "an unapproved fallback endpoint reached mid-turn must receive ZERO "
        f"requests -- got {unapproved.requests!r}"
    )
    assert len(approved.requests) == 1, "only the pre-swap request should have gone anywhere"


def _tool_results(handler) -> list[str]:
    out = []
    for req in handler.requests:
        for m in req.get("messages", []):
            if m.get("role") == "tool":
                out.append(m.get("content", ""))
    return out
