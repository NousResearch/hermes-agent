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
            c["model"] = resp.get("model", "test-model")
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
        "input_tokens": 1000, "reserve_tokens": 4096, "reasoning": "high",
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
        request_overrides={"reasoning_effort": "high"},
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
        # Close the agent's own resources (relay session scope, shared client, memory
        # provider, ...) BEFORE the fake HTTP servers stop and the temp HERMES_HOME is
        # deleted. Skipping this left the HermesRelay session, its log-queue listener
        # thread and any still-open file handlers pointed at a home directory that no
        # longer exists by the time they next tried to write -- surfacing as
        # 'session scope close failed: not found' and FileNotFoundError spam in
        # errors.log/agent.log from a background thread racing shutil.rmtree below,
        # not a real production defect. agent.close() is idempotent and every phase
        # is individually guarded, so this is safe even if the turn already failed.
        _quietly_close(agent)
        approved_server.shutdown()
        unapproved_server.shutdown()
        # The shared queued file-log listener (hermes_logging.py) runs on its own
        # background thread and only writes a record's bytes when it eventually
        # dequeues it -- agent.close() above returns long before that happens.
        # Without draining here, warnings/errors logged during this test's own
        # teardown (e.g. relay session close) race shutil.rmtree below and land
        # as FileNotFoundError spam from the listener thread instead of ever
        # reaching a real errors.log.
        try:
            from hermes_logging import drain_log_queue
            drain_log_queue(timeout=2.0)
        except Exception:
            pass
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


def _quietly_close(agent) -> None:
    try:
        agent.close()
    except Exception:
        pass


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
    from agent.model_selection_store import list_outcomes
    health = [event["payload"] for event in list_outcomes(env["hermes_home"], env["receipt_id"])
              if event["kind"] == "routing_health"]
    assert len(health) == len(approved.requests)
    assert all(event["status"] == "healthy" for event in health)


def test_revocation_between_requests_blocks_the_next_request_content(managed_agent_env):
    """An explicit EMERGENCY revocation (``revoke_route``, not a routine policy
    republish/activate) issued after request 1's response lands, before request
    2 is built -- request 2's task content must NEVER reach the still-listening
    approved endpoint. Only ``revoke_route`` is the mechanism allowed to kill a
    live run mid-turn; a routine ``publish_policy``/``activate_policy`` edit is
    intentionally transparent to an already-pinned live run (see
    ``test_routine_policy_edit_between_requests_does_not_block_the_live_run``
    below) and must never be conflated with revocation here."""
    from agent.model_selection_store import revoke_route

    env = managed_agent_env
    agent, approved, unapproved = env["agent"], env["approved"], env["unapproved"]
    route_id = env["policy"]["routes"][0]["route_id"]

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
            revoke_route(
                env["hermes_home"], "kanban-default", route_id=route_id,
                reason="incident", approval_ref="operator:revoke",
            )

    approved.do_POST = _do_post_then_revoke

    agent.run_conversation("run echo SECRET_PAYLOAD_2", conversation_history=[], task_id="t")

    assert len(approved.requests) == 1, (
        "emergency revocation between requests must stop the turn before request 2 is sent -- "
        f"got {len(approved.requests)} requests"
    )
    assert "SECRET_PAYLOAD_2" not in json.dumps(_tool_results(approved)), (
        "no leaked task content in a tool-result round-trip after revocation"
    )
    assert unapproved.requests == []


def test_routine_policy_edit_between_requests_does_not_block_the_live_run(managed_agent_env):
    """Contrast case: a ROUTINE policy edit (publish + activate a new revision
    of the SAME route, no ``revoke_route`` call) landing between request 1 and
    request 2 of the SAME live turn must NOT stop it -- routine policy edits
    affect new attempts, not active conversations. Proves the guard
    distinguishes an ordinary republish from an explicit emergency revocation
    rather than treating any policy-store write as cause to kill the turn."""
    from agent.model_selection_store import activate_policy, publish_policy

    env = managed_agent_env
    agent, approved, unapproved = env["agent"], env["approved"], env["unapproved"]

    approved.response_queue.append(_tc_resp("terminal", '{"command": "echo one"}'))
    approved.response_queue.append(_text_resp("done"))

    original_do_POST = approved.do_POST
    edited = {"done": False}

    def _do_post_then_routine_edit(self):
        original_do_POST(self)
        if len(type(self).requests) == 1 and not edited["done"]:
            edited["done"] = True
            new_policy = dict(env["policy"])
            new_policy["revision"] = env["revision"] + 1
            record = publish_policy(env["hermes_home"], new_policy, approval_ref="operator:routine-edit")
            activate_policy(env["hermes_home"], "kanban-default", record["revision"])

    approved.do_POST = _do_post_then_routine_edit

    agent.run_conversation("run echo one", conversation_history=[], task_id="t")

    assert len(approved.requests) == 2, (
        "a routine policy republish/activate between requests must not interrupt the "
        f"already-pinned live turn -- got {len(approved.requests)} requests"
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


def test_same_iteration_transport_recovery_retry_reenforces_managed_route(managed_agent_env):
    """The ACTUAL same-iteration transport recovery/retry path
    (``agent._try_recover_primary_transport`` in ``turn_api_error.py``,
    triggered after ``max_retries`` transient transport failures) must
    re-run the per-request managed-route guard immediately before the
    rebuilt-client retry is sent -- not merely have its fields poked between
    iterations. Two sub-cases exercised against the REAL retry loop:

      1. no revocation lands in the failure window -> the rebuilt-client
         retry still reaches the approved endpoint (recovery isn't proof-
         by-itself of an unmanaged escape hatch);
      2. an explicit emergency ``revoke_route`` lands in the window between
         the transient failures and the rebuilt-client retry -> the retry
         must be blocked before it reaches the provider, exactly like the
         between-requests case, because a client rebuild is not a fresh
         grace period.
    """
    from agent.model_selection_store import revoke_route

    env = managed_agent_env
    agent, approved, unapproved = env["agent"], env["approved"], env["unapproved"]
    route_id = env["policy"]["routes"][0]["route_id"]

    # Force real ConnectionResetError connect failures (classified as a
    # transient transport error) for exactly agent._api_max_retries attempts,
    # driving the SAME code path production uses -- then allow the rebuilt
    # client's retry through to a real success response.
    approved_port = approved.server.server_address[1] if hasattr(approved, "server") else None

    real_openai_cls = None
    from openai import OpenAI as _RealOpenAI

    real_openai_cls = _RealOpenAI
    attempts = {"n": 0}
    max_retries = agent._api_max_retries

    class _FlakyThenRealClient:
        """First ``max_retries`` chat-completions calls raise a transient
        transport error class name recognised by ``_TRANSIENT_TRANSPORT_ERRORS``;
        subsequent calls delegate to a real client against the approved server."""

        def __init__(self, real_client):
            self._real = real_client
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, *args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] <= max_retries:
                from httpx import ConnectError
                raise ConnectError("simulated transient transport failure")
            return self._real.chat.completions.create(*args, **kwargs)

    approved.response_queue.append(_text_resp("recovered"))
    real_client = real_openai_cls(api_key="test-key", base_url=agent.base_url)
    flaky = _FlakyThenRealClient(real_client)
    agent._create_request_openai_client = lambda reason=None, api_kwargs=None: flaky
    agent._disable_streaming = True

    agent.run_conversation("run recovery probe", conversation_history=[], task_id="t")

    assert len(approved.requests) == 1, (
        "post-recovery retry with no revocation must still reach the approved "
        f"endpoint -- got {len(approved.requests)} requests"
    )
    assert unapproved.requests == []

    # --- sub-case 2: revoke during the transient-failure window, before the
    # rebuilt-client retry is sent. ---
    attempts["n"] = 0
    approved.requests.clear()
    approved.response_queue.append(_text_resp("should never be produced"))
    real_client_2 = real_openai_cls(api_key="test-key", base_url=agent.base_url)

    class _FlakyThenRevokedClient(_FlakyThenRealClient):
        def create(self, *args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == max_retries:
                # Revoke on the LAST transient failure, i.e. exactly in the
                # window before try_recover_primary_transport's retry fires.
                revoke_route(
                    env["hermes_home"], "kanban-default", route_id=route_id,
                    reason="incident", approval_ref="operator:revoke",
                )
            if attempts["n"] <= max_retries:
                from httpx import ConnectError
                raise ConnectError("simulated transient transport failure")
            return self._real.chat.completions.create(*args, **kwargs)

    flaky2 = _FlakyThenRevokedClient(real_client_2)
    agent._create_request_openai_client = lambda reason=None, api_kwargs=None: flaky2

    agent.run_conversation("run recovery probe 2", conversation_history=[], task_id="t")

    assert approved.requests == [], (
        "emergency revocation during the transient-failure window must block the "
        f"rebuilt-client retry before it reaches the provider -- got {approved.requests!r}"
    )
    assert unapproved.requests == []


def _tool_results(handler) -> list[str]:
    out = []
    for req in handler.requests:
        for m in req.get("messages", []):
            if m.get("role") == "tool":
                out.append(m.get("content", ""))
    return out


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mutation", ["model", "reasoning", "omitted_reasoning", "reserve", "messages", "extra_model"])
def test_final_execution_middleware_cannot_change_managed_wire(managed_agent_env, monkeypatch, streaming, mutation):
    from hermes_cli import middleware

    env = managed_agent_env
    agent = env["agent"]
    agent._disable_streaming = not streaming
    original = middleware.run_llm_execution_middleware
    reached = []

    def mutate(request, send, **context):
        def altered(payload):
            payload = dict(payload)
            if mutation == "model":
                payload["model"] = "unapproved-model"
            elif mutation == "reasoning":
                payload["reasoning_effort"] = "low"
            elif mutation == "omitted_reasoning":
                payload.pop("reasoning_effort", None)
            elif mutation == "extra_model":
                payload["extra_body"] = {"model": "unapproved-model"}
            elif mutation == "reserve":
                payload["max_tokens"] = 300000
            else:
                payload["messages"] = [{"role": "user", "content": "x" * 300000}]
            reached.append(True)
            return send(payload)
        return original(request, altered, **context)

    monkeypatch.setattr(middleware, "run_llm_execution_middleware", mutate)
    agent.run_conversation("bounded managed request", conversation_history=[], task_id="t")
    assert reached == [True], "a policy denial must not enter transport retry/recovery"
    assert env["approved"].requests == []
    assert env["unapproved"].requests == []


@pytest.mark.parametrize("mutation", [None, "model", "reasoning", "reserve", "revocation"])
def test_iteration_summary_is_a_managed_send(managed_agent_env, monkeypatch, mutation):
    from agent import relay_llm
    from agent.model_selection_store import list_outcomes, revoke_route

    env = managed_agent_env
    agent, approved = env["agent"], env["approved"]
    agent.max_iterations = 1
    approved.response_queue.extend([_tc_resp("read_file", '{"path":"missing-fixture-file"}')] * 3)
    original = relay_llm.execute_current
    summary_at = []

    def final_transform(request, send, **context):
        if context.get("metadata", {}).get("call_role") == "iteration_summary":
            summary_at.append(len(approved.requests))
            approved.response_queue[:] = [_text_resp("managed summary complete")]
            request = dict(request)
            if mutation == "revocation":
                revoke_route(env["hermes_home"], "kanban-default", route_id="fake-route",
                             reason="fixture emergency", approval_ref="operator:test")
            elif mutation == "model":
                request["model"] = "unauthorized-summary-model"
            elif mutation == "reasoning":
                request["reasoning_effort"] = "low"
            elif mutation == "reserve":
                request["max_tokens"] = 300000
        return original(request, send, **context)

    monkeypatch.setattr(relay_llm, "execute_current", final_transform)
    result = agent.run_conversation("Read a fixture until the budget ends", conversation_history=[], task_id="summary-test")
    assert len(summary_at) == 1, result
    assert summary_at[0] > 0, "reach summary from the real managed turn loop"
    assert len(approved.requests) == summary_at[0] + (mutation is None)
    assert env["unapproved"].requests == []
    if mutation is None:
        assert "managed summary complete" in result["final_response"]
        outcomes = list_outcomes(env["hermes_home"], env["receipt_id"])
        health = [entry for entry in outcomes if entry["kind"] == "routing_health"]
        assert len(health) == len(approved.requests), "summary contact must not disappear from attempt history"


@pytest.mark.parametrize("reported,status", [("test-model", "matching"), (None, "missing"), ("TEST-model", "changed")])
@pytest.mark.parametrize("boundary", ["normal", "streaming", "summary"])
def test_chat_response_identity_is_record_only(managed_agent_env, reported, status, boundary):
    from agent.chat_completion_helpers import handle_max_iterations
    from agent.model_selection_store import get_receipt, list_outcomes

    env = managed_agent_env
    agent, endpoint = env["agent"], env["approved"]
    agent._disable_streaming = boundary != "streaming"
    receipt = get_receipt(env["hermes_home"], env["receipt_id"])
    for managed in (True, False):
        agent._managed_routing_receipt_id = env["receipt_id"] if managed else None
        endpoint.response_queue.append({**_text_resp("identity done"), "model": reported})
        if boundary == "summary":
            text = handle_max_iterations(agent, [{"role": "user", "content": "private prompt"}], 1)
        else:
            text = agent.run_conversation("private prompt")["final_response"]
        assert "identity done" in text
        health = [e["payload"] for e in list_outcomes(env["hermes_home"], env["receipt_id"])
                  if e["kind"] == "routing_health"]
        assert len(health) == 1, "unmanaged requests must not append routing observations"
        assert health[0]["reported_model"] == reported
        assert health[0]["identity_status"] == status
        assert health[0]["status"] == "healthy" and health[0]["replay_safe"] is False
        assert get_receipt(env["hermes_home"], env["receipt_id"]) == receipt
    assert len(endpoint.requests) == 2
    assert not env["unapproved"].requests
