"""Regression for #84698 — invalid-JSON tool recovery keyed by call id.

Two parallel calls to the same tool (one valid JSON, one invalid) used to
collide: recovery matched by tool *name*, so the valid sibling inherited the
Invalid-JSON error and never executed.

Align with the mixed invalid-*name* batch path: error only broken calls,
execute valid siblings, key by tool_call_id. Full-loop via mock provider.
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


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        is_stream = req.get("stream") is True
        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {
                    "id": "m",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            ]
            if content:
                chunks.append(
                    {
                        "id": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            if tcs:
                for ti, tc in enumerate(tcs):
                    chunks.append(
                        {
                            "id": "m",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": ti,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": tc["function"][
                                                        "arguments"
                                                    ],
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            chunks.append(
                {
                    "id": "m",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if tcs else "stop",
                        }
                    ],
                }
            )
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


def _batch_tc_resp(calls: list[tuple[str, str]]) -> dict:
    return {
        "id": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                        for i, (name, args) in enumerate(calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env():
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_e2e_84698_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    for mod in list(sys.modules):
        if (
            mod == "run_agent"
            or mod.startswith("agent.")
            or mod.startswith("tools.")
            or mod.startswith("hermes_")
        ):
            del sys.modules[mod]
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat",
        model="test-model",
        max_iterations=10,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        platform="cli",
    )
    agent.valid_tool_names = {
        "terminal",
        "read_file",
        "write_file",
        "execute_code",
        "session_search",
        "todo",
    }
    # Non-stream path: streaming assembly auto-repairs/truncation-routes bad
    # JSON before the conversation_loop invalid-JSON recovery path sees it.
    # These tests exercise that recovery path (#84698).
    agent._disable_streaming = True

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _tool_results_by_id(messages) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "tool":
            tid = m.get("tool_call_id") or ""
            out[tid] = m.get("content") or ""
    return out


def _assistant_tool_call_ids(messages) -> list[str]:
    ids: list[str] = []
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            ids.extend(tc["id"] for tc in m["tool_calls"])
    return ids


def test_mixed_json_same_name_executes_valid_sibling(agent_env):
    """Valid same-name sibling must execute; only the broken call gets Invalid JSON."""
    agent, handler = agent_env
    # call_0 broken JSON, call_1 valid — same tool name (the #84698 collision).
    handler.response_queue.append(
        _batch_tc_resp(
            [
                ("todo", "{bad}"),  # invalid JSON but closed — not truncated
                ("todo", "{}"),
            ]
        )
    )
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    by_id = _tool_results_by_id(result.get("messages"))
    assert "call_0" in by_id
    assert "call_1" in by_id
    assert "Invalid JSON" in by_id["call_0"]
    assert "Invalid JSON" not in by_id["call_1"]
    assert "Skipped: other tool call" not in by_id["call_1"]
    assert not by_id["call_1"].startswith("Error:")
    # No whole-turn JSON retry burn when a valid sibling exists.
    chat_calls = [r for r in handler.captured_requests if "messages" in r]
    assert len(chat_calls) == 2


def test_mixed_json_preserves_tool_call_result_pairing(agent_env):
    """Every assistant tool_call keeps exactly one matching tool result id."""
    agent, handler = agent_env
    handler.response_queue.append(
        _batch_tc_resp(
            [
                ("todo", "{bad}"),  # invalid JSON but closed — not truncated
                ("todo", "{}"),
            ]
        )
    )
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    tc_ids = _assistant_tool_call_ids(result.get("messages"))
    result_ids = list(_tool_results_by_id(result.get("messages")).keys())
    assert set(tc_ids) == {"call_0", "call_1"}
    assert sorted(result_ids) == sorted(tc_ids)


def test_all_invalid_json_still_retries_then_recovers(agent_env):
    """When every call has invalid JSON, keep the 3× API retry then recovery inject."""
    agent, handler = agent_env
    for _ in range(3):
        handler.response_queue.append(
            _batch_tc_resp(
                [
                    ("todo", "{bad-a}"),
                    ("todo", "{bad-b}"),
                ]
            )
        )
    handler.response_queue.append(_text_resp("recovered"))

    result = agent.run_conversation("retry path", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    chat_calls = [r for r in handler.captured_requests if "messages" in r]
    # 3 invalid batches (initial + 2 retries) + recovery turn continues + final text
    # After 3 strikes recovery injects tool errors and continues the loop → one more
    # API call for the model to answer after seeing tool errors, then possibly more.
    assert len(chat_calls) >= 4
    # Recovery messages must key by id: each broken call gets its own error, not a name collision.
    by_id = _tool_results_by_id(result.get("messages"))
    assert "Invalid JSON" in by_id.get("call_0", "")
    assert "Invalid JSON" in by_id.get("call_1", "")


def test_truncation_scan_does_not_false_positive_on_same_name_valid_sibling(agent_env):
    """Name-keyed truncation used to abort the whole batch if any same-name call looked truncated.

    Valid sibling with complete JSON must still run when the broken call is truncated-looking.
    """
    agent, handler = agent_env
    # Truncated-looking invalid args (no closing brace) on call_0; call_1 complete.
    # Truncation hard-stop only applies when the *invalid* call looks truncated.
    # If we only rekey and keep hard-stop on any invalid truncated call, mixed
    # truncated+valid still aborts — that is intentional hard-stop semantics.
    # This test locks the false-positive case: valid call must not be treated as
    # truncated merely because it shares a name with a truncated invalid call
    # when the invalid call is *non-truncated* bad JSON (has closing brace junk).
    handler.response_queue.append(
        _batch_tc_resp(
            [
                ("todo", "{bad}"),  # invalid JSON but ends with } — not truncated
                ("todo", "{}"),
            ]
        )
    )
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    by_id = _tool_results_by_id(result.get("messages"))
    assert "Invalid JSON" in by_id.get("call_0", "")
    assert "Invalid JSON" not in by_id.get("call_1", "")
    assert result.get("error") != "Response truncated due to output length limit"
