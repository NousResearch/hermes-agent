"""Reactive tool-schema recovery reaches the wire: the retry sends the rewritten schemas.

Apple's ``fm serve`` rejects every tool schema holding a union; Hermes flattens them and retries.

``fm serve`` (the macOS Foundation Models CLI's Chat Completions server, on-device and Private Cloud
Compute) decodes requests with Swift's JSONDecoder and 400s any ``anyOf``/``oneOf`` or ``type``
array with Foundation's generic decode failure. Hermes' built-in ``terminal``, ``skill_manage`` and
``manage_connections`` all carry one, so without the recovery no full-toolset turn reaches the model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error
from tests.fakes.fake_llm_provider import Error, Text, FakeLLMServer, write_hermes_home

ROOT = Path(__file__).parents[2]
# Verbatim from fm serve, curly apostrophes included.
FM_DECODE_FAILURE = "Invalid request: The data couldn’t be read because it isn’t in the correct format."


class _APIError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"message": message, "type": "invalid_request_error", "code": str(status_code)}}
        self.response = SimpleNamespace(headers={})


def test_fm_serve_decode_failure_classifies_as_tool_schema_union():
    for message in (FM_DECODE_FAILURE, FM_DECODE_FAILURE.replace("\u2019", "'")):
        verdict = classify_api_error(_APIError(message, 400), provider="custom", model="pcc")
        assert verdict.reason == FailoverReason.tool_schema_union
        assert (verdict.retryable, verdict.should_fallback) == (False, True)
    other = classify_api_error(_APIError(FM_DECODE_FAILURE, 500), provider="custom", model="pcc")
    assert other.reason != FailoverReason.tool_schema_union


def _has_union(node) -> bool:
    if isinstance(node, dict):
        if any(isinstance(node.get(k), list) for k in ("anyOf", "oneOf")) or isinstance(node.get("type"), list):
            return True
        return any(_has_union(v) for v in node.values())
    return isinstance(node, list) and any(_has_union(v) for v in node)


def _fm_serve(record):
    tools = record["body"].get("tools") or []
    if any(_has_union(t.get("function", {}).get("parameters")) for t in tools):
        return Error(status=400, message=FM_DECODE_FAILURE)
    return Text("PONG")


def _run_one_shot(tmp_path, responder):
    home = tmp_path / "hermes"
    with FakeLLMServer(responder) as srv:
        write_hermes_home(
            home, srv.base_url,
            extra_config="updates:\n  check: false\nmodels_dev:\n  url: http://127.0.0.1:9/api.json\n",
        )
        env = {**os.environ, "HERMES_HOME": str(home), "PYTHONUTF8": "1"}
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        for name in ("HERMES_DESKTOP", "HERMES_DASHBOARD_SESSION_TOKEN"):
            env.pop(name, None)
        proc = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "chat", "-q", "Reply with exactly PONG."],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", timeout=240, check=False,
        )
        requests = srv.main_requests()
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    assert len(requests) == 2, f"expected rejected + rewritten retry; got {len(requests)}"
    assert "PONG" in proc.stdout
    first, retry = (r["tools"] for r in requests)
    assert [t["function"]["name"] for t in first] == [t["function"]["name"] for t in retry]
    return first, retry


def test_full_toolset_turn_reaches_fm_serve_after_flattening_unions(tmp_path):
    first, retry = _run_one_shot(tmp_path, _fm_serve)
    assert _has_union(first) and not _has_union(retry)


def _has_pattern(node) -> bool:
    if isinstance(node, dict):
        return ("pattern" in node and "type" in node) or any(_has_pattern(v) for v in node.values())
    return isinstance(node, list) and any(_has_pattern(v) for v in node)


def _llama_cpp(record):
    if _has_pattern(record["body"].get("tools")):
        return Error(status=400, message="Unable to generate parser for this template. "
                                         "Automatic parser generation failed: error parsing grammar")
    return Text("PONG")


def test_llama_cpp_grammar_strip_reaches_the_retry(tmp_path):
    """Sibling rung, same seam: the strip used to edit ``agent.tools`` while the retry resent the
    failed attempt's request copy, so the second request still carried ``pattern`` and failed."""
    first, retry = _run_one_shot(tmp_path, _llama_cpp)
    assert _has_pattern(first) and not _has_pattern(retry)
