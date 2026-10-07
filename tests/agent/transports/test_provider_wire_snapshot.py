"""Golden wire snapshot: the exact request every registered provider is sent.

Each cell builds one request through the production assembly path
(``agent.turn_api_request.build_api_request``: cache decoration, sanitization, transport
kwargs, Codex preflight, request middleware) for one provider x three reasoning settings,
and compares it against ``tests/fixtures/provider_wire_snapshot.json``.

A change to shared request code that alters what ANY provider receives fails here, naming
the provider and the first diverging key, before a user on that provider finds out. A new
provider fails ``test_snapshot_covers_every_provider`` until its wire is reviewed and recorded.

Intentional wire changes: regenerate and review the JSON diff in the same PR::

    HERMES_UPDATE_WIRE_SNAPSHOT=1 .venv/bin/python -m pytest tests/agent/transports/test_provider_wire_snapshot.py
"""

from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

SNAPSHOT = Path(__file__).resolve().parents[2] / "fixtures" / "provider_wire_snapshot.json"
UPDATE = os.environ.get("HERMES_UPDATE_WIRE_SNAPSHOT") == "1"

REASONING = {
    "unset": None,
    "high": {"enabled": True, "effort": "high"},
    "off": {"enabled": False},
}

TOOLS = [{"type": "function", "function": {
    "name": "terminal", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]

# One completed tool round, then a fresh user turn: role mapping, tool-call/result pairing
# and every cache breakpoint slot.
MESSAGES = [
    {"role": "system", "content": "You are a test agent."},
    {"role": "user", "content": "list files"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
        "name": "terminal", "arguments": "{\"command\": \"ls\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"},
    {"role": "user", "content": "thanks, now summarize"},
]

# A representative model per provider (else the profile's first curated model).
MODELS = {
    "nous": "anthropic/claude-opus-5.5", "openai-codex": "gpt-5.5", "openai-api": "gpt-5.5",
    "xai-oauth": "grok-4", "qwen-oauth": "qwen3-coder-plus", "lmstudio": "qwen/qwen3-8b",
    "copilot": "gpt-5.5", "gemini": "gemini-3-pro", "kimi-coding": "kimi-k3", "kimi-coding-cn": "kimi-k2.5",
    "stepfun": "step-3", "arcee": "trinity-large", "actual": "actual-1", "minimax": "MiniMax-M3",
    "minimax-oauth": "MiniMax-M3", "minimax-cn": "MiniMax-M3", "anthropic": "claude-opus-5.5",
    "alibaba": "qwen3.5-plus", "alibaba-coding-plan": "qwen3-coder-plus", "xai": "grok-4",
    "ai-gateway": "anthropic/claude-opus-5.5", "opencode-zen": "claude-opus-5.5", "opencode-go": "kimi-k2.5",
    "kilocode": "anthropic/claude-opus-5.5", "xiaomi": "mimo-v2-pro", "tencent-tokenhub": "hunyuan-t1",
    "tencent-tokenplan": "hunyuan-t1", "ollama-cloud": "gpt-oss:120b",
    "bedrock": "us.anthropic.claude-opus-5-5-v1:0", "vertex": "gemini-3-pro", "azure-foundry": "gpt-5.5",
    "alibaba-cn": "qwen3.5-plus", "alibaba-token-plan": "qwen3.5-plus", "alibaba-token-plan-cn": "qwen3.5-plus",
    "alibaba-coding-plan-cn": "qwen3-coder-plus", "deepinfra": "deepseek-ai/DeepSeek-V4",
    "router": "openai/gpt-5.5", "openrouter": "anthropic/claude-opus-5.5", "custom": "my-local-model",
}
# Providers with a user-configured endpoint, or outside the auth registry.
BASE_URLS = {
    "vertex": "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi",
    "azure-foundry": "https://example.openai.azure.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "http://127.0.0.1:8080/v1",
}

# Values that legitimately differ per run or per release; everything else compares exactly.
_VOLATILE = (
    (re.compile(r"client=hermes-client-v[^\"]*"), "client=hermes-client-v<VERSION>"),
    (re.compile(r'"promptId": "[0-9a-f-]{36}"'), '"promptId": "<UUID>"'),
)


def _cells() -> list[tuple[str, str, str]]:
    from hermes_cli.auth import PROVIDER_REGISTRY
    from providers import get_provider_profile

    seen: set[str] = set()
    cells = []
    for pid, cfg in PROVIDER_REGISTRY.items():
        profile = get_provider_profile(pid)
        name = profile.name if profile else pid
        if name in seen or cfg.auth_type == "external_process":
            continue
        seen.add(name)
        fallback = profile.fallback_models[0] if profile and profile.fallback_models else "test-model"
        cells.append((name, MODELS.get(name, fallback), BASE_URLS.get(name) or cfg.inference_base_url))
    for name in ("openrouter", "custom"):
        if name not in seen:
            cells.append((name, MODELS[name], BASE_URLS[name]))
    return cells


@pytest.fixture
def offline(monkeypatch):
    """No network and no ambient AWS profile. Capability probes (Ollama ``/api/show``, LM Studio,
    catalogs) must take their offline answer, or the recorded wire depends on a live server.
    Name resolution is blocked too: some clients connect through paths ``socket.connect`` misses."""
    real_connect = socket.socket.connect

    def guarded(sock, address, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError(f"network disabled in wire snapshot: {address}")
        return real_connect(sock, address, *args, **kwargs)

    def no_dns(*_args, **_kwargs):
        raise socket.gaierror("DNS disabled in wire snapshot")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)


def _wire(provider: str, model: str, base_url: str) -> dict[str, Any]:
    from agent.turn_api_request import build_api_request
    from agent.turn_context import _PER_TURN_RESET_STATE
    from run_agent import AIAgent

    with patch("model_tools.get_tool_definitions", return_value=TOOLS), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(provider=provider, model=model, base_url=base_url, api_key="sk-test-0000000000",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, session_id="wire-snapshot")
    for name, value in _PER_TURN_RESET_STATE:
        setattr(agent, name, value)
    out: dict[str, Any] = {"api_mode": agent.api_mode}
    for label, reasoning in REASONING.items():
        agent.reasoning_config = reasoning
        messages = json.loads(json.dumps(MESSAGES))
        built = build_api_request(
            agent, api_messages=messages, _moa_prepared_request=None, tools_for_api=agent.tools,
            system_message=messages[0]["content"], messages=messages,
            original_user_message=messages[-1]["content"], approx_tokens=0, total_chars=0, retry_count=0,
            api_call_count=1, api_request_id="req-1", api_start_time=0.0, effective_task_id="task-1",
            turn_id="turn-1",
        )
        text = json.dumps(built.api_kwargs, sort_keys=True, default=repr)
        for pattern, replacement in _VOLATILE:
            text = pattern.sub(replacement, text)
        out[label] = json.loads(text)
    return out


def _first_divergence(expected: Any, actual: Any, path: str = "") -> str:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                return f"{path}.{key}: missing (expected {json.dumps(expected[key])[:200]})"
            if key not in expected:
                return f"{path}.{key}: unexpected {json.dumps(actual[key])[:200]}"
            if expected[key] != actual[key]:
                return _first_divergence(expected[key], actual[key], f"{path}.{key}")
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for i, (e, a) in enumerate(zip(expected, actual)):
            if e != a:
                return _first_divergence(e, a, f"{path}[{i}]")
    return f"{path}: expected {json.dumps(expected)[:200]} got {json.dumps(actual)[:200]}"


def _load() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}


CELLS = _cells()


@pytest.mark.parametrize("provider,model,base_url", CELLS, ids=[c[0] for c in CELLS])
def test_provider_wire_matches_snapshot(offline, provider, model, base_url):
    actual = _wire(provider, model, base_url)
    if UPDATE:
        golden = _load()
        golden[provider] = {"model": model, "base_url": base_url, **actual}
        SNAPSHOT.write_text(json.dumps(dict(sorted(golden.items())), indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        return
    expected = _load().get(provider)
    assert expected is not None, f"{provider} has no recorded wire; regenerate the snapshot (module docstring)"
    expected = {k: v for k, v in expected.items() if k not in ("model", "base_url")}
    assert actual == expected, f"{provider} wire changed — {_first_divergence(expected, actual)}"


def test_snapshot_covers_every_provider():
    recorded, current = set(_load()), {c[0] for c in CELLS}
    assert current <= recorded, f"providers without a recorded wire: {sorted(current - recorded)}"
    assert recorded <= current, f"snapshot entries for providers that no longer exist: {sorted(recorded - current)}"
