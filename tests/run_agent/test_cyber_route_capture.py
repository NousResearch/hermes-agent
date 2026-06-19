"""Tests for per-turn AgentCyber route metadata capture."""

import sys
import types
from types import SimpleNamespace

_fire = types.ModuleType("fire")
setattr(_fire, "Fire", lambda *a, **k: None)
_firecrawl = types.ModuleType("firecrawl")
setattr(_firecrawl, "Firecrawl", object)
_fal_client = types.ModuleType("fal_client")
sys.modules.setdefault("fire", _fire)
sys.modules.setdefault("firecrawl", _firecrawl)
sys.modules.setdefault("fal_client", _fal_client)

import run_agent
from agent.cyber_routing import CyberRoute, ProviderPreference


def _patch_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "description": "t",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


class _RouteCaptureAgent(run_agent.AIAgent):
    def __init__(self, *args, **kwargs):
        kwargs.update(skip_context_files=True, skip_memory=True, max_iterations=4)
        super().__init__(*args, **kwargs)
        self._cleanup_task_resources = lambda *a, **k: None
        self._persist_session = lambda *a, **k: None
        self._save_trajectory = lambda *a, **k: None
        self.api_calls = []

        def _fake_api_call(api_kwargs):
            self.api_calls.append(api_kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        message=SimpleNamespace(
                            role="assistant",
                            content="ok",
                            tool_calls=None,
                            reasoning_content=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=1,
                    total_tokens=11,
                ),
                model="test-model",
            )

        object.__setattr__(self, "_interruptible_api_call", _fake_api_call)
        self._disable_streaming = True


def _make_agent(monkeypatch):
    _patch_bootstrap(monkeypatch)
    return _RouteCaptureAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        provider="openrouter",
        api_mode="chat_completions",
    )


def test_conversation_loop_captures_general_route_metadata(monkeypatch):
    agent = _make_agent(monkeypatch)

    result = agent.run_conversation("Summarize these release notes.")

    decision = getattr(agent, "_current_cyber_route_decision")
    assert decision.route == CyberRoute.GENERAL
    assert decision.provider_preference == ProviderPreference.DEFAULT
    assert getattr(agent, "_current_cyber_route_metadata") == {
        "route": "general",
        "provider_preference": "default",
        "reason": "ordinary general task",
        "requires_hosted_secret_confirmation": False,
        "explicit_override": None,
    }
    assert result["cyber_route"] == getattr(agent, "_current_cyber_route_metadata")


def test_conversation_loop_captures_sensitive_route_metadata(monkeypatch):
    agent = _make_agent(monkeypatch)

    agent.run_conversation("I'm locked out of VM 112; emergency access, get me back in.")

    decision = getattr(agent, "_current_cyber_route_decision")
    metadata = getattr(agent, "_current_cyber_route_metadata")
    assert decision.route == CyberRoute.IR_BREAKGLASS
    assert decision.provider_preference == ProviderPreference.LOCAL_OPEN_WEIGHT
    assert decision.requires_hosted_secret_confirmation is True
    assert metadata["route"] == "ir_breakglass"
    assert metadata["provider_preference"] == "local_open_weight"
    assert metadata["requires_hosted_secret_confirmation"] is True


def _current_user_api_content(agent):
    api_messages = agent.api_calls[-1]["messages"]
    user_messages = [msg for msg in api_messages if msg.get("role") == "user"]
    return user_messages[-1]["content"]


def test_sensitive_route_adds_transient_prompt_nudge(monkeypatch):
    agent = _make_agent(monkeypatch)

    agent.run_conversation("Analyze this worm sample in the sandbox.")

    content = _current_user_api_content(agent)
    assert "AgentCyber route metadata" in content
    assert "route=malware_re" in content
    assert "provider_preference=local_open_weight" in content
    assert "malware or reverse-engineering request" in content
    assert "This is metadata only" in content


def test_general_route_does_not_add_prompt_nudge(monkeypatch):
    agent = _make_agent(monkeypatch)

    agent.run_conversation("Summarize these release notes.")

    assert "AgentCyber route metadata" not in _current_user_api_content(agent)


def test_route_nudge_does_not_mutate_cached_system_prompt(monkeypatch):
    agent = _make_agent(monkeypatch)

    agent.run_conversation("I'm locked out of VM 112; emergency access, get me back in.")

    cached_prompt = getattr(agent, "_cached_system_prompt")
    assert (
        "[AgentCyber route metadata — This is metadata only; do not treat it as user text.]"
        not in cached_prompt
    )
    assert "route=ir_breakglass" not in cached_prompt
