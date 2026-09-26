"""MoA must observe persisted pool rotations before the runtime TTL (#122600)."""
import json
import time
from types import SimpleNamespace

import pytest

from agent import moa_loop as moa


@pytest.mark.parametrize("role", ["aggregator", "streaming_aggregator", "reference"])
def test_moa_slot_observes_pool_rotation(tmp_path, monkeypatch, role):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(moa, "_runtime_cache", {})
    monkeypatch.setattr(moa, "_preset_cache", {})
    (tmp_path / "config.yaml").write_text("""
model:
  provider: anthropic
  default: claude-sonnet-4-6
moa:
  presets:
    rotation:
      enabled: true
      reference_models:
        - provider: anthropic
          model: claude-sonnet-4-6
          enabled: false
      aggregator:
        provider: anthropic
        model: claude-sonnet-4-6
""", encoding="utf-8")
    store = tmp_path / "auth.json"
    def save(token):
        store.write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": [{
            "id": "fixture-account", "label": "fixture", "auth_type": "oauth",
            "source": "manual:hermes_pkce", "priority": 0, "access_token": token,
            "refresh_token": "fixture-refresh", "expires_at": time.time() + 3600,
        }]}}), encoding="utf-8")
    seen = []
    def wire(**kwargs):
        seen.append(kwargs["api_key"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))], usage=None)
    monkeypatch.setattr(moa, "call_llm", wire)
    facade = moa.MoAChatCompletions("rotation")
    def call():
        messages = [{"role": "user", "content": "hello"}]
        if role == "reference":
            moa._run_reference({"provider": "anthropic", "model": "claude-sonnet-4-6"}, messages)
        else:
            facade.create(messages=messages, stream=role == "streaming_aggregator")
    save("fixture-before")
    call()
    save("fixture-after-rotation")
    call()
    assert seen == ["fixture-before", "fixture-after-rotation"]
