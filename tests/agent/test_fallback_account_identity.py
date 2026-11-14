"""#154428 / #91077: two named providers on one URL + model with different credentials are two
accounts (two quotas). After a 429 on one, the other is tried on every fallback walk; the same
account listed twice is still skipped. Credentials come from ``key_cmd`` (never minted here)."""

import json
import os
from pathlib import Path

import pytest

URL = "http://127.0.0.1:9/v1"


def _write_config(key_b_cmd: str) -> None:
    providers = {
        "account-a": {"base_url": URL, "key_cmd": "echo key-a", "default_model": "shared-model"},
        "account-b": {"base_url": URL, "key_cmd": key_b_cmd, "default_model": "shared-model"},
    }
    cfg = {"model": {"provider": "custom:account-a", "default": "shared-model"}, "providers": providers}
    # JSON is valid YAML; the CI test env ships no pyyaml.
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(json.dumps(cfg))


@pytest.mark.parametrize("key_b_cmd, skipped", [("echo key-b", False), ("echo key-a", True)],
                         ids=["second-account-tried", "same-account-skipped"])
def test_auxiliary_fallback_walk_tells_accounts_apart(key_b_cmd, skipped):
    """The aux skip predicate (fallback_chain, main fallback_providers, main-model last resort)."""
    _write_config(key_b_cmd)
    from agent.auxiliary_client import _failed_backend_skip

    skip = _failed_backend_skip("custom:account-a", "shared-model", failed_base_url=URL)
    assert skip("custom:account-b", "shared-model", URL, {"provider": "custom:account-b"}) is skipped


@pytest.mark.parametrize("key_b_cmd, skipped", [("echo key-b", False), ("echo key-a", True)],
                         ids=["second-account-tried", "same-account-skipped"])
def test_main_fallback_tells_accounts_apart(monkeypatch, key_b_cmd, skipped):
    """The main agent's fallback_providers walk, primary on a key_cmd token source."""
    _write_config(key_b_cmd)
    from types import SimpleNamespace

    from agent import chat_completion_helpers as cch
    from agent.command_token_source import build_command_token_provider

    monkeypatch.setattr("agent.fallback_cooldown._is_entitlement_rejected", lambda *a, **k: False)
    monkeypatch.setattr(cch, "_candidate_pool_exhausted", lambda *a, **k: False)
    monkeypatch.setattr(cch, "_fallback_entry_unavailable_without_network", lambda *a, **k: None)
    agent = SimpleNamespace(provider="custom:account-a", model="shared-model", base_url=URL,
                            api_key=build_command_token_provider("echo key-a", "account-a"))
    entry = {"provider": "custom:account-b", "model": "shared-model", "base_url": URL}
    key = ("custom:account-b", "shared-model")
    assert cch._should_skip_fallback_candidate(
        agent, entry, key, "custom:account-b", "shared-model", set()) is skipped
