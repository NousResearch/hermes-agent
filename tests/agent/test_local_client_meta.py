"""Loopback-only identity metadata for the local Turbohaul route."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent

from agent.auxiliary_client import (
    _build_call_kwargs,
    reset_runtime_main,
    set_runtime_main,
)
from agent.chat_completion_helpers import _local_client_meta_additions
from agent.local_client_meta import (
    agent_role_metadata,
    build_local_client_meta,
    is_loopback_base_url,
    local_client_meta_extra_body,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11410/v1",
        "http://localhost:11410/v1",
        "http://127.0.0.2:8080",
        "http://[::1]:11410/v1",
    ],
)
def test_loopback_urls_are_detected(url):
    assert is_loopback_base_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://192.168.1.10:8080",
        "file:///tmp/socket",
        "",
        None,
    ],
)
def test_non_loopback_urls_are_rejected(url):
    assert is_loopback_base_url(url) is False


def test_main_agent_role_metadata():
    agent = SimpleNamespace(_delegate_depth=0, _delegate_role=None)
    assert agent_role_metadata(agent) == {"is_main": True, "role": "main"}


def test_delegated_agent_role_metadata():
    agent = SimpleNamespace(_delegate_depth=2, _delegate_role="leaf")
    assert agent_role_metadata(agent) == {
        "is_sub_agent": True,
        "role": "sub_agent",
        "delegate_role": "leaf",
    }


def test_compression_task_has_priority_over_owner_role():
    meta = build_local_client_meta(
        session_id="lineage-root",
        role_metadata={"is_main": True, "role": "main"},
        task="compression",
    )
    assert meta == {
        "session_id": "lineage-root",
        "is_compression": True,
        "role": "compression",
    }


def test_generic_auxiliary_call_is_not_misclassified_as_main():
    meta = build_local_client_meta(
        session_id="lineage-root",
        role_metadata={"is_main": True, "role": "main"},
        task="title_generation",
    )
    assert meta == {
        "session_id": "lineage-root",
        "is_sub_agent": True,
        "role": "sub_agent",
        "auxiliary_task": "title_generation",
    }


def test_loopback_extra_body_preserves_identity_but_remote_gets_nothing():
    local = local_client_meta_extra_body(
        base_url="http://127.0.0.1:11410/v1",
        session_id="lineage-root",
        role_metadata={"is_main": True, "role": "main"},
    )
    assert local == {
        "client_meta": {
            "session_id": "lineage-root",
            "is_main": True,
            "role": "main",
        }
    }
    assert local_client_meta_extra_body(
        base_url="https://api.openai.com/v1",
        session_id="lineage-root",
        role_metadata={"is_main": True, "role": "main"},
    ) == {}


def test_main_builder_uses_lineage_scope_and_delegate_classification():
    main = SimpleNamespace(
        base_url="http://127.0.0.1:11410/v1",
        session_id="physical-segment",
        _delegate_depth=0,
        _delegate_role=None,
    )
    assert _local_client_meta_additions(main, "lineage-root") == {
        "client_meta": {
            "session_id": "lineage-root",
            "is_main": True,
            "role": "main",
        }
    }

    child = SimpleNamespace(
        base_url="http://127.0.0.1:11410/v1",
        session_id="child-session",
        _delegate_depth=1,
        _delegate_role="leaf",
    )
    assert _local_client_meta_additions(child, "") == {
        "client_meta": {
            "session_id": "child-session",
            "is_sub_agent": True,
            "role": "sub_agent",
            "delegate_role": "leaf",
        }
    }


def test_main_chat_completion_wire_contains_loopback_identity(monkeypatch):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.provider = "custom"
    agent.requested_provider = "custom:turbohaul-local"
    agent.base_url = "http://127.0.0.1:11410/v1"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "127.0.0.1"
    agent.model = "qwen3.8-27b"
    agent.session_id = "physical-segment"
    agent._delegate_depth = 0
    agent._delegate_role = None
    monkeypatch.setattr(
        "agent.chat_completion_helpers._prompt_cache_scope_for_agent",
        lambda _agent: "lineage-root",
    )

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hello"}])

    assert kwargs["extra_body"]["client_meta"] == {
        "session_id": "lineage-root",
        "is_main": True,
        "role": "main",
    }


def test_auxiliary_builder_injects_rotation_stable_scope_for_loopback_only():
    token = set_runtime_main(
        "custom",
        "qwen3.8-27b",
        base_url="http://127.0.0.1:11410/v1",
        session_id="physical-segment",
        cache_scope="lineage-root",
    )
    try:
        kwargs = _build_call_kwargs(
            "custom",
            "qwen3.8-27b",
            [{"role": "user", "content": "compress"}],
            base_url="http://127.0.0.1:11410/v1",
            task="compression",
        )
        assert kwargs["extra_body"]["client_meta"] == {
            "session_id": "lineage-root",
            "is_compression": True,
            "role": "compression",
        }

        remote = _build_call_kwargs(
            "custom",
            "qwen3.8-27b",
            [{"role": "user", "content": "compress"}],
            base_url="https://example.com/v1",
            task="compression",
        )
        assert "client_meta" not in remote.get("extra_body", {})
    finally:
        reset_runtime_main(token)
