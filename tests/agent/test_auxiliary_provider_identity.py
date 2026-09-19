"""Provider-identity propagation for auxiliary Responses alias policy."""
from types import SimpleNamespace

import pytest

from agent.auxiliary_client import CodexAuxiliaryClient, _resolve_auto_branch


def _search_files_tool():
    return {
        "type": "function",
        "function": {
            "name": "search_files",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _wire_names(adapter):
    kwargs, _, _ = adapter._build_responses_kwargs({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_search_files_tool()],
    })
    return [tool["name"] for tool in kwargs["tools"]]


@pytest.mark.parametrize("async_mode", [False, True])
def test_auto_route_provider_identity_reaches_request_adapter(monkeypatch, async_mode):
    real_client = SimpleNamespace(
        base_url="https://proxy.example/v1",
        api_key="test-key",
    )
    wrapper = CodexAuxiliaryClient(real_client, "test-model")
    monkeypatch.setattr(
        "agent.auxiliary_client._resolve_auto_route",
        lambda **_kwargs: (wrapper, "test-model", "opencode-go"),
    )

    routed, _ = _resolve_auto_branch(SimpleNamespace(
        main_runtime={}, task="title", model=None,
        async_mode=async_mode, is_vision=False,
    ))
    adapter = routed.chat.completions._sync if async_mode else routed.chat.completions

    assert _wire_names(adapter) == ["hermes_search_files"]


def test_named_custom_provider_identity_reaches_request_adapter(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: test-model\n"
        "custom_providers:\n"
        "  - name: opencode-go-bridge\n"
        "    base_url: https://proxy.example/v1\n"
        "    api_key: test-key\n"
        "    api_mode: codex_responses\n"
    )
    from agent.auxiliary_client import resolve_provider_client

    wrapper, _ = resolve_provider_client("opencode-go-bridge", "test-model")

    assert _wire_names(wrapper.chat.completions) == ["hermes_search_files"]
