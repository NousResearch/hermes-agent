"""An anthropic_messages snapshot without the native client's fields fails with a named error."""

from types import SimpleNamespace

import pytest

from agent.primary_runtime_snapshot import _build_primary_runtime_snapshot, _snapshot_primary_runtime


def _agent(api_mode, **native):
    return SimpleNamespace(
        model="claude-sonnet-4-5",
        provider="anthropic",
        requested_provider="anthropic",
        base_url="https://api.anthropic.com",
        api_mode=api_mode,
        api_key="sk-test",
        _client_kwargs={},
        _use_prompt_caching=False,
        _use_native_cache_layout=False,
        context_compressor=SimpleNamespace(context_length=200_000, threshold_tokens=150_000),
        **native,
    )


_NATIVE = {
    "_anthropic_api_key": "sk-ant-native",
    "_anthropic_base_url": "https://api.anthropic.com",
    "_is_anthropic_oauth": False,
}


@pytest.mark.parametrize("build", ["init", "switch"])
def test_anthropic_route_without_native_client_raises_named_error(build):
    agent = _agent("anthropic_messages")
    with pytest.raises(RuntimeError, match="native Anthropic client was not initialized") as exc:
        if build == "init":
            _snapshot_primary_runtime(agent)
        else:
            _build_primary_runtime_snapshot(agent, "anthropic_messages")
    assert "_anthropic_api_key" in str(exc.value)
    assert "sk-test" not in str(exc.value)


@pytest.mark.parametrize("build", ["init", "switch"])
def test_anthropic_route_with_native_client_snapshots_its_fields(build):
    agent = _agent("anthropic_messages", **_NATIVE)
    if build == "init":
        _snapshot_primary_runtime(agent)
        rt = agent._primary_runtime
    else:
        rt = _build_primary_runtime_snapshot(agent, "anthropic_messages")
    assert rt["anthropic_api_key"] == "sk-ant-native"
    assert rt["anthropic_base_url"] == "https://api.anthropic.com"
    assert rt["is_anthropic_oauth"] is False


def test_other_routes_do_not_need_native_fields():
    agent = _agent("chat_completions")
    _snapshot_primary_runtime(agent)
    assert "anthropic_api_key" not in agent._primary_runtime
