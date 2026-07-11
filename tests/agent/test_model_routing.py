from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.model_routing import _first_proposal, apply_pre_model_route
from hermes_cli.plugins import PRE_MODEL_ROUTE_HOOK_VERSION, VALID_HOOKS


@pytest.fixture(autouse=True)
def _registered_route_hook():
    with patch("hermes_cli.plugins.has_hook", return_value=True):
        yield


def test_capability_is_registered():
    assert PRE_MODEL_ROUTE_HOOK_VERSION == 1
    assert "pre_model_route" in VALID_HOOKS


def test_first_valid_proposal_wins_and_aliases_normalize():
    assert _first_proposal([
        None,
        {},
        {
            "model": " ",
            "new_model": " first ",
            "provider": " ",
            "target_provider": " p ",
            "reason": " capacity ",
        },
        {"model": "second"},
    ]) == {"model": "first", "provider": "p", "reason": "capacity"}
    assert _first_proposal([{"model": " "}, "bad"]) is None
    assert _first_proposal([
        {"model": "m", "provider": 123, "reason": "malformed"},
        {"model": "next", "provider": "safe"},
    ]) == {"model": "next", "provider": "safe", "reason": ""}


def _agent():
    agent = SimpleNamespace(
        session_id="s",
        model="old",
        provider="old-provider",
        base_url="https://old",
        api_key="secret",
        platform="cli",
        _user_id="u",
        _gateway_session_key="g",
        _chat_id="c",
        _chat_name="name",
        _chat_type="dm",
        _thread_id="t",
    )
    agent.switch_model = MagicMock()
    return agent


@patch("hermes_cli.config.get_compatible_custom_providers", return_value=[])
@patch("hermes_cli.config.load_config_readonly", return_value={})
@patch("hermes_cli.model_switch.switch_model")
@patch("hermes_cli.plugins.invoke_hook")
def test_successful_route_uses_hermes_resolver(
    mock_hook, mock_resolve, _config, _custom
):
    mock_hook.return_value = [
        {"model": "next", "provider": "target", "reason": "capacity"}
    ]
    mock_resolve.return_value = SimpleNamespace(
        success=True,
        new_model="resolved",
        target_provider="canonical",
        api_key="resolved-key",
        base_url="https://new",
        api_mode="chat_completions",
    )
    agent = _agent()
    messages = [{"role": "user", "content": "hello"}]

    assert apply_pre_model_route(
        agent, user_message="hello", messages=messages, is_first_turn=True
    )
    assert mock_hook.call_args.kwargs["conversation_history"] == messages
    assert mock_hook.call_args.kwargs["conversation_history"] is not messages
    assert "api_key" not in mock_hook.call_args.kwargs
    mock_resolve.assert_called_once()
    agent.switch_model.assert_called_once_with(
        new_model="resolved",
        new_provider="canonical",
        api_key="resolved-key",
        base_url="https://new",
        api_mode="chat_completions",
        prune_fallback_chain=False,
    )


@patch(
    "hermes_cli.plugins.invoke_hook",
    return_value=[{}, {"model": " ", "reason": "TOP-SECRET"}],
)
def test_invalid_proposal_is_noop(_hook, caplog):
    agent = _agent()
    assert not apply_pre_model_route(
        agent, user_message="x", messages=[], is_first_turn=True
    )
    agent.switch_model.assert_not_called()
    assert "ignored malformed proposal" in caplog.text
    assert "TOP-SECRET" not in caplog.text


@patch("hermes_cli.plugins.invoke_hook", return_value=[{"model": "old"}])
@patch("hermes_cli.model_switch.switch_model")
def test_exact_current_route_is_noop(mock_resolve, _hook):
    agent = _agent()
    assert not apply_pre_model_route(
        agent, user_message="x", messages=[], is_first_turn=False
    )
    mock_resolve.assert_not_called()
    agent.switch_model.assert_not_called()


@patch("hermes_cli.plugins.invoke_hook", return_value=[])
def test_payload_uses_agent_image_detection_and_deep_copies_messages(mock_hook):
    agent = _agent()
    agent._content_has_image_parts = MagicMock(return_value=True)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "original"}}],
        }
    ]

    user_message = [{"type": "text", "text": {"value": "original-user"}}]

    assert not apply_pre_model_route(
        agent, user_message=user_message, messages=messages, is_first_turn=True
    )

    payload = mock_hook.call_args.kwargs
    assert payload["has_images"] is True
    assert payload["user_message"] == user_message
    assert payload["user_message"] is not user_message
    payload["user_message"][0]["text"]["value"] = "mutated-user"
    assert user_message[0]["text"]["value"] == "original-user"
    assert payload["messages"] == messages
    assert payload["messages"] is not messages
    assert payload["conversation_history"] == messages
    assert payload["conversation_history"] is not messages
    payload["messages"][0]["content"][0]["image_url"]["url"] = "mutated"
    assert messages[0]["content"][0]["image_url"]["url"] == "original"
    assert (
        payload["conversation_history"][0]["content"][0]["image_url"]["url"]
        == "original"
    )
    agent._content_has_image_parts.assert_called_once_with(messages[0]["content"])


@patch("hermes_cli.plugins.has_hook", return_value=False)
def test_no_registered_hook_skips_payload_copy(_has_hook):
    agent = _agent()
    with patch("agent.model_routing._deepcopy_hook_payload") as copy_payload:
        assert not apply_pre_model_route(
            agent, user_message="x", messages=[], is_first_turn=True
        )
    copy_payload.assert_not_called()


@patch("hermes_cli.plugins.invoke_hook")
def test_uncopyable_payload_skips_hook_fail_open(mock_hook, caplog):
    class Uncopyable:
        def __deepcopy__(self, _memo):
            raise TypeError("not copyable")

    agent = _agent()
    assert not apply_pre_model_route(
        agent,
        user_message=Uncopyable(),
        messages=[],
        is_first_turn=True,
    )
    mock_hook.assert_not_called()
    assert "continuing with the current route" in caplog.text


@patch("hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("plugin broke"))
def test_hook_exception_is_fail_open(_hook, caplog):
    agent = _agent()
    assert not apply_pre_model_route(
        agent, user_message="x", messages=[], is_first_turn=True
    )
    assert "continuing with the current route" in caplog.text
    agent.switch_model.assert_not_called()


def test_shell_hook_route_result_is_narrowly_normalized():
    from agent.shell_hooks import _parse_response

    assert _parse_response(
        "pre_model_route",
        '{"model":" ","new_model":" m ","provider":" ","target_provider":" p ","extra":"ignored"}',
    ) == {"model": "m", "provider": "p"}
    assert _parse_response("pre_model_route", '{"provider":"p"}') is None
    assert _parse_response("pre_model_route", '{"model":"m","provider":123}') is None


def test_fallback_chain_is_preserved_for_routing_and_pruned_by_default():
    from agent.agent_runtime_helpers import _fallback_chain_after_switch

    chain = [
        {"provider": "old-provider", "model": "old-fallback"},
        {"provider": "other", "model": "safe"},
    ]
    assert (
        _fallback_chain_after_switch(chain, "old-provider", "new-provider", prune=False)
        == chain
    )
    assert _fallback_chain_after_switch(chain, "old-provider", "new-provider") == [
        {"provider": "other", "model": "safe"}
    ]
