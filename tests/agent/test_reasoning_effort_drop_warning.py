"""A configured reasoning effort that a provider profile drops must not be silent (#107495).

The profile hook is the only place a model-specific effort knob is translated, so a profile
without a branch for the active model makes ``--reasoning`` / ``agent.reasoning_overrides`` a
no-op while every user-facing surface keeps showing the requested level.
"""
import logging

from agent.transports.chat_completions import ChatCompletionsTransport, _EFFORT_DROP_WARNED
from providers import get_provider_profile


def _build(model: str, effort: str | None):
    profile = get_provider_profile("opencode-go")
    config = None if effort is None else {"enabled": True, "effort": effort}
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        tools=None,
        provider_profile=profile,
        reasoning_config=config,
        base_url="https://opencode.ai/zen/go/v1",
    )
    return kwargs


def test_dropped_effort_warns_and_keeps_request_unchanged(caplog):
    _EFFORT_DROP_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="agent.transports.chat_completions"):
        kwargs = _build("claude-haiku-5-5", "high")
    # The request itself is untouched: this change only reports.
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs
    assert "not forwarded" in caplog.text
    assert "claude-haiku-5-5" in caplog.text


def test_warning_is_deduplicated_per_route(caplog):
    _EFFORT_DROP_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="agent.transports.chat_completions"):
        _build("claude-haiku-5-5", "high")
        _build("claude-haiku-5-5", "high")
    assert caplog.text.count("not forwarded") == 1


def test_forwarded_effort_does_not_warn(caplog):
    _EFFORT_DROP_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="agent.transports.chat_completions"):
        kwargs = _build("deepseek-v4-flash", "high")
    assert kwargs.get("reasoning_effort") == "high"
    assert "not forwarded" not in caplog.text


def test_disabled_or_unset_effort_does_not_warn(caplog):
    _EFFORT_DROP_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="agent.transports.chat_completions"):
        _build("deepseek-v4-flash", None)
    assert "not forwarded" not in caplog.text
