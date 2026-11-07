"""The provider-fallback switch notice reads its copy through agent.i18n (#148428).

``try_activate_fallback`` hardcoded the English frame ("Model fallback: ... via ...
unavailable (...)") regardless of ``display.language``. The notice now resolves through
``gateway.errors.model_fallback`` (+ ``model_fallback_retry`` for the cooldown tail), so a
localized interface gets the localized frame with the same {slots} and the same switch
semantics. The reason label stays as ``_fallback_reason_text`` output: operator-facing,
greppable, untranslated — the localization is of the frame, not the failure taxonomy.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.error_classifier import FailoverReason


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()  # type: ignore[assignment]  # tests stub the client
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


def _activate(agent):
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_client(), "resolved"),
    ):
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True


def test_fallback_notice_uses_english_frame():
    """The notice agrees with the catalog key, slot for slot."""
    agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
    _activate(agent)
    pending = agent._pending_fallback_notice
    assert pending and len(pending) == 1
    notice = str(pending[0])
    assert notice.startswith("⚠️ Model fallback: ")
    assert "gpt-4o" in notice
    assert "openai" in notice
    assert "rate limit" in notice
    assert "unavailable" in notice
    assert "recovery is not guaranteed" in notice


def test_fallback_notice_localizes_under_display_language(tmp_path, monkeypatch):
    """A zh interface must render the zh frame (not the English one) on the same event path."""
    (tmp_path / "config.yaml").write_text("display:\n  language: zh\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import i18n

    i18n.reset_language_cache()
    try:
        assert i18n.get_language() == "zh"
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        _activate(agent)
        pending = agent._pending_fallback_notice
        assert pending
        notice = str(pending[0])
        assert "模型降级" in notice
        assert "gpt-4o" in notice
        assert "已切换为" in notice
        assert "回切不保证成功" in notice
    finally:
        i18n.reset_language_cache()
