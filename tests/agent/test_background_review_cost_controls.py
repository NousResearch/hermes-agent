"""Unit coverage for the background-review aux-model selector + routed digest.

Covers the two behaviors this change adds:
  • _resolve_review_runtime — auto/same-model → not routed (main model, warm
    cache); a configured different model → routed with resolved credentials.
  • _digest_history — compact replay used ONLY on the routed path (recent tail
    verbatim + a digest of older turns), preserving role alternation.

Pure-function / config-driven; no live model calls.
"""
from typing import Any
from unittest.mock import patch

import pytest

from agent import background_review as br
from agent import i18n


def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


# ---------------------------------------------------------------------------
# _resolve_review_runtime — the aux-model selector
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, provider="openai-codex", model="gpt-5.5"):
        self.provider = provider
        self.model = model
        self._credential_pool: Any = None
        self.request_overrides = {}
        self.max_tokens: int | None = None

    def _current_main_runtime(self):
        return {
            "api_key": "parent-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_app_server",
        }


def test_routing_auto_inherits_parent_and_downgrades_codex_app_server():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"provider": "auto", "model": ""}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    assert rt["api_mode"] == "codex_responses"  # downgraded so agent-loop tools dispatch


def test_routing_to_different_model_marks_routed_and_resolves_credentials():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    fake_rp = {
        "provider": "openrouter", "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions",
        "credential_pool": "routed-pool",
        "request_overrides": {"extra_body": {"store": False}},
        "max_output_tokens": 2048,
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_rp):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is True
    assert rt["provider"] == "openrouter"
    assert rt["model"] == "google/gemini-3-flash-preview"
    assert rt["api_key"] == "or-key"
    assert rt["credential_pool"] == "routed-pool"
    assert rt["request_overrides"] == {"extra_body": {"store": False}}
    assert rt.get("max_tokens") is None


def test_unrouted_runtime_keeps_parent_pool_and_overrides():
    agent = _FakeAgent()
    agent._credential_pool = "parent-pool"
    agent.request_overrides = {"service_tier": "priority"}
    agent.max_tokens = 4096
    with patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}):
        rt = br._resolve_review_runtime(agent)
    assert rt["credential_pool"] == "parent-pool"
    assert rt["request_overrides"] == {"service_tier": "priority"}
    assert rt["max_tokens"] == 4096


def test_routing_same_model_as_parent_is_not_routed():
    agent = _FakeAgent(provider="openrouter", model="anthropic/claude-opus-4.8")
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "anthropic/claude-opus-4.8",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False  # same model/provider → keep full-replay path


def test_routing_resolution_failure_falls_back_to_parent():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=RuntimeError("boom")):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"


# ---------------------------------------------------------------------------
# _digest_history — routed-path compact replay
# ---------------------------------------------------------------------------

def test_digest_under_tail_returns_full():
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert br._digest_history(msgs, tail=24) == msgs


def test_digest_collapses_old_keeps_tail_verbatim():
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 50))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 50))
    out = br._digest_history(msgs, tail=10)
    # First message is the synthetic digest (user role → alternation preserved).
    assert out[0]["role"] == "user"
    # Recent tail preserved verbatim.
    assert out[-1] == msgs[-1]
    assert len(out) == 11  # 1 digest + 10 tail


def test_digest_does_not_open_tail_on_a_tool_message():
    msgs = []
    for i in range(40):
        msgs.append(_msg("user", "u" + "x" * 50))
        msgs.append(_msg("assistant", "", tool_calls=[
            {"function": {"name": "terminal", "arguments": "{}"}}]))
        msgs.append({"role": "tool", "content": "result " + "w" * 50})
    out = br._digest_history(msgs, tail=2)
    # The verbatim tail (after the digest) must not begin on a bare tool message.
    assert out[1]["role"] != "tool"




# ---------------------------------------------------------------------------
# Cost / configurability controls (issue #87250)
# ---------------------------------------------------------------------------



def test_enabled_false_disables_automatic_review():
    cfg = {"auxiliary": {"background_review": {"enabled": False}}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert br.load_background_review_settings()[0] is False


def test_unresolvable_review_provider_falls_back_with_visible_warning(caplog):
    """The fork silently ran on the main model with only a debug line (#116055): the fallback must
    identify the affected setting at WARNING and reach the user-visible rail without echoing route data."""
    import logging

    agent = _FakeAgent()
    emitted = []
    agent._emit_warning = emitted.append
    cfg = {"auxiliary": {"background_review": {"provider": "no-such-provider", "model": "review-model"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        with caplog.at_level(logging.WARNING, logger="agent.background_review"):
            rt = br._resolve_review_runtime(agent)
            br._resolve_review_runtime(agent)

    assert rt["routed"] is False and rt["model"] == "gpt-5.5"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 2
    assert all("auxiliary.background_review" in w and "AuthError" in w for w in warnings)
    from agent.i18n import t
    expected = t("display.review.routing_fallback_warning", task_provider="…", task_model="…",
                 error="AuthError", provider=agent.provider, model=agent.model)
    expected += t("gateway.compress.hygiene_timeout_doctor_hint")
    assert warnings == [expected, expected]
    assert all("no-such-provider" not in w and "review-model" not in w for w in warnings)
    assert len(emitted) == 1 and emitted[0] == warnings[0]  # once per agent on the user rail


def test_routing_fallback_never_sends_exception_contents_to_log_or_ui(caplog):
    import logging

    fake_credential = "synthetic-review-credential-not-real"
    fake_url = "https://synthetic-user:synthetic-password@example.invalid/v1?token=synthetic-token"
    error = RuntimeError(f"routing rejected {fake_credential} at {fake_url}\nprivate second line")
    agent = _FakeAgent()
    emitted = []
    agent._emit_warning = emitted.append
    task = {"provider": "review-provider", "model": "review-model"}
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=error), \
         patch.object(br.logger, "warning", wraps=br.logger.warning) as log_warning:
        with caplog.at_level(logging.WARNING, logger="agent.background_review"):
            first = br._resolve_review_runtime(agent, task)
            second = br._resolve_review_runtime(agent, task)

    assert first == second and first["routed"] is False
    assert first["provider"] == agent.provider and first["model"] == agent.model
    warnings = [record.getMessage() for record in caplog.records if record.name == "agent.background_review"]
    assert len(warnings) == log_warning.call_count == 2
    assert len(emitted) == 1 and emitted[0] == warnings[0]
    # Inspect raw logging arguments too: safety must precede any formatter redaction.
    all_sinks = "\n".join([*warnings, *emitted, str(log_warning.call_args_list)])
    for raw in (fake_credential, fake_url, "synthetic-user", "synthetic-password", "synthetic-token", "private second line"):
        assert raw not in all_sinks
    assert all("RuntimeError" in warning and "hermes doctor" in warning for warning in warnings)
    assert all(f"{agent.provider}/{agent.model}" in warning for warning in warnings)


def test_routing_fallback_does_not_stringify_exception():
    class SensitiveError(Exception):
        def __str__(self):
            raise AssertionError("exception details must not be read")

    agent = _FakeAgent()
    emitted = []
    agent._emit_warning = emitted.append
    with patch.object(br.logger, "warning") as log_warning:
        br._warn_review_routing_fallback(agent, "review-provider", "review-model", SensitiveError())
    assert log_warning.call_count == 1
    assert len(emitted) == 1 and "SensitiveError" in emitted[0]


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_routing_fallback_localizes_safe_diagnostic(lang, caplog, monkeypatch, tmp_path):
    import logging

    # Exercise real bundled translations without reading a user's profile or overlays.
    monkeypatch.setattr(i18n, "_current_home", lambda: str(tmp_path))
    monkeypatch.setattr(i18n, "_resolve_language", lambda home: lang)
    i18n.reset_language_cache()
    credential = "synthetic-review-credential-not-real"
    credential_url = "https://synthetic-user:synthetic-password@example.invalid/v1?token=synthetic-token"
    error = RuntimeError(f"routing rejected {credential} at {credential_url}\nprivate second line")
    agent = _FakeAgent()
    emitted = []
    agent._emit_warning = emitted.append
    task = {"provider": credential, "model": credential_url}
    try:
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=error), \
             patch.object(br.logger, "warning", wraps=br.logger.warning) as log_warning:
            with caplog.at_level(logging.WARNING, logger="agent.background_review"):
                first = br._resolve_review_runtime(agent, task)
                second = br._resolve_review_runtime(agent, task)

        expected = i18n.t(
            "display.review.routing_fallback_warning", task_provider="…", task_model="…",
            error="RuntimeError", provider=agent.provider, model=agent.model,
        ) + i18n.t("gateway.compress.hygiene_timeout_doctor_hint")
        warnings = [record.getMessage() for record in caplog.records if record.name == "agent.background_review"]
        assert first == second and first["routed"] is False
        assert first["provider"] == agent.provider and first["model"] == agent.model
        assert warnings == [expected, expected] and log_warning.call_count == 2
        assert emitted == [expected]  # one cohesive localized warning, once per agent
        assert expected.count("⚠") == 1 and expected.count("RuntimeError") == 1
        assert "auxiliary.background_review" in expected and "hermes doctor" in expected
        assert f"{agent.provider}/{agent.model}" in expected
        if lang == "ar":
            assert "تعذّر توجيه" in expected and "سيتم الرجوع" in expected and "على المضيف" in expected
        else:
            assert "could not be routed" in expected and "falling back to" in expected
        all_sinks = "\n".join([*warnings, *emitted, str(log_warning.call_args_list)])
        for raw in (credential, credential_url, "synthetic-user", "synthetic-password", "synthetic-token", "private second line"):
            assert raw not in all_sinks
    finally:
        i18n.reset_language_cache()
