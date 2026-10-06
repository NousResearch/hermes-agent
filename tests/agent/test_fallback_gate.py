"""``pre_fallback_activate``: plugins can veto automatic provider fallback.

Pins the contract of agent/fallback_gate.py and its call sites:
  1. no hook / non-block results / a raising dispatch -> fallback allowed (fail-open);
  2. a ``{"action": "block"}`` result stops the mid-turn switch, keeps the primary runtime and
     ends fallback for the rest of the turn (no re-ask, no walk past the veto);
  3. the hook receives the turn context (stage, session, from/to route, reason);
  4. resolution-time fallback (gated_fallback_entries, e.g. resolve_runtime_with_fallback) honours the
     veto by ending the walk, so the primary's error surfaces instead of a fallback entry resolving.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from agent.fallback_gate import PRE_FALLBACK_ACTIVATE_HOOK, fallback_veto
from run_agent import AIAgent


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
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://api.openai.com/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


def _install_hook(monkeypatch, results=None, *, raises=None):
    calls = []

    def _invoke(name, **kwargs):
        calls.append((name, kwargs))
        if raises is not None:
            raise raises
        return list(results or [])

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == PRE_FALLBACK_ACTIVATE_HOOK)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return calls


def test_hook_is_registered_as_valid():
    from hermes_cli.plugins import VALID_HOOKS
    assert PRE_FALLBACK_ACTIVATE_HOOK in VALID_HOOKS


class TestFallbackVeto:
    def test_no_hook_allows(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
        assert fallback_veto("turn", to_provider="openai", to_model="gpt-4o") is None

    def test_non_block_results_allow(self, monkeypatch):
        _install_hook(monkeypatch, [None, "allow", {"action": "allow"}, {"message": "x"}])
        assert fallback_veto("turn", to_provider="openai", to_model="gpt-4o") is None

    def test_first_block_wins(self, monkeypatch):
        _install_hook(monkeypatch, [{"action": "allow"}, {"action": "BLOCK", "message": "not approved"},
                                    {"action": "block", "message": "second"}])
        assert fallback_veto("turn") == "not approved"

    def test_block_without_message_has_default_text(self, monkeypatch):
        _install_hook(monkeypatch, [{"action": "block"}])
        assert fallback_veto("startup") == "fallback blocked by a plugin policy"

    def test_dispatch_error_fails_open(self, monkeypatch):
        _install_hook(monkeypatch, raises=RuntimeError("plugin exploded"))
        assert fallback_veto("turn") is None

    def test_message_is_capped(self, monkeypatch):
        _install_hook(monkeypatch, [{"action": "block", "message": "x" * 5000}])
        assert len(fallback_veto("turn")) == 500


class TestTurnFallbackVeto:
    CHAIN = [{"provider": "openai", "model": "gpt-4o"}, {"provider": "deepseek", "model": "deepseek-chat"}]

    def test_veto_keeps_primary_and_ends_turn_fallback(self, monkeypatch):
        calls = _install_hook(monkeypatch, [{"action": "block", "message": "user said no"}])
        agent = _make_agent(fallback_model=self.CHAIN)
        agent.session_id = "sess-1"
        primary = (agent.model, agent.provider, agent.base_url)
        with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False
            resolve.assert_not_called()
        assert (agent.model, agent.provider, agent.base_url) == primary
        assert agent._fallback_activated is False
        assert agent._fallback_index == len(self.CHAIN)
        assert agent._has_pending_fallback() is False
        # A second recovery path in the same turn does not ask again.
        with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False
            resolve.assert_not_called()
        assert len(calls) == 1
        name, kwargs = calls[0]
        assert name == PRE_FALLBACK_ACTIVATE_HOOK
        assert kwargs["stage"] == "turn"
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["to_provider"] == "openai" and kwargs["to_model"] == "gpt-4o"
        assert kwargs["from_model"] == primary[0]
        assert kwargs["reason"]

    def test_next_turn_asks_again(self, monkeypatch):
        calls = _install_hook(monkeypatch, [{"action": "block", "message": "no"}])
        agent = _make_agent(fallback_model=self.CHAIN)
        assert agent._try_activate_fallback(reason=FailoverReason.overloaded) is False
        agent._restore_primary_runtime()
        assert agent._fallback_index == 0
        assert agent._try_activate_fallback(reason=FailoverReason.overloaded) is False
        assert len(calls) == 2

    def test_allowed_switch_proceeds(self, monkeypatch):
        calls = _install_hook(monkeypatch, [None])
        agent = _make_agent(fallback_model=self.CHAIN)
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        assert agent.model == "gpt-4o"
        assert agent._fallback_index == 1
        assert len(calls) == 1


class TestStartupFallbackVeto:
    def test_resolve_runtime_with_fallback_reraises_primary_on_veto(self, monkeypatch):
        from hermes_cli.auth import AuthError
        from hermes_cli import runtime_provider as rp

        calls = _install_hook(monkeypatch, [{"action": "block", "message": "not approved"}])
        primary_exc = AuthError("primary key expired")
        resolved = []

        def _resolve(**kwargs):
            resolved.append(kwargs.get("requested"))
            if kwargs.get("requested") == "anthropic":
                raise primary_exc
            return {"provider": kwargs.get("requested"), "api_key": "k", "base_url": "https://x"}

        monkeypatch.setattr(rp, "resolve_runtime_provider", _resolve)
        config = {"fallback_providers": [{"provider": "openai", "model": "gpt-4o"}]}
        with pytest.raises(AuthError):
            rp.resolve_runtime_with_fallback(config, requested="anthropic", target_model="claude")
        assert resolved == ["anthropic"]
        assert calls and calls[0][1]["stage"] == "startup"
        assert calls[0][1]["to_model"] == "gpt-4o"

    def test_resolve_runtime_with_fallback_allowed(self, monkeypatch):
        from hermes_cli.auth import AuthError
        from hermes_cli import runtime_provider as rp

        _install_hook(monkeypatch, [])

        def _resolve(**kwargs):
            if kwargs.get("requested") == "anthropic":
                raise AuthError("primary key expired")
            return {"provider": kwargs.get("requested"), "api_key": "k", "base_url": "https://x"}

        monkeypatch.setattr(rp, "resolve_runtime_provider", _resolve)
        config = {"fallback_providers": [{"provider": "openai", "model": "gpt-4o"}]}
        runtime, entry = rp.resolve_runtime_with_fallback(config, requested="anthropic", target_model="claude")
        assert entry and entry["model"] == "gpt-4o"


class TestGatedFallbackEntries:
    def test_veto_stops_the_walk_and_reports(self, monkeypatch):
        from hermes_cli.fallback_config import gated_fallback_entries

        calls = _install_hook(monkeypatch, [{"action": "block", "message": "ask first"}])
        vetoed = []
        chain = [{"provider": "openai"}, "junk", {"provider": "openai", "model": "gpt-4o"},
                 {"provider": "deepseek", "model": "deepseek-chat"}]
        walked = list(gated_fallback_entries(chain, None, "anthropic", "claude", platform="cron", job_id="j1",
                                             on_veto=lambda p, m, msg: vetoed.append((p, m, msg))))
        # Malformed entries pass through for the walker's own checks; the first usable one is vetoed.
        assert walked == [{"provider": "openai"}, "junk"]
        assert vetoed == [("openai", "gpt-4o", "ask first")]
        assert len(calls) == 1
        kwargs = calls[0][1]
        assert (kwargs["stage"], kwargs["platform"], kwargs["job_id"]) == ("startup", "cron", "j1")
        assert (kwargs["from_provider"], kwargs["from_model"]) == ("anthropic", "claude")

    def test_no_veto_yields_everything(self, monkeypatch):
        from hermes_cli.fallback_config import gated_fallback_entries

        _install_hook(monkeypatch, [None])
        chain = [{"provider": "openai", "model": "gpt-4o"}, {"provider": "deepseek", "model": "deepseek-chat"}]
        assert list(gated_fallback_entries(chain)) == chain
        assert list(gated_fallback_entries(None)) == []
