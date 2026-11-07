"""Every path that moves ``base_url`` to another endpoint starts that endpoint without the old one's
rejection verdicts; a refresh that keeps the endpoint keeps them (agent/route_verdicts.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.route_verdicts import ROUTE_REJECTION_FIELDS
from tests.agent.test_primary_runtime_restore import _make_agent, _make_transport_error

_PAID = "https://inference-api.nousresearch.com/v1"
_WELCOME = "https://welcome-api.nousresearch.com/v1"


def _verdicts(agent):
    return {name: bool(getattr(agent, name, False)) for name in ROUTE_REJECTION_FIELDS}


def _tripped(agent):
    for name in ROUTE_REJECTION_FIELDS:
        setattr(agent, name, True)
    return agent


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
@pytest.mark.parametrize(("new_url", "kept"), [(_WELCOME, False), (_PAID + "/", True)])
def test_nous_refresh_onto_another_host_drops_the_old_hosts_verdicts(api_mode, new_url, kept):
    """The welcome-tier heal moves the session between the welcome and paid hosts: a verdict one
    host earned must not follow it. A refreshed key on the same host keeps them."""
    agent = _tripped(_make_agent(provider="nous", base_url=_PAID))
    agent.api_mode = api_mode
    agent._rebuild_anthropic_client = MagicMock()
    with patch("hermes_cli.auth.resolve_nous_runtime_credentials",
               return_value={"api_key": "fresh-agent-key", "base_url": new_url}), \
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_refresh_nous_client_credentials(force=True) is True
    assert agent.base_url == new_url.rstrip("/")
    assert _verdicts(agent) == dict.fromkeys(ROUTE_REJECTION_FIELDS, kept)


def test_provider_refresh_hook_returning_another_base_url_drops_the_old_verdicts():
    agent = _tripped(_make_agent())
    profile = SimpleNamespace(name="p", refresh_runtime_credentials=lambda **_kw: ("tok", "https://moved.example/v1"))
    with patch("agent.conversation_loop._credential_refresh_profile", return_value=profile), \
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent.base_url == "https://moved.example/v1"
    assert not any(_verdicts(agent).values())


def test_transport_recovery_back_to_the_snapshot_url_drops_the_rotated_urls_verdicts():
    """A pool rotation moved the primary to U2 and U2 learned a verdict; transport recovery puts
    the agent back on the snapshot URL, which never earned it."""
    agent = _make_agent()
    snapshot_url = agent._primary_runtime["base_url"]
    agent.base_url = "https://rotated.example/v1"
    agent._stream_options_unsupported = True
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()), patch("time.sleep"):
        assert agent._try_recover_primary_transport(_make_transport_error(), retry_count=3, max_retries=3) is True
    assert agent.base_url == snapshot_url
    assert agent._stream_options_unsupported is False


def test_transport_recovery_on_the_same_url_keeps_the_verdicts():
    agent = _tripped(_make_agent())
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()), patch("time.sleep"):
        assert agent._try_recover_primary_transport(_make_transport_error(), retry_count=3, max_retries=3) is True
    assert all(_verdicts(agent).values())
