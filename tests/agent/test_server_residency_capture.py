"""Tests for swap-proxy residency capture.

_capture_server_residency asks the resolved provider profile for
``resident_models`` and records the tuple on the agent. Profiles without
the method (every non-llamacpp provider), bare llama-server endpoints
(the plugin returns None), and probe failures all record None, so the
residency indicator stays hidden and degrades with no error.
"""

from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


def _agent():
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "llamacpp"
    agent.requested_provider = "llamacpp"
    agent.base_url = "http://192.168.77.10:8080/v1"
    return agent


class TestCaptureServerResidency:
    def test_profile_with_resident_models_records_tuple(self):
        agent = _agent()
        profile = SimpleNamespace(
            resident_models=lambda *, base_url=None: ("qwen-small",)
        )
        with patch("providers.resolve_provider_profile", return_value=profile):
            agent._capture_server_residency()
        assert agent.last_server_residency == ("qwen-small",)

    def test_profile_without_method_records_none(self):
        agent = _agent()
        with patch(
            "providers.resolve_provider_profile", return_value=SimpleNamespace()
        ):
            agent._capture_server_residency()
        assert agent.last_server_residency is None

    def test_bare_server_none_stays_none(self):
        agent = _agent()
        profile = SimpleNamespace(resident_models=lambda *, base_url=None: None)
        with patch("providers.resolve_provider_profile", return_value=profile):
            agent._capture_server_residency()
        assert agent.last_server_residency is None

    def test_probe_error_degrades_to_none(self):
        agent = _agent()

        def _boom(*, base_url=None):
            raise OSError("connection refused")

        profile = SimpleNamespace(resident_models=_boom)
        with patch("providers.resolve_provider_profile", return_value=profile):
            agent._capture_server_residency()
        assert agent.last_server_residency is None

    def test_stale_value_cleared_when_provider_stops_reporting(self):
        agent = _agent()
        agent.last_server_residency = ("old-model",)
        with patch(
            "providers.resolve_provider_profile", return_value=SimpleNamespace()
        ):
            agent._capture_server_residency()
        assert agent.last_server_residency is None
