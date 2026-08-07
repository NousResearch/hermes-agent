"""The aux-provider compression warning must log WHY it fired.

``check_compression_model_feasibility`` computes a specific, actionable
message - either "configured provider 'X' is unavailable, reauthenticate" or
"no provider configured, run hermes setup" - stores it on the agent, emits it
to the user, and then logs a GENERIC line that carries neither.

That branch is the one where compression "will drop middle turns without a
summary", i.e. silent context loss. When it fires on a long-running gateway
the operator has only the log, and the log cannot distinguish a transient
auth failure on a configured provider (reauthenticate) from an unconfigured
install (run setup). Observed 21 times on a live install that HAD a provider
configured, with no way to tell which case it was.
"""

from types import SimpleNamespace

import pytest

from agent import conversation_compression as cc


@pytest.fixture()
def agent():
    return SimpleNamespace(
        compression_enabled=True,
        _compression_warning=None,
        _emitted=[],
        _emit_status=lambda msg: None,
        _current_main_runtime=lambda: None,
    )


def _force_unavailable_aux(monkeypatch, configured_provider: str):
    """Make aux-client resolution fail so the warning branch runs.

    Patches the ``agent.auxiliary_client`` seam the function imports locally.
    """
    import agent.auxiliary_client as aux

    monkeypatch.setattr(
        aux, "get_text_auxiliary_client", lambda *a, **k: (None, ""), raising=False
    )
    monkeypatch.setattr(
        aux,
        "_try_configured_fallback_for_unavailable_client",
        lambda *a, **k: (None, "", ""),
        raising=False,
    )
    monkeypatch.setattr(
        aux,
        "_resolve_task_provider_model",
        lambda task: (configured_provider, None, None, None, None),
        raising=False,
    )


class TestAuxWarningCarriesTheReason:
    def test_configured_but_unavailable_provider_is_named_in_the_log(
        self, agent, monkeypatch, caplog
    ):
        _force_unavailable_aux(monkeypatch, "xai-oauth")

        with caplog.at_level("WARNING"):
            cc.check_compression_model_feasibility(agent)

        assert agent._compression_warning, "warning should be set on the agent"
        logged = " ".join(r.getMessage() for r in caplog.records)
        # The operator must be able to tell WHICH failure this was from the log
        # alone: a configured-but-unavailable provider needs reauthentication,
        # not `hermes setup`.
        assert "xai-oauth" in logged

    def test_unconfigured_case_logs_the_setup_remedy(self, agent, monkeypatch, caplog):
        _force_unavailable_aux(monkeypatch, "")

        with caplog.at_level("WARNING"):
            cc.check_compression_model_feasibility(agent)

        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "hermes setup" in logged or "OPENROUTER_API_KEY" in logged

    def test_logged_text_matches_the_user_facing_warning(
        self, agent, monkeypatch, caplog
    ):
        """One message, two sinks. Divergence is how the log went generic."""
        _force_unavailable_aux(monkeypatch, "xai-oauth")

        with caplog.at_level("WARNING"):
            cc.check_compression_model_feasibility(agent)

        logged = " ".join(r.getMessage() for r in caplog.records)
        stored = (agent._compression_warning or "").lstrip("⚠").strip()
        assert stored and stored in logged
