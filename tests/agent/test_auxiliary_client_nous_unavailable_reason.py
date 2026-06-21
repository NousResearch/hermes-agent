"""Regression tests for actionable auxiliary-client auth diagnostics.

The goal loop used to show only "no auxiliary client configured" when
``get_text_auxiliary_client("goal_judge")`` returned ``(None, None)``. For
Nous-backed auxiliary tasks that can happen when the stored OAuth refresh token
is invalid/expired: the resolver swallowed the real AuthError at debug level,
so operators were sent chasing config/model issues instead of re-authing.
"""
from __future__ import annotations

import logging


def test_nous_runtime_auth_failure_is_recorded_and_described(monkeypatch):
    from agent import auxiliary_client as aux
    from hermes_cli.auth import AuthError

    # Keep this test hermetic and deterministic.
    aux._clear_nous_runtime_failure()
    aux._WARNED_NOUS_RUNTIME_FAILURES.clear()
    monkeypatch.setattr(
        aux,
        "_resolve_task_provider_model",
        lambda task=None, provider=None, model=None, base_url=None, api_key=None: (
            "nous",
            "deepseek/deepseek-v4-pro",
            "",
            "",
            "",
        ),
    )
    monkeypatch.setattr(aux, "_read_nous_auth", lambda: {"access_token": "stale"})

    def _raise_invalid_grant(*args, **kwargs):
        raise AuthError(
            "Invalid refresh token",
            provider="nous",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        _raise_invalid_grant,
    )

    assert aux._resolve_nous_runtime_api(force_refresh=False) is None

    reason = aux.describe_auxiliary_client_unavailable("goal_judge")
    assert "goal_judge auxiliary client unavailable" in reason
    assert "Nous Portal runtime credentials unavailable" in reason
    assert "Invalid refresh token" in reason
    assert "invalid_grant" in reason
    assert "hermes model" in reason
    assert reason != "no auxiliary client configured"


def test_try_nous_warns_once_with_recorded_runtime_failure(monkeypatch, caplog):
    from agent import auxiliary_client as aux
    from hermes_cli.auth import AuthError

    aux._clear_nous_runtime_failure()
    aux._WARNED_NOUS_RUNTIME_FAILURES.clear()
    monkeypatch.setattr(aux, "_read_nous_auth", lambda: {"access_token": "stale"})
    monkeypatch.setattr(aux, "_nous_api_key", lambda provider: "")
    monkeypatch.setattr(
        "hermes_cli.models.get_nous_recommended_aux_model",
        lambda vision=False: "qwen/qwen3.6-flash",
    )

    def _raise_invalid_grant(*args, **kwargs):
        raise AuthError(
            "Invalid refresh token",
            provider="nous",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_runtime_credentials",
        _raise_invalid_grant,
    )

    caplog.set_level(logging.WARNING, logger="agent.auxiliary_client")
    assert aux._try_nous() == (None, None)
    assert aux._try_nous() == (None, None)

    messages = [record.getMessage() for record in caplog.records]
    runtime_messages = [
        msg for msg in messages
        if "runtime credential resolution failed" in msg
    ]
    assert len(runtime_messages) == 1
    assert "Invalid refresh token" in runtime_messages[0]
    assert "invalid_grant" in runtime_messages[0]
