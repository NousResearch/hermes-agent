"""Compression route attribution regressions for #72636/#72637."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest



def test_real_call_llm_reports_concrete_auto_route_and_strips_query() -> None:
    """The real call_llm orchestration must resolve ``auto`` to the client
    backend and sanitize its endpoint before publishing the wire attempt."""
    from agent.auxiliary_client import call_llm

    secret = "sk-route-secret"
    client = MagicMock()
    client.base_url = f"https://api.minimax.chat/v1?key={secret}"
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    routes: list[tuple[str, str | None, str]] = []

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "minimax/minimax-m2.7"),
        ),
    ):
        call_llm(
            task="compression",
            messages=[{"role": "user", "content": "summarize"}],
            route_callback=lambda *route: routes.append(route),
        )

    assert routes == [(
        "minimax",
        "minimax/minimax-m2.7",
        "https://api.minimax.chat/v1",
    )]
    assert secret not in repr(routes)


def test_real_call_llm_updates_route_when_fallback_fails() -> None:
    """If a fallback becomes the terminal failing request, the last observed
    identity must describe that fallback rather than the stale primary."""
    from agent.auxiliary_client import call_llm

    auth_error = type("Auth401", (Exception,), {"status_code": 401})("expired key")
    primary = MagicMock()
    primary.base_url = "https://api.minimax.chat/v1"
    primary.chat.completions.create.side_effect = auth_error

    fallback = MagicMock()
    fallback.base_url = "https://openrouter.ai/api/v1"
    fallback.chat.completions.create.side_effect = ValueError(
        "fallback malformed response"
    )
    routes: list[tuple[str, str | None, str]] = []

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "minimax/minimax-m2.7"),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            return_value=(fallback, "fallback-model", "openrouter"),
        ),
    ):
        with pytest.raises(ValueError, match="fallback malformed response"):
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                route_callback=lambda *route: routes.append(route),
            )

    assert routes == [
        ("minimax", "minimax/minimax-m2.7", "https://api.minimax.chat/v1"),
        ("openrouter", "fallback-model", "https://openrouter.ai/api/v1"),
    ]


def test_real_call_llm_quarantined_fallback_auth_propagates_last_attempt_error() -> None:
    """Primary rate-limit → fallback candidate with an unrefreshable 401 →
    no remaining candidate. The error that propagates must be the FALLBACK's
    terminal auth error, not the primary's rate-limit: the route snapshot
    already identifies the fallback, so re-raising the primary would pair one
    attempt's endpoint with a different attempt's failure class (#72636
    review, defect 1)."""
    from agent.auxiliary_client import call_llm

    primary_err = type("Rate429", (Exception,), {"status_code": 429})(
        "429 Too Many Requests: rate limit exceeded"
    )
    fallback_err = type("Auth401", (Exception,), {"status_code": 401})(
        "expired fallback key"
    )
    primary = MagicMock()
    primary.base_url = "https://api.minimax.chat/v1"
    primary.chat.completions.create.side_effect = primary_err

    fallback = MagicMock()
    fallback.base_url = "https://openrouter.ai/api/v1"
    fallback.chat.completions.create.side_effect = fallback_err
    routes: list[tuple[str, str | None, str]] = []

    # Discovery order for an auto primary: configured chain (None) → main
    # chain (None) → payment fallback (the 401-looping candidate). After the
    # candidate is quarantined, the chain is walked once more — this time it
    # must yield nothing, so the chain exhausts.
    payment_walks = [
        (fallback, "fallback-model", "openrouter"),
        (None, None, ""),
    ]

    def _payment_fallback(*_args, **_kwargs):
        return payment_walks.pop(0) if payment_walks else (None, None, "")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "minimax/minimax-m2.7"),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=_payment_fallback,
        ),
    ):
        with pytest.raises(Exception) as excinfo:
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                route_callback=lambda *route: routes.append(route),
            )

    # The terminal route snapshot is the fallback...
    assert routes == [
        ("minimax", "minimax/minimax-m2.7", "https://api.minimax.chat/v1"),
        ("openrouter", "fallback-model", "https://openrouter.ai/api/v1"),
    ]
    # ...so the propagated error must be the fallback's own 401, chained to
    # the primary origin — never the primary's 429 re-raised bare.
    assert excinfo.value is fallback_err, (
        f"propagated error is not the quarantined fallback 401: {excinfo.value!r}"
    )
    assert excinfo.value.__cause__ is primary_err


def test_real_call_llm_quarantined_fallback_keeps_primary_error_for_other_tasks() -> None:
    """Same wire scenario as the compression test above, but for an auxiliary
    task that did NOT opt into route attribution (no route_callback, task is
    not compression — e.g. web_extract or title generation): call_llm's
    exception contract must be unchanged from before this PR. The swallowed
    fallback 401 stays swallowed and the caller observes the PRIMARY error
    (429), which upper layers key retry/backoff decisions on (#72636
    review, defect 1 scoping)."""
    from agent.auxiliary_client import call_llm

    primary_err = type("Rate429", (Exception,), {"status_code": 429})(
        "429 Too Many Requests: rate limit exceeded"
    )
    fallback_err = type("Auth401", (Exception,), {"status_code": 401})(
        "expired fallback key"
    )
    primary = MagicMock()
    primary.base_url = "https://api.minimax.chat/v1"
    primary.chat.completions.create.side_effect = primary_err

    fallback = MagicMock()
    fallback.base_url = "https://openrouter.ai/api/v1"
    fallback.chat.completions.create.side_effect = fallback_err

    payment_walks = [
        (fallback, "fallback-model", "openrouter"),
        (None, None, ""),
    ]

    def _payment_fallback(*_args, **_kwargs):
        return payment_walks.pop(0) if payment_walks else (None, None, "")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "minimax/minimax-m2.7", None, None, None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "minimax/minimax-m2.7"),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=_payment_fallback,
        ),
    ):
        with pytest.raises(Exception) as excinfo:
            call_llm(
                task="web_extract",
                messages=[{"role": "user", "content": "describe"}],
            )

    # Pre-PR contract: the caller sees the primary 429, not the fallback's
    # quarantined 401 — retry/backoff classification must not flip to
    # "credential broken" for tasks outside the attribution path.
    assert excinfo.value is primary_err, (
        f"non-compression task received the quarantined fallback error "
        f"instead of the primary: {excinfo.value!r}"
    )
