"""Nous model healing and credential-refresh rungs for auxiliary calls.

Dependencies are read from the auxiliary router at call time so its existing
resolver and diagnostic seams remain the bindings production uses.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.auxiliary_client import _LadderRoute, _LadderStep


def _refreshed_nous_step(
    route: _LadderRoute, kwargs: dict[str, Any], message: str, *, allow_fallback: bool = True,
) -> Optional[_LadderStep]:
    """Rebuild the Nous client after a credential event; None when nothing refreshed."""
    from agent.auxiliary_client import _LadderStep, _refresh_nous_auxiliary_client, logger

    refreshed_client, refreshed_model = _refresh_nous_auxiliary_client(
        cache_provider=route.resolved_provider or "nous", model=route.final_model,
        lookup_model=route.resolved_model, lookup_task=route.task, async_mode=route.async_mode,
        base_url=route.resolved_base_url, api_key=route.resolved_api_key,
        api_mode=route.resolved_api_mode, main_runtime=route.main_runtime,
        is_vision=(route.task == "vision"),
        **({"allow_fallback": False} if not allow_fallback else {}),
    )
    if refreshed_client is None:
        return None
    logger.info(message, route.task or "call", route.tag)
    if allow_fallback and refreshed_model and refreshed_model != kwargs.get("model"):
        kwargs["model"] = refreshed_model
    return _LadderStep("call", (refreshed_client, kwargs))


def _ladder_nous_rungs(
    first_err: Exception, route: _LadderRoute, kwargs: dict[str, Any], client_is_nous: bool,
    allow_fallback: bool = True,
):
    """Nous-only rungs: stale-model self-heal, paid-account refresh, 401 refresh.
    Returns ``(response, None)`` or ``(None, first_err)`` to fall through."""
    from agent.auxiliary_client import (
        _LadderStep, _credential_rung_accepts, _is_auth_error, _is_connection_error,
        _is_model_not_found_error, _is_payment_error, _nous_portal_account_has_fresh_paid_access,
        _refresh_nous_recommended_model, _rung, logger,
    )

    client, task, tag = route.client, route.task, route.tag
    if not allow_fallback and client_is_nous and _is_model_not_found_error(first_err):
        logger.warning(
            "Auxiliary %s%s: selected model %r was not found; allow_fallback=False "
            "prevents Nous catalog healing. Select an available model or set "
            "allow_fallback=True to permit a replacement.",
            task or "call", tag, kwargs.get("model"),
        )
    # A long-lived process can pin a Portal model since dropped from the catalog (every call
    # 404s); force a fresh Portal fetch and retry once.
    if allow_fallback and _is_model_not_found_error(first_err) and client_is_nous:
        healed_model = _refresh_nous_recommended_model(
            vision=(task == "vision"), stale_model=kwargs.get("model"))
        if healed_model and healed_model != kwargs.get("model"):
            logger.warning("Auxiliary %s%s: model %r no longer in Nous catalog; "
                           "retrying with refreshed recommendation %r",
                           task or "call", tag, kwargs.get("model"), healed_model)
            kwargs["model"] = healed_model
            resp, first_err = yield from _rung(_LadderStep("call", (client, kwargs)), lambda exc: True)
            if first_err is None:
                return resp, None
    # Auth refresh parity with the main agent.
    if _is_payment_error(first_err) and client_is_nous and _nous_portal_account_has_fresh_paid_access():
        step = _refreshed_nous_step(
            route, kwargs,
            "Auxiliary %s%s: refreshed Nous runtime credentials after paid account check, retrying",
            allow_fallback=allow_fallback)
        if step is not None:
            resp, first_err = yield from _rung(
                step, lambda exc: _credential_rung_accepts(exc) or _is_connection_error(exc))
            if first_err is None:
                return resp, None
    if _is_auth_error(first_err) and client_is_nous:
        step = _refreshed_nous_step(
            route, kwargs, "Auxiliary %s%s: refreshed Nous runtime credentials after 401, retrying",
            allow_fallback=allow_fallback)
        if step is not None:
            resp, first_err = yield from _rung(
                step, lambda exc: _credential_rung_accepts(exc) or _is_connection_error(exc))
            if first_err is None:
                return resp, None
    return None, first_err
