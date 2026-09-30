"""Stage-1 (requested route) model/provider resolution, shared by the CLI and the Kanban dispatcher.

This module owns the pure core of ``HermesCLI._init_model_and_provider``: given argv-level
``model`` / ``provider`` plus the already-loaded CLI config, it computes the requested route
(the model string, the requested provider label, and any URL-bearing alias's explicit
``base_url`` / ``api_key``) with NO side effects:

- no network I/O (the localhost model auto-detect stays in the CLI tail),
- no credential reads (``env_provider`` is passed in, never read from the environment),
- no ``self`` mutation (the CLI method assigns the result onto ``self``).

The Kanban per-provider concurrency budget (#123654) resolves each run's budget key through
this same function so dispatcher-side keys match what the worker will request at startup,
without ever running stage-2 runtime resolution (credential pools, OAuth refresh, HTTP
auto-detect) under the dispatch lock.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

from hermes_cli.config import split_model_config_default


def normalize_moa_model(model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """``moa:<preset>`` -> ``("moa", preset)`` (same routing as ``/moa``); anything
    else -> ``(None, model)``.

    The ONE parser for the MoA virtual-provider prefix (#56828): stage-1
    route resolution folds it here, and ``cli._normalize_moa_model`` delegates
    to this function so the prefix rules can never diverge between interactive
    and non-interactive paths.
    """
    if isinstance(model, str) and model.strip().lower().startswith("moa:"):
        preset = model.strip().split(":", 1)[1].strip()
        if preset:
            return "moa", preset
    return None, model


class RequestedRoute(NamedTuple):
    """Stage-1 resolution result: what the worker will request at startup.

    ``model`` is the post-alias, post-MoA model string (possibly empty when the CLI
    auto-detect tail will fill it); ``requested_provider`` the provider label the runtime
    resolver receives, folding the whole ladder (MoA prefix > explicit ``--provider`` >
    startup alias route > nested ``model.default`` provider > ``model.provider`` >
    ``HERMES_INFERENCE_PROVIDER`` > ``"auto"``); ``explicit_base_url`` /
    ``explicit_api_key`` carry a URL-bearing direct alias's endpoint and credential
    (None = not pinned by the route). ``config_model`` / ``nested_provider`` expose the
    split ``model.default`` pieces for CLI consumers (``_model_is_default``).
    """

    model: str
    requested_provider: str
    explicit_base_url: Optional[str] = None
    explicit_api_key: Optional[str] = None
    config_model: str = ""
    nested_provider: str = ""


def resolve_requested_route(
    *,
    model: Optional[str],
    provider: Optional[str],
    model_config: Dict[str, Any],
    user_providers: Optional[dict] = None,
    custom_providers: Optional[list] = None,
    env_provider: Optional[str] = None,
) -> RequestedRoute:
    """Resolve the startup-requested model/provider route (pure stage 1).

    Mirrors the body of ``HermesCLI._init_model_and_provider`` (the precedence ladder:
    MoA prefix > explicit ``--provider`` > startup alias route > nested ``model.default``
    provider > ``model.provider`` > ``HERMES_INFERENCE_PROVIDER`` > ``"auto"``) minus the two
    impure tails the CLI keeps for itself: the localhost model auto-detect and the
    ``--provider <named custom>`` default-model lookup (both change only the MODEL string,
    never the provider; the named-custom tail is re-derived by the dispatcher's key step from
    the same config it reads here).

    ``model_config`` is the caller's ``CLI_CONFIG["model"]`` (or a profile's ``model``
    section); ``env_provider`` is the caller's ``HERMES_INFERENCE_PROVIDER`` value (the
    dispatcher passes the assignee profile's scoped read, the CLI its own os.getenv).
    """
    config_model, nested_provider = split_model_config_default(
        model_config.get("default") or model_config.get("model") or ""
    )
    resolved_model = model or config_model or ""
    cfg_provider = model_config.get("provider") or env_provider
    startup_provider_override = startup_base_url_override = startup_api_key_override = ""
    if resolved_model:
        # Late import: model_switch pulls the providers registry; keep this module's import
        # graph light for dispatcher-side use.
        from hermes_cli.model_switch import resolve_startup_model_route

        startup_route = resolve_startup_model_route(
            resolved_model,
            explicit_provider=provider or "",
            current_provider=(provider or nested_provider or cfg_provider or ""),
            user_providers=user_providers,
            custom_providers=custom_providers,
        )
        if startup_route is not None:
            resolved_model = startup_route.model
            startup_provider_override = startup_route.provider
            startup_base_url_override = startup_route.base_url
            startup_api_key_override = startup_route.api_key
    # ``moa:<preset>`` selects the MoA virtual provider in one shot (parity with the
    # interactive /moa command and the model picker, #56828): the prefix wins over --provider.
    # One parser, owned here and delegated to by ``cli._normalize_moa_model`` (#123654 R1 A5).
    moa_provider_override, _moa_model = normalize_moa_model(resolved_model)
    resolved_model = _moa_model or ""
    requested_provider = (
        moa_provider_override or provider or startup_provider_override or nested_provider
        or cfg_provider or "auto"
    )
    return RequestedRoute(
        model=resolved_model,
        requested_provider=requested_provider,
        explicit_base_url=startup_base_url_override or None,
        explicit_api_key=startup_api_key_override or None,
        config_model=config_model,
        nested_provider=nested_provider,
    )
