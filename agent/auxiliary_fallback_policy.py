"""Fallback-policy helpers for the auxiliary client router.

Extracted verbatim from ``agent/auxiliary_client.py`` (R3-S1 window,
lines 4891-5369, epic #78647).  The monolith re-exports these names at
the old location; every call-time edge back into ``agent.auxiliary_client``
(including co-extracted sibling defs) is a function-local lazy import so
test patches on the monolith stay effective and the monolith's mid-file
import of this module never observes a partially initialized module.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

logger = logging.getLogger("agent.auxiliary_client")


def _try_payment_fallback(
    failed_provider: str,
    task: str = None,
    reason: str = "payment error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try alternative providers after a payment/credit or connection error.

    Iterates the standard auto-detection chain, skipping the provider that
    failed.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    from agent.auxiliary_client import (
        _get_provider_chain,
        _is_provider_unhealthy,
        _log_skip_unhealthy,
        _read_main_provider,
    )
    # Normalise the failed provider label for matching.
    skip = failed_provider.lower().strip()
    # Also skip Step-1 main-provider path if it maps to the same backend.
    # (e.g. main_provider="openrouter" → skip "openrouter" in chain)
    main_provider = _read_main_provider()
    skip_labels = {skip}
    if main_provider and main_provider.lower() in skip:
        skip_labels.add(main_provider.lower())
    # Map common resolved_provider values back to chain labels.
    _alias_to_label = {"openrouter": "openrouter", "nous": "nous",
                       "openai-codex": "openai-codex", "codex": "openai-codex",
                       "custom": "local/custom", "local/custom": "local/custom"}
    skip_chain_labels = {_alias_to_label.get(s, s) for s in skip_labels}

    tried = []
    for label, try_fn in _get_provider_chain():
        if label in skip_chain_labels:
            continue
        if _is_provider_unhealthy(label):
            _log_skip_unhealthy(label, task)
            tried.append(f"{label} (unhealthy)")
            continue
        client, model = try_fn()
        if client is not None:
            logger.info(
                "Auxiliary %s: %s on %s — falling back to %s (%s)",
                task or "call", reason, failed_provider, label, model or "default",
            )
            return client, model, label
        tried.append(label)

    logger.warning(
        "Auxiliary %s: %s on %s and no fallback available (tried: %s)",
        task or "call", reason, failed_provider, ", ".join(tried),
    )
    return None, None, ""


def _try_main_agent_model_fallback(
    failed_provider: str,
    task: str = None,
    reason: str = "error",
    failed_model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Last-resort fallback to the user's main agent provider + model.

    Used after the configured fallback_chain is exhausted (or empty) for
    users with an explicit auxiliary provider.  This is the "safety net"
    layer: if nothing the user asked for can serve the request, try the
    main chat model before giving up.

    ``failed_model`` narrows the same-provider skip to the exact
    (provider, model) pair that just failed, mirroring
    :func:`_try_configured_fallback_chain`.  This matters for self-hosted /
    custom endpoints serving several models behind one provider label: the
    aux compression model timing out says nothing about the health of the
    main agent model deployed on the same URL (real incident: aux
    ``glm-5.2`` hung and timed out while main ``macaron-v1-venti`` on the
    identical endpoint was serving 448K-token turns fine — the
    provider-label skip discarded the one fallback that would have worked).

    - Model-specific runtime failures (timeout, connection, rate limit,
      model-incompatible, invalid response) pass ``failed_model``: skip the
      main model only when it IS the exact model that failed.
    - Provider-wide failures (auth 401, payment 402) and legacy callers
      leave ``failed_model`` as None, keeping the whole-provider skip —
      the shared credentials/account are broken, so the main model on the
      same provider cannot help either.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    from agent.auxiliary_client import (
        _is_provider_unhealthy,
        _log_skip_unhealthy,
        _read_main_model,
        _read_main_provider,
        _resolve_moa_aggregator,
        resolve_provider_client,
    )
    main_provider = (_read_main_provider() or "").strip()
    main_model = (_read_main_model() or "").strip()
    if main_provider.lower() == "moa":
        # MoA virtual provider: fall back to the preset's aggregator — the
        # acting model — instead of the unreachable "moa"/<preset-name> pair.
        _agg_provider, _agg_model = _resolve_moa_aggregator(main_model)
        if not _agg_provider or not _agg_model:
            return None, None, ""
        main_provider, main_model = _agg_provider, _agg_model
    if not main_provider or not main_model or main_provider.lower() in {"auto", ""}:
        return None, None, ""

    # Identity + scope semantics owned by agent.backend_identity (#72468):
    # model-scoped failures skip only the exact deployment that failed;
    # provider-wide failures (no failed_model) skip the credential surface.
    from agent.backend_identity import (
        BackendIdentity,
        FailureScope,
        should_skip_candidate,
    )

    skip_model = (failed_model or "").strip().lower() or None
    if should_skip_candidate(
        BackendIdentity.build(provider=main_provider, model=main_model),
        BackendIdentity.build(provider=failed_provider, model=skip_model),
        FailureScope.MODEL if skip_model else FailureScope.CREDENTIAL,
    ):
        # The thing that failed IS the main model (or the failure was
        # provider-wide) — nothing to fall back to.
        return None, None, ""
    if _is_provider_unhealthy(main_provider):
        _log_skip_unhealthy(main_provider, task)
        return None, None, ""

    try:
        client, resolved_model = resolve_provider_client(
            provider=main_provider, model=main_model,
        )
    except Exception:
        client, resolved_model = None, None

    if client is None:
        return None, None, ""

    label = f"main-agent({main_provider})"
    logger.info(
        "Auxiliary %s: %s on %s — falling back to main agent model %s (%s)",
        task or "call", reason, failed_provider, label, resolved_model or main_model,
    )
    return client, resolved_model or main_model, label


# ── Context-window screening for runtime fallback chains (issue #52392) ──
#
# When the runtime auxiliary fallback chain selects a candidate that is
# reachable but has a context window smaller than the compression task
# requires, the call errors out instead of continuing to the next, viable
# candidate. The startup feasibility check in
# ``agent.conversation_compression.check_compression_model_feasibility``
# already filters too-small auxiliary models at startup, but the runtime
# fallback chain (``_try_configured_fallback_chain`` and
# ``_try_main_fallback_chain``) does not apply the same filter, so
# compression can stop at the first alive door even if the room behind it
# is too small.
#
# The helpers below screen each candidate by its effective context window
# before it is returned. ``None`` results from ``get_model_context_length``
# are passed through (we cannot prove a model is too small, so we do not
# block it). This preserves the existing fallback surface for
# unrecognised/custom models while closing the gap on the well-known ones.

def _task_minimum_context_length(task: Optional[str]) -> Optional[int]:
    """Return the minimum context length required for an auxiliary task.

    Only ``compression`` carries an explicit minimum today (the same
    ``MINIMUM_CONTEXT_LENGTH`` (64K) floor that
    ``check_compression_model_feasibility`` already enforces at startup).
    Other tasks (``vision``, ``title_generation``, ``web_extract``,
    ``skills_hub``, ``mcp``, ``session_search``) return ``None`` — they
    have no per-task context floor and the runtime chain must remain
    permissive for them.

    Returns ``None`` for an empty/``None`` task name so the helper is a
    safe no-op when called from generic sites.
    """
    if not task:
        return None
    if task == "compression":
        return MINIMUM_CONTEXT_LENGTH
    return None


def _candidate_context_window(
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
) -> Optional[int]:
    """Resolve the effective context window for a fallback candidate.

    Thin wrapper around :func:`agent.model_metadata.get_model_context_length`
    that swallows probe failures (returns ``None``). Callers treat
    ``None`` as "unknown — pass through" so the existing fallback
    surface is preserved when the context-length resolver chain cannot
    determine a value (custom endpoints, models not in the registry,
    offline endpoints).

    Best-effort, never raises — the runtime fallback chain must keep
    moving even if the resolver hits a probe error.
    """
    from agent.auxiliary_client import get_model_context_length
    if not model:
        return None
    try:
        ctx = get_model_context_length(
            model,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
        )
    except Exception as exc:
        logger.debug(
            "Auxiliary fallback: could not resolve context window for %s/%s: %s",
            provider, model, exc,
        )
        return None
    # ``get_model_context_length`` returns an int (with a 256K default
    # fallback when nothing else matches). We still propagate ``None`` if
    # a future change returns ``Optional[int]`` — being explicit is
    # cheap and the test suite covers both shapes.
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    return None


def _try_configured_fallback_chain(
    task: str,
    failed_provider: str,
    reason: str = "error",
    failed_model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try user-configured fallback_chain for a specific auxiliary task.

    Reads auxiliary.<task>.fallback_chain from config.yaml and tries each
    entry in order.  Each entry must have at least ``provider``; ``model``,
    ``base_url``, and ``api_key`` are optional.

    ``failed_model`` narrows the skip check to the exact (provider, model)
    pair that just failed, rather than the whole provider. Without it every
    entry sharing the failed provider is skipped (the original behaviour).
    Callers pass it only when a sibling model on the same provider could
    plausibly recover:

    - Model-specific runtime failures (timeout, connection, rate limit,
      model-incompatible, invalid response) pass ``failed_model`` so a
      chain that intentionally lists several models under the same provider
      — e.g. two more NVIDIA NIM models after the primary NIM model times
      out — is not skipped wholesale. Only the exact model that failed is
      skipped; the siblings still run instead of jumping straight to the
      main-agent-model safety net.
    - Provider-wide failures (auth 401, payment 402) and "no client could
      be built" callers leave ``failed_model`` as None, keeping the whole
      provider skipped — the shared credentials/account behind every model
      on that provider are broken, so a sibling can't help and the
      main-agent-model safety net should be reached instead.

    Returns:
        (client, model, provider_label) or (None, None, "") if no fallback.
    """
    from agent.auxiliary_client import (
        _candidate_context_window,
        _fallback_entry_api_key,
        _get_auxiliary_task_config,
        _resolve_fallback_entry,
        _task_minimum_context_length,
    )
    if not task:
        return None, None, ""

    task_config = _get_auxiliary_task_config(task)
    chain = task_config.get("fallback_chain")
    if not chain or not isinstance(chain, list):
        return None, None, ""

    skip_model = (failed_model or "").strip().lower() or None
    # Identity + scope semantics owned by agent.backend_identity (#59561,
    # #72468): a failed_model means the failure was model-scoped (timeout /
    # connection / rate limit) — only the exact deployment is skipped; no
    # failed_model means provider-wide (auth/payment) — the whole credential
    # surface is skipped.
    from agent.backend_identity import (
        BackendIdentity,
        FailureScope,
        should_skip_candidate,
    )

    failed_ident = BackendIdentity.build(
        provider=failed_provider, model=skip_model,
    )
    failure_scope = (
        FailureScope.MODEL if skip_model else FailureScope.CREDENTIAL
    )
    tried = []
    min_ctx = _task_minimum_context_length(task)

    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider", "")).strip()
        if not fb_provider:
            continue
        fb_model_raw = str(entry.get("model", "")).strip()
        if should_skip_candidate(
            BackendIdentity.build(
                provider=fb_provider,
                model=fb_model_raw,
                base_url=str(entry.get("base_url") or ""),
            ),
            failed_ident,
            failure_scope,
        ):
            continue
        fb_model = fb_model_raw or None

        label = f"fallback_chain[{i}]({fb_provider})"

        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception:
            fb_client, resolved_model = None, None

        if fb_client is not None:
            if min_ctx is not None and resolved_model:
                fb_ctx = _candidate_context_window(
                    fb_provider,
                    resolved_model,
                    base_url=str(entry.get("base_url") or ""),
                    api_key=_fallback_entry_api_key(entry) or "",
                )
                if fb_ctx is not None and fb_ctx < min_ctx:
                    logger.info(
                        "Auxiliary %s: skipping %s (%s context=%d < min=%d), continuing chain",
                        task, label, resolved_model, fb_ctx, min_ctx,
                    )
                    tried.append(f"{label} (context too small: {fb_ctx}<{min_ctx})")
                    continue
            logger.info(
                "Auxiliary %s: %s on %s — configured fallback to %s (%s)",
                task, reason, failed_provider, label, resolved_model or fb_model or "default",
            )
            return fb_client, resolved_model or fb_model, label
        tried.append(label)

    if tried:
        logger.debug(
            "Auxiliary %s: configured fallback_chain exhausted (tried: %s)",
            task, ", ".join(tried),
        )
    return None, None, ""


def _try_configured_fallback_for_unavailable_client(
    task: Optional[str],
    failed_provider: str,
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try task fallback_chain when an explicit aux provider cannot build.

    This covers the "no client" case before any request is sent: missing
    raw env key, unavailable OAuth/pool credentials, or provider resolver
    returning ``(None, None)``.  It deliberately stops at the configured
    per-task fallback chain; the main-agent model remains the last-resort
    runtime fallback for request-time capacity errors.
    """
    from agent.auxiliary_client import _try_configured_fallback_chain
    explicit = (failed_provider or "").strip().lower()
    if not task or not explicit or explicit in {"auto"}:
        return None, None, ""
    return _try_configured_fallback_chain(
        task,
        explicit,
        reason="provider unavailable",
    )


def _fallback_entry_api_key(entry: Dict[str, Any]) -> Optional[str]:
    """Resolve inline or env-backed API key from a fallback-chain entry.

    Delegates to the centralized, secret-scope-aware resolver so this path
    doesn't leak another profile's credential via a raw ``os.getenv`` under
    gateway multiplexing (see ``hermes_cli.fallback_config.resolve_entry_api_key``).
    """
    from hermes_cli.fallback_config import resolve_entry_api_key

    return resolve_entry_api_key(entry)


def _resolve_fallback_entry(entry: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve one fallback entry through the central provider router."""
    from agent.auxiliary_client import (
        _fallback_destination_from_entry,
        _fallback_entry_api_key,
        resolve_provider_client,
    )
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip() or None
    if not provider or not model:
        return None, None
    base_url = str(entry.get("base_url") or "").strip() or None
    api_key = _fallback_entry_api_key(entry)
    api_mode = str(entry.get("api_mode") or entry.get("transport") or "").strip() or None
    client, resolved_model = resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
    )
    if client is not None:
        try:
            client._hermes_fallback_destination = _fallback_destination_from_entry(
                entry, client, resolved_model
            )
        except Exception:
            pass
    return client, resolved_model


def _try_main_fallback_chain(
    task: Optional[str],
    failed_provider: str = "",
    reason: str = "error",
) -> Tuple[Optional[Any], Optional[str], str]:
    """Try the top-level main-agent fallback chain for an auxiliary call.

    ``provider: auto`` auxiliary tasks should respect the user's declared
    main fallback policy before dropping into Hermes' built-in discovery
    chain. The top-level chain is read through ``get_fallback_chain`` so
    both modern ``fallback_providers`` and legacy ``fallback_model`` entries
    participate in the same order as the main agent.
    """
    from agent.auxiliary_client import (
        _candidate_context_window,
        _fallback_entry_api_key,
        _is_provider_unhealthy,
        _log_skip_unhealthy,
        _read_main_provider,
        _resolve_fallback_entry,
        _task_minimum_context_length,
    )
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.fallback_config import get_fallback_chain

        chain = get_fallback_chain(load_config_readonly())
    except Exception as exc:
        logger.debug("Auxiliary %s: could not load main fallback chain: %s", task or "call", exc)
        return None, None, ""

    if not chain:
        return None, None, ""

    failed_norm = (failed_provider or "").strip().lower()
    main_norm = (_read_main_provider() or "").strip().lower()
    skip = {p for p in (failed_norm, main_norm, "auto") if p}
    tried: List[str] = []
    min_ctx = _task_minimum_context_length(task)

    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        fb_provider = str(entry.get("provider") or "").strip()
        fb_model = str(entry.get("model") or "").strip()
        if not fb_provider or not fb_model:
            continue
        fb_norm = fb_provider.lower()
        label = f"fallback_providers[{i}]({fb_provider})"
        if fb_norm in skip:
            tried.append(f"{label} (skipped)")
            continue
        if _is_provider_unhealthy(fb_norm):
            _log_skip_unhealthy(fb_norm, task)
            tried.append(f"{label} (unhealthy)")
            continue
        try:
            fb_client, resolved_model = _resolve_fallback_entry(entry)
        except Exception as exc:
            logger.debug("Auxiliary %s: main fallback %s failed to resolve: %s", task or "call", label, exc)
            fb_client, resolved_model = None, None
        if fb_client is not None:
            if min_ctx is not None:
                fb_ctx = _candidate_context_window(
                    fb_provider,
                    resolved_model or fb_model,
                    base_url=str(entry.get("base_url") or ""),
                    api_key=_fallback_entry_api_key(entry) or "",
                )
                if fb_ctx is not None and fb_ctx < min_ctx:
                    logger.info(
                        "Auxiliary %s: skipping %s (context=%d < min=%d), continuing chain",
                        task or "call", label, fb_ctx, min_ctx,
                    )
                    tried.append(f"{label} (context too small: {fb_ctx}<{min_ctx})")
                    continue
            logger.info(
                "Auxiliary %s: %s on %s — main fallback chain to %s (%s)",
                task or "call", reason, failed_provider or "auto", label,
                resolved_model or fb_model,
            )
            return fb_client, resolved_model or fb_model, fb_provider
        tried.append(label)

    if tried:
        logger.debug(
            "Auxiliary %s: main fallback chain exhausted (tried: %s)",
            task or "call", ", ".join(tried),
        )
    return None, None, ""
