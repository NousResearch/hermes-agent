"""Init-time fallback chain: entry normalization, chain setup, and the fallback's api_mode."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("run_agent")


def _fallback_entries(fallback_model) -> list[dict[str, Any]]:
    """Normalize legacy single-dict ``fallback_model`` / list ``fallback_providers``."""
    if isinstance(fallback_model, dict):
        fallback_model = [fallback_model]
    if not isinstance(fallback_model, list):
        return []
    return [
        f for f in fallback_model if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]


def _init_fallback_chain(agent, fallback_model):
    # Stable pool-entry identity: OAuth refreshes can replace the token before a failed
    # request is recovered, so the key value alone can't attribute the failure.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    # Ordered backups tried when the primary is exhausted (legacy single-dict or list).
    agent._fallback_chain = _fallback_entries(fallback_model)
    agent._fallback_index = 0
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)
    # Legacy attribute kept for backward compat (tests, external callers)
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    chain = agent._fallback_chain
    if chain and not agent.quiet_mode:
        labels = [f"{f['model']} ({f['provider']})" for f in chain]
        if len(chain) == 1:
            print(f"🔄 Fallback model: {labels[0]}")
        else:
            print(f"🔄 Fallback chain ({len(chain)} providers): " + " → ".join(labels))


def recompute_init_fallback_api_mode(agent, fb_client, entry: dict[str, Any]) -> None:
    """Give an init-time fallback its own api_mode instead of the unreachable primary's.

    A Nous primary configured with ``api_mode: codex_responses`` otherwise leaves a fallback whose
    profile serves that model on chat completions (``api_mode_for_model``) on the Responses API,
    which silently drops reasoning content (#46527).
    The per-turn path (``try_activate_fallback``) already recomputes via
    ``_fallback_api_mode_resolved``; this keeps init in agreement, and runs before the
    ``_primary_runtime`` snapshot so ``restore_primary_runtime`` replays the recomputed mode.
    The entry's hint comes first, as in ``bind_route_entry``: an explicit ``api_mode`` (on the entry
    or its ``providers.<name>`` block) or a non-chat hint is kept, and only the chat default is
    re-detected, so a declared Messages relay on a plain host stays on Messages (#33062, #81932).
    An Actual route is pinned to chat completions before any hint, as in the binder.
    A detection failure keeps the primary's mode and never blocks the fallback itself.
    """
    from agent.chat_completion_helpers import _fallback_api_mode_hint, _fallback_api_mode_resolved
    from hermes_cli.providers import is_actual_route
    fb_base_url = str(getattr(fb_client, "base_url", "") or "")
    try:
        if is_actual_route(agent.provider, fb_base_url):
            agent.api_mode = "chat_completions"  # Actual serves chat completions only, whatever the hint.
        else:
            explicit, hinted = _fallback_api_mode_hint(
                entry, agent.provider, str(entry.get("base_url") or "").strip() or None)
            agent.api_mode = hinted if explicit or hinted != "chat_completions" else _fallback_api_mode_resolved(
                agent, agent.provider, agent.model, fb_base_url)
    except Exception:
        logger.debug("Init-time fallback api_mode detection failed for %s", agent.provider, exc_info=True)
        return
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()


def _resolvable_fallback_entries(fallback_model, refused: list[tuple[str, str]], start: int = 0):
    """Yield ``(index, entry, routed_client, model)`` for each fallback entry from *start* on that
    the router can resolve.

    Entries that raise or resolve no usable credentials are appended to *refused* as
    ``(provider, reason)`` and skipped.
    """
    from agent.auxiliary_client import resolve_provider_client
    from agent.chat_completion_helpers import _fallback_api_mode_hint
    for _index, _fb in enumerate(_fallback_entries(fallback_model)):
        if _index < start:
            continue
        _fb_provider = str(_fb["provider"])
        try:
            from hermes_cli.fallback_config import resolve_entry_api_key
            _fb_explicit_key = resolve_entry_api_key(_fb)
            _fb_base_url = str(_fb.get("base_url") or "").strip() or None
            # Tell the resolver a known wire: unhinted, it rewrites a dual-surface ``/anthropic`` base to
            # ``/v1`` and a Messages client loses its path. The chat default is only a guess, so it stays
            # unset and the resolver can still detect Messages from a named block's ``/anthropic`` base.
            _explicit, _hinted = _fallback_api_mode_hint(_fb, _fb_provider, _fb_base_url)
            _fb_client, _fb_model = resolve_provider_client(
                _fb["provider"], model=_fb["model"], raw_codex=True,
                explicit_base_url=_fb_base_url, explicit_api_key=_fb_explicit_key,
                api_mode=_hinted if _explicit or _hinted != "chat_completions" else None,
            )
        except Exception as _fb_exc:
            logger.debug("Init-time fallback entry %s failed: %s", _fb_provider, _fb_exc, exc_info=True)
            # A bare exception (``KeyError()``) stringifies empty; name its type instead.
            refused.append((_fb_provider, str(_fb_exc) or type(_fb_exc).__name__))
            continue
        if _fb_client is None:
            # The router returns None when no credentials are usable for the entry — a skip
            # that leaves no trace otherwise, hiding key-less fallback entries from the log.
            logger.debug(
                "Init-time fallback entry %s resolved no usable credentials", _fb_provider
            )
            refused.append((_fb_provider, "no usable credentials"))
            continue
        yield _index, _fb, _fb_client, _fb_model


def _activate_init_fallback(agent, index, entry, fb_client, fb_model, _provider_timeout) -> dict[str, Any] | None:
    """Switch the agent to a resolved fallback entry; return its OpenAI-client kwargs.

    ``None`` when the entry is a MoA preset: the facade is bound and there is no OpenAI
    client to construct. ``agent._init_fallback_next_index`` records where a later init walk resumes.
    """
    agent._fallback_activated = True
    agent._init_fallback_next_index = index + 1
    from agent.chat_completion_helpers import _rebind_fallback_credential_pool
    provider = str(entry["provider"]).strip().lower()
    if provider == "moa":
        # The chokepoint handed back the preset's aggregator client, which only proves the
        # preset resolves and its aggregator has credentials. A MoA entry means the preset
        # itself (same as ``provider: moa`` in config), so bind the facade, not the aggregator.
        from agent.moa_loop import bind_moa_runtime
        _rebind_fallback_credential_pool(agent, provider, entry["model"])
        bind_moa_runtime(agent, entry["model"])
        return None
    from agent.agent_init import _client_kwargs_from_routed
    agent.provider = agent.requested_provider = entry["provider"]
    agent.model = fb_model or entry["model"]
    recompute_init_fallback_api_mode(agent, fb_client, entry)
    client_kwargs = _client_kwargs_from_routed(fb_client, _provider_timeout)
    agent.base_url = client_kwargs.get("base_url", agent.base_url)
    # The primary's pool must not rotate its keys into the fallback's route (the per-turn
    # binder does the same through ``bind_route_entry``).
    _rebind_fallback_credential_pool(agent, provider, agent.model)
    return client_kwargs


NATIVE_WIRE_MODES = ("anthropic_messages", "bedrock_converse")


def bind_native_wire_fallback(agent, client_kwargs: dict[str, Any], _provider_timeout) -> bool:
    """Bind an init-time fallback whose api_mode is not OpenAI-wire the way ``_build_client``
    binds a primary with that mode. True when bound (no OpenAI client to construct)."""
    from agent.agent_init import _init_anthropic_client, _init_bedrock_client
    if agent.api_mode == "anthropic_messages":
        _init_anthropic_client(agent, client_kwargs.get("api_key"), client_kwargs.get("base_url"), _provider_timeout)
        return True
    if agent.api_mode == "bedrock_converse":
        _init_bedrock_client(agent, client_kwargs.get("base_url"))
        return True
    return False


def construct_init_fallback_client(agent, fallback_model, _provider_timeout,
                                   build_exc: Exception) -> dict[str, Any] | None:
    """Bind the next fallback entry that builds a client, after the current route's build raised.

    Resumes after any entry init already bound. Returns the fallback's OpenAI-client kwargs
    (``agent.client`` is set), or ``None`` when the entry bound itself (MoA facade, Anthropic
    or Bedrock wire). Re-raises *build_exc* when no entry yields a client.
    """
    from agent.agent_init import _prepare_openai_client_kwargs
    failed = agent.provider
    refused: list[tuple[str, str]] = []
    start = getattr(agent, "_init_fallback_next_index", 0)
    for index, entry, fb_client, fb_model in _resolvable_fallback_entries(fallback_model, refused, start):
        try:
            fb_kwargs = _activate_init_fallback(agent, index, entry, fb_client, fb_model, _provider_timeout)
            self_bound = fb_kwargs is None or bind_native_wire_fallback(agent, fb_kwargs, _provider_timeout)
            if not self_bound:
                _prepare_openai_client_kwargs(agent, fb_kwargs, _provider_timeout)
                agent.client = agent._create_openai_client(fb_kwargs, reason="agent_init_fallback", shared=True)
        except Exception as fb_exc:
            logger.debug("Init-time fallback entry %s failed to build a client: %s", entry["provider"], fb_exc,
                         exc_info=True)
            # A native bind that raised part-way must not leave its route on the agent.
            agent._anthropic_client, agent._is_anthropic_oauth = None, False
            refused.append((str(entry["provider"]), str(fb_exc) or type(fb_exc).__name__))
            continue
        logger.warning(
            "%r failed to build its client (%s: %s); using fallback %s (%s)",
            failed, type(build_exc).__name__, build_exc, agent.model, agent.provider,
        )
        if not agent.quiet_mode:
            print(f"⚠️  {failed} could not build its client ({type(build_exc).__name__}); using fallback "
                  f"{agent.model} ({agent.provider})")
            if fb_kwargs is None:
                print(f"🤖 AI Agent initialized with MoA preset: {agent.model}")
        return None if self_bound else fb_kwargs
    if refused:
        logger.warning(
            "%r failed to build its client (%s: %s); fallback entries refused: %s",
            failed, type(build_exc).__name__, build_exc,
            "; ".join(f"{_p} ({_r})" for _p, _r in refused),
        )
    raise build_exc
