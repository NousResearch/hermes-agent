"""Turn-scoped model override returned by ``pre_llm_call`` plugins.

A plugin callback may return ``{"runtime_override": {"model": "..."}}`` to run the
current turn's API calls against another model. The override is ephemeral:
``_collect_pre_llm_call_context`` resolves and stages it on ``agent._runtime_override``
for the turn, ``conversation_loop`` applies it around request assembly and the retry
loop, and the pre-override model and projected state are restored when the scope exits.

``model`` is the only supported key. ``provider`` / ``api_key`` / ``base_url`` are
refused so credentials and the network destination never flow through a hook return;
``system_prompt`` is refused because the system-prompt prefix is byte-stable for the
life of a conversation (see ``agent/AGENTS.md``). Unsupported keys are logged and
ignored, never fatal.

Two explicit semantics govern the model name and the model-derived state:

* **Unresolvable names are dropped, never applied blindly.** The name is resolved with
  ``hermes_cli.models_validate.validate_requested_model`` — the same validator the
  ``/model`` switch path calls (``hermes_cli.model_switch._validate_switch``). A name
  the validator does not positively recognize is logged with a warning naming the
  rejected value, and the override is dropped so the turn proceeds on the session
  model. The turn never crashes and a bogus model never reaches the wire.

* **Model-owned state is projected conditionally.** ``agent.model`` alone is not
  enough: the request/preflight path also reads the context window
  (``context_compressor.context_length`` / ``threshold_tokens``), the prompt-cache
  flags (``_use_prompt_caching`` / ``_use_native_cache_layout``), ``reasoning_config``
  and the model capability flags. When the override model differs from the session
  model in any of those — context window, provider (or provider-implied
  transport/credential shape), reasoning/vision capability, prompt-cache support — the
  scope invalidates the cached system prompt and re-projects that state with the
  canonical helper (``agent.agent_runtime_helpers._apply_model_owned_state``, the same
  one ``switch_model`` uses), restoring it on scope exit. When none differ (the same
  model, or a distinct name with an identical projection) the scope is a strict no-op:
  the cached system prompt stays byte-stable and no projection runs.

  Conditional rather than "never": a switch to a model with a different context window
  otherwise overflows or mis-scales compression. Conditional rather than "always":
  invalidating the prompt prefix and re-resolving state on every override would drop
  the session's prompt-cache reuse and pay resolution cost even when the override
  changes nothing the turn reads. Provider is part of the trigger set so a future
  contract that routes across providers cannot silently under-project.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

#: Keys a plugin may override. Anything else is logged and ignored.
RUNTIME_OVERRIDE_KEYS = frozenset({"model"})

_MISSING = object()

#: Model-owned state the scope snapshots and restores when it projects.
_PROJECTED_ATTRS = (
    "_custom_providers",
    "_config_context_length",
    "_use_prompt_caching",
    "_use_native_cache_layout",
    "context_compressor",
    "reasoning_config",
    "_cached_system_prompt",
)


def _resolve_override_model(agent: Any, model: str) -> str:
    """Resolve a hook-supplied model name with the model-switch path's validator.

    Returns the accepted (possibly typo-corrected) model, or ``""`` when the name
    cannot be positively resolved. Resolution is skipped when the name already is the
    session model or the agent has no active route (bare test agents) — there is
    nothing to reject, so the name is passed through unchanged.
    """
    session_model = str(getattr(agent, "model", "") or "").strip()
    if model == session_model:
        return model
    provider = str(getattr(agent, "provider", "") or "").strip()
    if not provider:
        return model
    api_key = getattr(agent, "api_key", None)
    try:
        from hermes_cli.models_validate import validate_requested_model

        verdict = validate_requested_model(
            model,
            provider,
            api_key=api_key if isinstance(api_key, str) else None,
            base_url=str(getattr(agent, "base_url", "") or "") or None,
            api_mode=str(getattr(agent, "api_mode", "") or "") or None,
        )
    except Exception as exc:  # noqa: BLE001 — a resolver failure must never crash the turn
        logger.warning(
            "pre_llm_call runtime_override: model %r could not be resolved (%s); "
            "override dropped, turn continues on %r",
            model, exc, session_model,
        )
        return ""
    if isinstance(verdict, dict) and verdict.get("accepted") and verdict.get("recognized"):
        corrected = verdict.get("corrected_model")
        if isinstance(corrected, str) and corrected.strip():
            return corrected.strip()
        return model
    message = verdict.get("message") if isinstance(verdict, dict) else None
    logger.warning(
        "pre_llm_call runtime_override: model %r is not resolvable for provider %r%s; "
        "override dropped, turn continues on %r",
        model, provider, f" ({message})" if message else "", session_model,
    )
    return ""


def validate_runtime_override(overrides: Any, agent: Any = None) -> Dict[str, str]:
    """Return the supported, correctly-typed keys of a plugin ``runtime_override``.

    Unsupported keys and wrong-typed values are logged and dropped, so a misbehaving
    plugin can never crash the turn. When ``agent`` is supplied, ``model`` is also
    resolved with the model-switch path's validator; an unresolvable name is logged
    with a warning and dropped instead of being applied blindly.
    """
    if not isinstance(overrides, dict):
        logger.warning(
            "pre_llm_call runtime_override ignored: expected dict, got %s",
            type(overrides).__name__,
        )
        return {}
    valid: Dict[str, str] = {}
    for key, value in overrides.items():
        if key not in RUNTIME_OVERRIDE_KEYS:
            logger.warning(
                "pre_llm_call runtime_override: unsupported key %r ignored (supported: %s)",
                key, ", ".join(sorted(RUNTIME_OVERRIDE_KEYS)),
            )
            continue
        if not isinstance(value, str) or not value.strip():
            logger.warning(
                "pre_llm_call runtime_override: key %r must be a non-empty string, ignored",
                key,
            )
            continue
        valid[key] = value.strip()
    if agent is not None and "model" in valid:
        resolved = _resolve_override_model(agent, valid["model"])
        if resolved:
            valid["model"] = resolved
        else:
            valid.pop("model", None)
    return valid


def _model_projection_key(agent: Any, model: str, provider: str) -> tuple:
    """The model-derived values the turn reads for one ``(model, provider)`` route.

    Built from the canonical resolvers: context length via
    ``model_metadata.get_model_context_length`` (the one the switch path re-points the
    compressor with), capability flags via ``models_dev.get_model_capabilities``
    (cache-only, never a network probe on the turn path), and the prompt-cache policy
    via the agent's own ``_anthropic_prompt_cache_policy``. Best-effort: a resolver
    that raises contributes ``None`` so a comparison never crashes the turn.
    """
    base_url = str(getattr(agent, "base_url", "") or "")
    api_key = getattr(agent, "api_key", "")
    if not isinstance(api_key, str):
        api_key = ""
    try:
        from agent.model_metadata import get_model_context_length

        context_length: Optional[int] = get_model_context_length(
            model, base_url=base_url, api_key=api_key, provider=provider,
            config_context_length=getattr(agent, "_config_context_length", None),
            custom_providers=getattr(agent, "_custom_providers", None),
        )
    except Exception:  # noqa: BLE001
        context_length = None
    try:
        from agent.models_dev import get_model_capabilities

        caps = get_model_capabilities(provider, model, allow_network=False)
        capabilities = None if caps is None else (caps.supports_reasoning, caps.supports_vision)
    except Exception:  # noqa: BLE001
        capabilities = None
    try:
        cache_flags = agent._anthropic_prompt_cache_policy(
            provider=provider, base_url=base_url,
            api_mode=str(getattr(agent, "api_mode", "") or ""), model=model,
        )
    except Exception:  # noqa: BLE001
        cache_flags = None
    return (provider, context_length, capabilities, cache_flags)


def _projection_required(agent: Any, model: str, provider: Optional[str] = None) -> bool:
    """True when the override model differs in a way the turn reads.

    Compares the session route's projection key with the override's. A same-model
    override, an agent without a resolvable route, or a model whose window,
    capabilities and prompt-cache flags are identical is a strict no-op.
    """
    session_model = getattr(agent, "model", None)
    if not model or model == session_model:
        return False
    session_provider = str(getattr(agent, "provider", "") or "").strip()
    if not session_provider:
        return False  # no route to resolve model-derived state against
    destination_provider = provider if provider is not None else session_provider
    return _model_projection_key(agent, session_model, session_provider) != _model_projection_key(
        agent, model, destination_provider
    )


def _snapshot_projection(agent: Any) -> Dict[str, Any]:
    """Snapshot the projected attributes and isolate the compressor for the scope.

    ``context_compressor`` is swapped for a scope-owned shallow copy because the
    projection re-points it through ``update_model``, which assigns in place: the
    session compressor must never be mutated, or restoring the reference on exit would
    not undo the override's context length. The copy keeps the durable session handles,
    so bookkeeping for a compression that actually fires persists.
    """
    snapshot = {name: getattr(agent, name, _MISSING) for name in _PROJECTED_ATTRS}
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        agent.context_compressor = copy.copy(compressor)
    return snapshot


def _restore_projection(agent: Any, snapshot: Dict[str, Any]) -> None:
    for name, value in snapshot.items():
        if value is _MISSING:
            try:
                delattr(agent, name)
            except Exception:  # noqa: BLE001
                pass
            continue
        try:
            setattr(agent, name, value)
        except Exception:  # noqa: BLE001 — restore must never raise
            pass


@contextmanager
def apply_runtime_override(agent: Any, overrides: Dict[str, str]) -> Iterator[None]:
    """Run the enclosed turn step with ``agent.model`` set to the override model.

    Restores the pre-override model and projected state on exit unless a fallback or
    redirect took the route while the scope was open — that route then stands, and the
    staged override is cleared so later loop iterations do not re-apply it.

    Model-owned state is projected only when ``_projection_required`` says the override
    changes something the turn reads; otherwise the scope is a strict no-op and the
    cached system prompt stays byte-stable. A missing or empty override is always a
    no-op.
    """
    model = overrides.get("model")
    if not model:
        yield
        return
    previous = getattr(agent, "model", _MISSING)
    projection = _snapshot_projection(agent) if _projection_required(agent, model) else None
    agent.model = model
    if projection is not None:
        # The cached prompt's context-file caps scale with the compressor window, so a
        # projected model needs a fresh build. The pre-override bytes are restored on exit.
        agent._cached_system_prompt = None
        try:
            from agent.agent_runtime_helpers import _apply_model_owned_state

            _apply_model_owned_state(agent, model, snapshot=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pre_llm_call runtime_override: could not project model-derived state "
                "for %r (%s); override dropped, turn continues on %r",
                model, exc, previous if previous is not _MISSING else None,
            )
            _restore_projection(agent, projection)
            if previous is _MISSING:
                try:
                    del agent.model
                except Exception:  # noqa: BLE001
                    pass
            else:
                agent.model = previous
            yield
            return
    try:
        yield
    finally:
        if getattr(agent, "model", None) == model:
            if previous is _MISSING:
                del agent.model
            else:
                agent.model = previous
            if projection is not None:
                _restore_projection(agent, projection)
        else:
            agent._runtime_override = {}


def consume_runtime_override(agent: Any) -> None:
    """Drop the turn's staged override once the fallback chain owns the route.

    A proactive override owns only the primary attempt. ``try_activate_fallback``
    calls this on success, so later loop iterations (and the scope's own exit) do
    not re-apply a model that just failed.
    """
    try:
        agent._runtime_override = {}
    except (AttributeError, TypeError):
        pass
