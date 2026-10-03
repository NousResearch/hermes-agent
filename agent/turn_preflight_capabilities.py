"""Advisory model-capability pre-flight for the conversation turn loop.

The runtime sends tool schemas, image content and reasoning parameters on every
request. When the active model does not support one of them the provider only
says so afterwards, as an opaque ``400 unsupported parameter``, which reads like a
Hermes bug. This module turns that into a warning the user sees BEFORE the call.

Two rules make it safe to run on the hot path:

* **Advisory only.** Nothing here returns a verdict, so no turn can be blocked by
  it — an unknown model is far more common than a broken one, and a pre-flight
  that blocks would strand those users on a spinner.
* **Absence is not a negative.** ``ModelCapabilities`` uses ``Optional[bool]`` for
  the capabilities models.dev may simply not record, and ``get_model_capabilities``
  returns ``None`` outright for a model it cannot resolve. Only a capability
  metadata positively declares ``False`` is reported.

Capability data comes from the existing models.dev lookup
(:func:`agent.models_dev.get_model_capabilities`), which already merges the live
catalog with ``model_overrides`` — never a second regex table.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.conversation_loop")

# Capability attribute on ModelCapabilities -> what the request carries when it is set.
_SENT_LABEL = {
    "supports_tools": "tool schemas",
    "supports_vision": "image content",
    "supports_reasoning": "reasoning parameters",
}


def _model_capabilities(provider: str, model: str):
    """``ModelCapabilities`` for the pair, or None when unresolvable.

    ``allow_network=False``: this sits on the turn hot path and must answer from
    the models.dev disk cache or not at all.
    """
    from agent.models_dev import get_model_capabilities

    return get_model_capabilities(provider, model, allow_network=False)


def _sent_tools(agent: Any, messages: Any) -> bool:
    return bool(getattr(agent, "tools", None))


def _sent_vision(agent: Any, messages: Any) -> bool:
    from agent.vision_message_prep import _is_image_part

    return isinstance(messages, (list, tuple)) and any(
        isinstance(m, dict) and isinstance(m.get("content"), list)
        and any(_is_image_part(part) for part in m["content"])
        for m in messages
    )


def _reasoning_control_keys() -> Any:
    """The reasoning wire-control vocabulary.

    Owned by the auxiliary ladder (``agent.auxiliary_client._PROFILE_REASONING_KEYS``) and
    imported rather than copied: one key list, so a route that adds a control is recognised
    here the same turn it is recognised there.
    """
    from agent.auxiliary_client import _PROFILE_REASONING_KEYS

    return _PROFILE_REASONING_KEYS


def _carries_reasoning_control(value: Any) -> bool:
    """True when a payload node carries a reasoning wire control (recursive: a profile may
    nest its own shape, e.g. an OpenAI-compatible Gemini's ``extra_body.extra_body``)."""
    if not isinstance(value, dict):
        return False
    keys = _reasoning_control_keys()
    return any(
        str(key).strip().lower() in keys or _carries_reasoning_control(nested)
        for key, nested in value.items()
    )


def _sent_reasoning(api_kwargs: Any) -> bool:
    """True when the ASSEMBLED request carries reasoning wire controls.

    ``agent.reasoning_config`` is intent, not evidence. The request builder drops the
    reasoning fields on a route that refuses them (and for the rest of a session whose
    effort the route rejected), while a provider profile can project its own shape
    (DeepInfra's top-level ``reasoning_effort``). Warning off the raw config told users the
    provider might reject reasoning parameters the request never carried; read the outgoing
    request instead — the same rule the image scan follows.
    """
    if not isinstance(api_kwargs, dict):
        return False
    keys = _reasoning_control_keys()
    if any(str(key).strip().lower() in keys for key in api_kwargs):
        return True
    return _carries_reasoning_control(api_kwargs.get("extra_body"))


def _emit_once(agent: Any, key: tuple, message: str) -> None:
    """Warn at most once per (provider, model, capability) per agent.

    The turn loop runs this on every iteration, so an undeduped warning would
    repeat up to ``max_iterations`` times for one incompatibility.
    """
    warned = getattr(agent, "_capability_preflight_warned", None)
    if not isinstance(warned, set):
        warned = set()
        agent._capability_preflight_warned = warned
    if key in warned:
        return
    warned.add(key)
    logger.warning(message)
    agent._emit_diagnostic_status(message)


def capability_mismatches(
    caps: Any, *, sends_tools: bool, sends_vision: bool, sends_reasoning: bool,
) -> list[str]:
    """Capabilities the request sends that ``caps`` declares unsupported.

    A capability that is ``None`` on ``caps`` is undetermined, not unsupported,
    and is skipped. Split out from the agent so the rule is directly testable.
    """
    sends = {
        "supports_tools": sends_tools,
        "supports_vision": sends_vision,
        "supports_reasoning": sends_reasoning,
    }
    return [name for name, sent in sends.items() if sent and getattr(caps, name, None) is False]


def warn_capability_mismatches(agent: Any, messages: Any, api_kwargs: Any) -> None:
    """Warn about capabilities the outgoing request uses but the model lacks.

    Runs on the ASSEMBLED request (``api_kwargs`` as built for the call, with ``messages``
    the conversation it was built from), so every signal is something the request actually
    carries rather than something the agent merely intends.

    Never blocks and never raises: a catalog read failure degrades to silence,
    since a missing warning is strictly better than a dead turn. Only the catalog
    lookup is guarded — a bug in the sent-detection below must surface, not hide
    behind the same silence this feature relies on.
    """
    provider = str(getattr(agent, "provider", "") or "")
    model = str(getattr(agent, "model", "") or "")
    if not provider or not model:
        return
    try:
        caps = _model_capabilities(provider, model)
    except Exception:
        logger.debug("Capability pre-flight lookup failed for %s/%s", provider, model, exc_info=True)
        return
    if caps is None:
        return
    mismatched = capability_mismatches(
        caps,
        sends_tools=_sent_tools(agent, messages),
        sends_vision=_sent_vision(agent, messages),
        sends_reasoning=_sent_reasoning(api_kwargs),
    )
    for name in mismatched:
        _emit_once(
            agent,
            (provider, model, name),
            f"Note: {provider}/{model} does not support {_SENT_LABEL[name]} — "
            f"the provider may reject the request. Switching models avoids it.",
        )


__all__: list[str] = ["capability_mismatches", "warn_capability_mismatches"]