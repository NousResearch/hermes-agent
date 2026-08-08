"""Shared subagent model and reasoning semantics for Hermes CLI surfaces.

The profile-scoped overrides live under ``delegation``. Model/provider
resolution is not reimplemented here: direct values and both CLI pickers go
through the same ``model_switch.switch_model`` or full provider-setup pipeline
as the primary model controls, so aliases, credentials, catalog validation,
and provider-specific normalization cannot drift. Reasoning values use the
runtime's canonical ``parse_reasoning_effort`` contract so CLI state and child
construction agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


_MISSING_ACTIVE_PROVIDER = object()


@dataclass(frozen=True)
class SubagentModelStatus:
    """Persisted override; no model and no provider means inherit the parent."""

    model: Optional[str]
    provider: Optional[str]
    inherits_parent: bool


@dataclass(frozen=True)
class SubagentReasoningStatus:
    """Persisted child reasoning override; no effort means inherit parent."""

    effort: Optional[str]
    inherits_parent: bool


def _reasoning_status_from_config(config: dict[str, Any]) -> SubagentReasoningStatus:
    delegation = config.get("delegation")
    if not isinstance(delegation, dict) or "reasoning_effort" not in delegation:
        return SubagentReasoningStatus(None, True)

    from hermes_constants import parse_reasoning_effort

    parsed = parse_reasoning_effort(delegation.get("reasoning_effort"))
    if parsed is None:
        # Match child construction: an empty/invalid legacy value has no
        # override effect and therefore inherits the parent's reasoning.
        return SubagentReasoningStatus(None, True)
    if not parsed.get("enabled", True):
        return SubagentReasoningStatus("none", False)
    return SubagentReasoningStatus(str(parsed["effort"]), False)


def _status_from_config(config: dict[str, Any]) -> SubagentModelStatus:
    delegation = config.get("delegation")
    if not isinstance(delegation, dict):
        delegation = {}
    model = str(delegation.get("model") or "").strip() or None
    provider = str(delegation.get("provider") or "").strip() or None
    return SubagentModelStatus(model, provider, not model and not provider)


def get_subagent_model_status() -> SubagentModelStatus:
    from hermes_cli.config import load_config

    return _status_from_config(load_config())


def get_subagent_reasoning_status() -> SubagentReasoningStatus:
    from hermes_cli.config import load_config

    return _reasoning_status_from_config(load_config())


def _mutate_config(mutator):
    """Apply one delegation mutation through the existing config API."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    result = mutator(config)
    save_config(config)
    return result


def _persist_override(
    model: Optional[str], provider: Optional[str]
) -> SubagentModelStatus:
    """Atomically persist (or clear) the two-key delegation override."""

    def apply(config):
        delegation = config.get("delegation")
        if not isinstance(delegation, dict):
            delegation = {}
        else:
            delegation = dict(delegation)

        if model:
            delegation["model"] = str(model).strip()
            if provider:
                delegation["provider"] = str(provider).strip()
            else:
                delegation.pop("provider", None)
        else:
            delegation.pop("model", None)
            delegation.pop("provider", None)

        if delegation:
            config["delegation"] = delegation
        else:
            config.pop("delegation", None)
        return _status_from_config(config)

    return _mutate_config(apply)


def persist_subagent_switch_result(result: Any) -> SubagentModelStatus:
    """Commit a successful shared ``ModelSwitchResult`` to delegation config."""

    if not getattr(result, "success", False):
        message = str(getattr(result, "error_message", "") or "Invalid subagent model")
        raise ValueError(message)
    model = str(getattr(result, "new_model", "") or "").strip()
    provider = str(getattr(result, "target_provider", "") or "").strip()
    if not model:
        raise ValueError("Model selection resolved to an empty model")
    return _persist_override(model, provider or None)


def resolve_subagent_model(model: str, *, provider: Optional[str] = None):
    """Resolve and validate through the canonical model-switch pipeline."""

    raw_model = str(model or "").strip()
    if not raw_model:
        raise ValueError("model is required")

    from hermes_cli.config import load_config
    from hermes_cli.inventory import load_picker_context
    from hermes_cli.model_switch import switch_model

    context = load_picker_context()
    current_override = _status_from_config(load_config())
    target_provider = str(provider or current_override.provider or "").strip()
    result = switch_model(
        raw_input=raw_model,
        current_provider=current_override.provider or context.current_provider,
        current_model=current_override.model or context.current_model,
        current_base_url=context.current_base_url,
        current_api_key="",
        is_global=False,
        explicit_provider=target_provider,
        user_providers=context.user_providers,
        custom_providers=context.custom_providers,
    )
    if not result.success:
        raise ValueError(result.error_message or "Invalid subagent model")
    return result


def set_subagent_model(
    model: str, *, provider: Optional[str] = None
) -> SubagentModelStatus:
    """Resolve, validate, normalize, then persist a subagent override."""

    return persist_subagent_switch_result(
        resolve_subagent_model(model, provider=provider)
    )


def reset_subagent_model() -> SubagentModelStatus:
    """Remove only model/provider; preserve every other delegation setting."""

    return _persist_override(None, None)


def _persist_reasoning(effort: Optional[str]) -> SubagentReasoningStatus:
    """Persist or clear only the child reasoning override."""

    def apply(config):
        delegation = config.get("delegation")
        if not isinstance(delegation, dict):
            delegation = {}
        else:
            delegation = dict(delegation)

        if effort is None:
            delegation.pop("reasoning_effort", None)
        else:
            delegation["reasoning_effort"] = effort

        if delegation:
            config["delegation"] = delegation
        else:
            config.pop("delegation", None)
        return _reasoning_status_from_config(config)

    return _mutate_config(apply)


def set_subagent_reasoning_effort(effort: str) -> SubagentReasoningStatus:
    """Validate, canonicalize, and persist child reasoning effort."""

    from hermes_constants import parse_reasoning_effort

    raw = str(effort or "").strip()
    parsed = parse_reasoning_effort(raw)
    if parsed is None:
        raise ValueError(
            "Invalid subagent reasoning effort. Expected one of: "
            "none, minimal, low, medium, high, xhigh, max, ultra"
        )
    canonical = "none" if not parsed.get("enabled", True) else str(parsed["effort"])
    return _persist_reasoning(canonical)


def reset_subagent_reasoning_effort() -> SubagentReasoningStatus:
    """Make future children inherit the parent's reasoning configuration."""

    return _persist_reasoning(None)


def list_subagent_picker_providers(*, refresh: bool = False) -> list[dict[str, Any]]:
    """Return the same authenticated provider/model inventory as model pickers."""

    if refresh:
        try:
            from hermes_cli.models import clear_provider_models_cache

            clear_provider_models_cache()
        except Exception:
            pass

    from hermes_cli.inventory import build_models_payload, load_picker_context

    context = load_picker_context()
    return list(
        build_models_payload(
            context,
            probe_custom_providers=refresh,
            probe_current_custom_provider=not refresh,
        ).get("providers")
        or []
    )


def _canonical_picker_provider(model_config: Any, full_config: dict[str, Any]) -> str:
    """Return the runtime-addressable provider selected by the full picker.

    The manual custom-endpoint flow temporarily writes ``model.provider=custom``
    plus ``model.base_url`` and then saves a named ``custom_providers`` entry.
    Delegation must persist that entry's canonical ``custom:<name>`` slug; bare
    ``custom`` is ambiguous when more than one endpoint exists.
    """

    if not isinstance(model_config, dict):
        return ""
    provider = str(model_config.get("provider") or "").strip()
    if provider != "custom":
        return provider

    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.providers import custom_provider_slug
    from hermes_cli.route_identity import normalize_route_base_url

    selected_url = normalize_route_base_url(model_config.get("base_url"))
    if not selected_url:
        raise ValueError("Custom model selection did not persist an endpoint URL")
    for entry in get_compatible_custom_providers(full_config):
        if not isinstance(entry, dict):
            continue
        if normalize_route_base_url(entry.get("base_url")) != selected_url:
            continue
        identity = str(entry.get("provider_key") or entry.get("name") or "").strip()
        if identity:
            return custom_provider_slug(identity)
    raise ValueError(
        "Custom endpoint was selected but no matching saved custom provider was found"
    )


def _read_auth_active_provider() -> Any:
    """Read the primary auth provider before the picker mutates it."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store

    with _auth_store_lock():
        store = _load_auth_store()
        return store.get("active_provider", _MISSING_ACTIVE_PROVIDER)


def _restore_primary_route(model_before: Any, active_provider_before: Any) -> None:
    """Restore both primary-route stores and report partial cleanup."""
    import copy

    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    from hermes_cli.config import load_config, save_config

    restore_errors: list[BaseException] = []
    try:
        config = load_config()
        if model_before is None:
            config.pop("model", None)
        else:
            config["model"] = copy.deepcopy(model_before)
        save_config(config)
    except BaseException as exc:
        restore_errors.append(exc)

    try:
        with _auth_store_lock():
            store = _load_auth_store()
            if active_provider_before is _MISSING_ACTIVE_PROVIDER:
                store.pop("active_provider", None)
            else:
                store["active_provider"] = active_provider_before
            _save_auth_store(store)
    except BaseException as exc:
        restore_errors.append(exc)

    if restore_errors:
        details = "; ".join(str(exc) for exc in restore_errors)
        raise RuntimeError(f"Could not restore the primary model/auth route: {details}")


def select_subagent_model_interactively(
    *, refresh: bool = False, initial_provider: Optional[str] = None
) -> Optional[SubagentModelStatus]:
    """Run the complete ``hermes model`` flow for the delegation target.

    Provider logins, credentials, custom-provider additions, and auxiliary
    configuration are deliberately retained. The temporary primary route is
    restored before the confirmed selection is committed under ``delegation.*``.
    """
    import copy
    import sys

    from hermes_cli.auth import capture_model_selection
    from hermes_cli.config import load_config
    from hermes_cli.main import select_provider_and_model

    if refresh:
        try:
            from hermes_cli.models import clear_provider_models_cache

            clear_provider_models_cache()
        except Exception:
            pass

    before_config = load_config()
    initial_status = _status_from_config(before_config)
    picker_provider = initial_provider or initial_status.provider
    picker_model = (
        initial_status.model
        if not initial_provider or initial_provider == initial_status.provider
        else None
    )
    model_before = copy.deepcopy(before_config.get("model"))
    active_provider_before = _read_auth_active_provider()
    selected_config: Optional[dict[str, Any]] = None
    selections: list[str] = []

    print()
    print("  Select the provider + model to use for subagents.")
    print("  This is the full `hermes model` setup flow: provider login and custom")
    print("  provider additions are kept; your active primary model is unchanged.")
    print()

    try:
        with capture_model_selection(selections.append):
            select_provider_and_model(
                initial_model=picker_model,
                initial_provider=picker_provider,
            )
        if selections:
            selected_config = copy.deepcopy(load_config())
    finally:
        active_error = sys.exc_info()[1]
        try:
            _restore_primary_route(model_before, active_provider_before)
        except Exception as restore_error:
            message = (
                "Could not restore the primary model/auth route after subagent "
                f"selection: {restore_error}"
            )
            if active_error is not None:
                active_error.add_note(message)
            else:
                raise RuntimeError(message) from restore_error

    if not selections or selected_config is None:
        return None

    model_config = selected_config.get("model")
    selected_model = selections[-1]
    selected_provider = _canonical_picker_provider(model_config, selected_config)
    return set_subagent_model(selected_model, provider=selected_provider or None)
