"""Provider-aware ``opusplan`` model selection.

``opusplan`` is a special model value (``model.default: opusplan``, ``/model opusplan``,
``--model opusplan``). The main session runs on the active provider's **plan** model (the big /
reasoning one) while delegated children run on its **exec** model (the worker one).
This is an orchestration preset, not Claude Code's Plan-mode lifecycle: the parent
does not automatically change models after planning, and delegation remains tool-driven.

The plan/exec pair of a provider resolves in this order, never by guessing:

1. ``providers.<name>.opusplan: {plan: <id>, exec: <id>}`` in config.yaml (custom / local
   endpoints, e.g. a LiteLLM router, have no built-in opus/sonnet notion);
2. the provider plugin's own ``model_aliases`` when it defines both ``opus`` and ``sonnet``;
3. otherwise :class:`OpusplanError`, naming the provider and the config block to add.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

OPUSPLAN = "opusplan"
ROLE_PLAN = "plan"
ROLE_EXEC = "exec"
_ROLES = (ROLE_PLAN, ROLE_EXEC)
# Provider-plugin alias names that stand for the big (plan) and the cheap (exec) model.
_PLAN_ALIAS, _EXEC_ALIAS = "opus", "sonnet"
_UNSET_PROVIDERS = frozenset({"", "auto"})


class OpusplanError(ValueError):
    """``opusplan`` cannot be resolved for the active provider; the message is user-facing."""


def is_opusplan(value: Any) -> bool:
    """True when ``value`` is the ``opusplan`` model keyword (case-insensitive)."""
    return isinstance(value, str) and value.strip().lower() == OPUSPLAN


def picker_model_ids(models: Iterable[str], provider: str, *, base_url: str = "", user_providers: Any = None) -> list[str]:
    """Offer the preset only for providers with a declared pair; never infer one or resolve auth."""
    result = [m for m in models if not is_opusplan(m)]
    if not _clean_names([provider]):
        return result
    try:
        resolve_opusplan_pair([provider], base_url=base_url, user_providers=user_providers or {})
    except OpusplanError:
        return result
    return [OPUSPLAN, *result]


def _norm_model_id(model: Any) -> str:
    return str(model or "").strip().lower().removesuffix("[1m]")


def _clean_names(names: Iterable[Any]) -> list[str]:
    """Distinct provider names in priority order, without the unresolved ``auto`` placeholder."""
    out: list[str] = []
    for name in names:
        text = str(name or "").strip()
        if text.lower() not in _UNSET_PROVIDERS and text not in out:
            out.append(text)
    return out


def _expand_auto(names: Iterable[Any]) -> list[str]:
    """``names`` cleaned; when none is concrete, the provider ``auto`` currently resolves to."""
    cleaned = _clean_names(names)
    if cleaned:
        return cleaned
    from hermes_cli.auth import AuthError, resolve_provider
    try:
        return _clean_names([resolve_provider("auto")])
    except AuthError:  # no provider configured at all: the caller's error names "auto"
        return []


def _entry_base_url(entry: Mapping[str, Any]) -> str:
    for key in ("base_url", "url", "api"):
        raw = entry.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _find_provider_entry(
    providers: Any, names: list[str], base_url: str,
) -> tuple[Optional[str], Optional[dict]]:
    """The ``providers:`` entry for the active provider: by key/name/``custom:`` slug in ``names`` priority
    order (the first name that matches wins, not the first entry in file order), else by endpoint."""
    if not isinstance(providers, Mapping):
        return None, None
    from hermes_cli.providers import custom_provider_aliases, normalize_provider
    from hermes_cli.route_identity import normalize_route_base_url
    entries = [(str(key), entry) for key, entry in providers.items() if isinstance(entry, Mapping)]
    for name in names:
        low = name.strip().lower()
        cands = {low, low.removeprefix("custom:"), normalize_provider(low)}
        for key, entry in entries:
            aliases = {a.lower() for a in custom_provider_aliases(str(entry.get("name") or ""), key)}
            if cands & aliases:
                return key, dict(entry)
    wanted = normalize_route_base_url(base_url)
    if wanted:
        for key, entry in entries:
            if normalize_route_base_url(_entry_base_url(entry)) == wanted:
                return key, dict(entry)
    return None, None


def _config_pair(key: Optional[str], entry: Mapping[str, Any]) -> Optional[tuple[str, str]]:
    block = entry.get(OPUSPLAN)
    if block is None:
        return None
    plan, exec_ = (str(block.get(role) or "").strip() if isinstance(block, Mapping) else "" for role in _ROLES)
    if not (plan and exec_):
        raise OpusplanError(
            f"providers.{key}.opusplan must set both 'plan' and 'exec' to model ids "
            f"(got plan={plan or None!r}, exec={exec_ or None!r}).")
    return plan, exec_


def _alias_pair(names: list[str]) -> Optional[tuple[str, str]]:
    """``(opus, sonnet)`` model ids from a provider plugin's own ``model_aliases`` table."""
    from providers import get_provider_profile
    for name in names:
        profile = get_provider_profile(name)
        aliases = {str(k).lower(): str(v) for k, v in (getattr(profile, "model_aliases", None) or {}).items()}
        if aliases.get(_PLAN_ALIAS) and aliases.get(_EXEC_ALIAS):
            return aliases[_PLAN_ALIAS], aliases[_EXEC_ALIAS]
    return None


def _default_providers() -> Any:
    from hermes_cli.config import load_config_readonly
    return load_config_readonly().get("providers")


def resolve_opusplan_pair(
    provider_names: Iterable[Any], *, base_url: str = "", user_providers: Any = None,
) -> tuple[str, str]:
    """``(plan_model, exec_model)`` for the active provider. ``provider_names`` lists the identities
    the session knows it by (requested name first, resolved id after); ``base_url`` lets an endpoint
    match its ``providers:`` entry when the session only kept the generic ``custom`` id.
    ``user_providers`` is the config ``providers:`` mapping (loaded when None)."""
    names = _expand_auto(provider_names)
    providers = _default_providers() if user_providers is None else user_providers
    key, entry = _find_provider_entry(providers, names, base_url)
    if entry is not None:
        pair = _config_pair(key, entry)
        if pair is not None:
            return pair
    pair = _alias_pair(names)
    if pair is not None:
        return pair
    label = names[0] if names else "auto"
    block_key = key or (names[0].removeprefix("custom:") if names else "<provider>")
    raise OpusplanError(
        f"'opusplan' needs a plan/exec model pair, but provider '{label}' defines none. Add an "
        f"'opusplan' block for it under providers in config.yaml:\n"
        f"  providers:\n    {block_key}:\n      opusplan:\n        plan: <big/reasoning model id>\n"
        f"        exec: <cheap/worker model id>")


def resolve_opusplan_model(
    role: str, provider_names: Iterable[Any], *, base_url: str = "", user_providers: Any = None,
) -> str:
    """The plan or exec model id of the active provider (see :func:`resolve_opusplan_pair`)."""
    if role not in _ROLES:
        raise ValueError(f"opusplan role must be one of {_ROLES}, got {role!r}")
    pair = resolve_opusplan_pair(provider_names, base_url=base_url, user_providers=user_providers)
    return pair[_ROLES.index(role)]


def resolve_model_in_config(
    model: Any, role: str, cfg: Optional[Mapping[str, Any]], requested_provider: Any = None, base_url: str = "",
) -> Any:
    """``model`` unchanged unless it is ``opusplan``; then its ``role`` model for the provider named by
    ``requested_provider``, else ``cfg['model']['provider']``. ``cfg`` is a full config mapping."""
    if not is_opusplan(model):
        return model
    cfg = cfg or {}
    model_cfg = _model_section(cfg)
    # A concrete requested provider is the only identity; the config provider is the default when none was given.
    names = _clean_names([requested_provider]) or _clean_names([model_cfg.get("provider")])
    return resolve_opusplan_model(
        role, names, base_url=base_url or str(model_cfg.get("base_url") or ""), user_providers=cfg.get("providers") or {})


def _model_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    section = cfg.get("model")
    return section if isinstance(section, Mapping) else {}


def configured_default_model(cfg: Mapping[str, Any]) -> str:
    """``model.default`` (or the ``model: <name>`` shorthand) of a config mapping, as text."""
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, str):
        return model_cfg.strip()
    if isinstance(model_cfg, Mapping):
        raw = model_cfg.get("default") or model_cfg.get("model")
        return raw.strip() if isinstance(raw, str) else ""
    return ""


def stored_mode(raw: Any) -> bool:
    """Read the persisted mode bit without importing a frontend or guessing from a model ID."""
    import json
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return False
    return isinstance(raw, Mapping) and raw.get(OPUSPLAN) is True


def mark_agent(agent: Any, active: bool) -> None:
    """Record on a live agent whether its session runs under ``opusplan`` (read by delegation)."""
    if agent is not None:
        agent.opusplan_active = bool(active)


def session_opusplan_active(parent_agent: Any) -> bool:
    """Whether the parent session runs under ``opusplan``. An explicit boolean flag set by the
    session (``/model opusplan``, CLI start) wins; otherwise the profile config decides: the
    configured default is ``opusplan`` AND the agent is still on that plan model, so a ``/model``
    switch away from it ends the plan/exec split."""
    flag = getattr(parent_agent, "opusplan_active", None)
    if isinstance(flag, bool):
        return flag
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly()
    if not is_opusplan(configured_default_model(cfg)):
        return False
    live = _norm_model_id(getattr(parent_agent, "model", ""))
    plan = resolve_model_in_config(
        OPUSPLAN, ROLE_PLAN, cfg, getattr(parent_agent, "requested_provider", None) or getattr(parent_agent, "provider", None),
        str(getattr(parent_agent, "base_url", "") or ""))
    return bool(live) and live == _norm_model_id(plan)


def delegation_model(
    cfg_model: Any, cfg_provider: Any, cfg_base_url: Any, parent_agent: Any,
) -> Optional[str]:
    """Exec model for a delegated child, or None to leave ``delegation.model`` as configured.

    ``delegation.model: opusplan`` always selects the exec model; an empty ``delegation.model``
    does so while the parent session runs under opusplan. An explicit model id, or a direct
    ``delegation.base_url`` endpoint, is never overridden. A ``delegation.provider`` pins which
    provider's pair applies; otherwise the parent's provider does."""
    if not is_opusplan(cfg_model):
        if cfg_model or cfg_base_url or not session_opusplan_active(parent_agent):
            return None
    if cfg_provider:
        return resolve_opusplan_model(ROLE_EXEC, [cfg_provider])
    return resolve_opusplan_model(
        ROLE_EXEC, [getattr(parent_agent, "requested_provider", None), getattr(parent_agent, "provider", None)],
        base_url=str(getattr(parent_agent, "base_url", "") or ""))


def resolve_startup_model(
    model: Any, role: str, requested_provider: Any, *, base_url: str = "", cfg: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, bool]:
    """``(model, was_opusplan)`` for a launch-time model value. A non-opusplan value passes through;
    ``opusplan`` becomes the ``role`` model of ``requested_provider`` (config provider when empty)."""
    if not is_opusplan(model):
        return model, False
    if cfg is None:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
    return resolve_model_in_config(model, role, cfg, requested_provider, base_url), True


def plan_model(value: Any, cfg: Mapping[str, Any]) -> Any:
    """``value`` (a configured main-model string) with ``opusplan`` replaced by the plan model of ``cfg``'s provider."""
    return resolve_model_in_config(value, ROLE_PLAN, cfg)


def configured_plan_model(cfg: Mapping[str, Any]) -> str:
    """The main model ``cfg`` selects (``model: <name>`` or ``model.default`` / ``model.model``), with
    ``opusplan`` resolved to the plan model; "" when none is configured."""
    section = cfg.get("model", {})
    if isinstance(section, str):
        return plan_model(section, cfg)
    if isinstance(section, Mapping):
        return plan_model(section.get("default") or section.get("model") or "", cfg)
    return ""


def apply_session_override(agent: Any, override: Optional[Mapping[str, Any]]) -> None:
    """A gateway session's ``/model`` override outranks the config default: copy its ``opusplan`` flag onto the agent."""
    flag = (override or {}).get(OPUSPLAN)
    if agent is not None and isinstance(flag, bool):
        agent.opusplan_active = flag
