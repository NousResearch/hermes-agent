"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _normalized_base_url(value: Any) -> str:
    return value.strip().rstrip("/") if isinstance(value, str) else ""


def _normalize_provider_id(value: str) -> str:
    """Collapse provider aliases to the canonical id (lazy to avoid import cycles).

    ``opencode-zen`` is the legacy pre-rename id of ``opencode``; ``zen`` is a
    short alias. Without normalization, two fallback entries naming the same
    backend under different ids are kept as separate slots, shifting the chain
    and letting an alias slip past the same-backend skip (#85235).
    """
    try:
        from hermes_cli.providers import normalize_provider
    except Exception:  # pragma: no cover - import resilience
        # Silent degradation: alias collapse is disabled and we fall back to a
        # bare lowercase. Log so a future import cycle/refactor can't regress
        # the dedup invisibly.
        logger.warning(
            "fallback provider alias collapse degraded to lowercase: "
            "could not import hermes_cli.providers.normalize_provider",
            exc_info=True,
        )
        return value.lower()
    return normalize_provider(value)


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``api_key_env`` accepted as alias); None when neither
    yields a value so ``resolve_runtime_provider`` falls through to standard credential resolution.
    ``key_env`` goes through ``agent.secret_scope.get_secret``, not raw ``os.getenv``: in a
    multiplexed gateway a bare env read ignores the active profile's scope and can return another
    profile's credential.
    """
    if not isinstance(entry, dict):
        return None
    if inline := str(entry.get("api_key") or "").strip():
        return inline
    if key_env := str(entry.get("key_env") or entry.get("api_key_env") or "").strip():
        from agent.secret_scope import get_secret
        return (get_secret(key_env) or "").strip() or None
    return None


def effective_runtime_provider(
    entry: dict[str, Any] | None, runtime: dict[str, Any] | None
) -> str:
    """Provider identity to persist/display for a resolved fallback entry.

    ``resolve_runtime_provider`` returns the bare billing class ``"custom"``
    for every named ``providers:`` / ``custom_providers:`` entry; the entry's
    configured id only survives in ``requested_provider``. Fallback resolvers
    that persist ``runtime["provider"]`` as the agent identity therefore label
    sessions/billing rows ``custom`` instead of the configured provider name —
    while the manual ``/model`` switch path correctly persists the named id
    (#98739). Same class as the delegation fix in ``tools/delegate_tool.py``.

    Returns the entry's requested identity when the resolved provider is the
    bare ``custom`` class; a genuinely ad-hoc endpoint (requested provider IS
    ``custom``) keeps the bare class unchanged.
    """
    runtime = runtime or {}
    resolved = str(runtime.get("provider") or "").strip()
    if resolved.lower() != "custom":
        return resolved
    requested = str(
        runtime.get("requested_provider")
        or (entry or {}).get("provider")
        or ""
    ).strip()
    if requested and requested.lower() != "custom":
        return requested
    return resolved


def pre_agent_fallback_notice(
    primary_provider: Any, primary_model: Any, fallback_provider: Any, fallback_model: Any
) -> str:
    """User-visible one-shot line for a provider switch made during credential resolution, before
    any AIAgent exists (#74349). Shared by the messaging gateway, the TUI/Desktop gateway and cron
    so the three pre-agent fallback paths cannot drift in wording."""
    primary_desc = "/".join(str(p).strip() for p in (primary_provider, primary_model) if p) or "primary"
    fallback_desc = "/".join(str(p).strip() for p in (fallback_provider, fallback_model) if p) or "fallback"
    return f"⚠️ Provider fallback: {primary_desc} unavailable; using {fallback_desc} for this response."



def is_fallback_heuristics_enabled(
    config: dict[str, Any] | None = None,
    entry: dict[str, Any] | None = None,
) -> bool:
    """Check whether dynamic fallback heuristics are enabled.

    By default, fallback heuristics are OFF (False) and must be explicitly switched on.
    Users can enable heuristics via:
    - Entry level: ``entry["heuristic_enabled"]`` or ``entry["enable_heuristic"]``
    - Top level config: ``fallback_heuristics: true`` (or ``fallback_heuristics_enabled: true``)
    - Agent section: ``agent.fallback_heuristics: true``
    - Fallback section: ``fallback.heuristics: true``
    """
    if entry and isinstance(entry, dict):
        if entry.get("heuristic_enabled") is True or entry.get("enable_heuristic") is True:
            return True
        if entry.get("heuristic_enabled") is False or entry.get("enable_heuristic") is False:
            return False

    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly() or {}
        except (OSError, ValueError, TypeError, KeyError):
            config = {}

    if not isinstance(config, dict):
        return False

    if "fallback_heuristics" in config:
        return bool(config["fallback_heuristics"])
    if "fallback_heuristics_enabled" in config:
        return bool(config["fallback_heuristics_enabled"])

    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        if "fallback_heuristics" in agent_cfg:
            return bool(agent_cfg["fallback_heuristics"])
        if "fallback_heuristics_enabled" in agent_cfg:
            return bool(agent_cfg["fallback_heuristics_enabled"])

    fallback_cfg = config.get("fallback")
    if isinstance(fallback_cfg, dict):
        if "heuristics" in fallback_cfg:
            return bool(fallback_cfg["heuristics"])
        if "heuristics_enabled" in fallback_cfg:
            return bool(fallback_cfg["heuristics_enabled"])

    return False


def _iter_fallback_entries(raw: Any, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    candidates = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or entry.get("model_provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        is_already_materialized = bool(entry.get("_is_heuristic") or entry.get("criteria_matched"))
        has_criteria = not is_already_materialized and (
            bool(entry.get("criteria") or entry.get("heuristic")) or (
                model.startswith(("auto:", "heuristic:", "criteria:"))
            )
        )
        if not provider and has_criteria:
            try:
                from hermes_cli.fallback_heuristics import parse_fallback_criteria
                crit = parse_fallback_criteria(entry)
                provider = crit.provider or "nous"
            except (ValueError, TypeError, KeyError):
                provider = "nous"
            entry = {**entry, "provider": provider}
        if not provider:
            continue
        if has_criteria:
            # Check if heuristics are switched on (default: OFF)
            heuristics_active = is_fallback_heuristics_enabled(config, entry)
            if not heuristics_active:
                logger.debug(
                    "Fallback heuristic on provider %s skipped because fallback heuristics are disabled by default. "
                    "Enable via 'fallback_heuristics: true' in config.yaml or 'hermes fallback heuristics on'.",
                    provider,
                )
                if model and not model.startswith(("auto:", "heuristic:", "criteria:")):
                    normalized = {**entry, "provider": provider, "model": model}
                    normalized.pop("criteria", None)
                    normalized.pop("heuristic", None)
                    base_url = _normalized_base_url(entry.get("base_url"))
                    if base_url:
                        normalized["base_url"] = base_url
                    entries.append(normalized)
                continue

            try:
                from hermes_cli.fallback_heuristics import resolve_fallback_entry
                resolved_entries = resolve_fallback_entry(entry)
            except Exception as exc:
                logger.warning("Failed to resolve fallback criteria for provider %s: %s", provider, exc, exc_info=True)
                resolved_entries = []
            for res in resolved_entries:
                res_provider = str(res.get("provider") or "").strip()
                res_model = str(res.get("model") or "").strip()
                if not res_provider or not res_model:
                    continue
                normalized = {**res, "provider": res_provider, "model": res_model}
                base_url = _normalized_base_url(res.get("base_url"))
                if base_url:
                    normalized["base_url"] = base_url
                entries.append(normalized)
            continue
        if not model:
            continue
        normalized = {**entry, "provider": provider, "model": model}
        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url
        entries.append(normalized)

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for e in entries:
        ident = _entry_identity(e)
        if ident not in seen:
            seen.add(ident)
            deduped.append(e)
    return deduped


def get_stored_fallback_rules(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the raw, declarative fallback rules stored in config.

    Preserves inactive heuristic rules, does not query catalogs or materialize candidates,
    iterating over both fallback_providers and fallback_model, deduping by identity and
    marking legacy items with _from_fallback_model = True.
    Always returns fresh dict copies.
    """
    if not isinstance(config, dict):
        return []
    import copy

    rules: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    fp_raw = config.get("fallback_providers")
    fp_candidates = [fp_raw] if isinstance(fp_raw, dict) else fp_raw if isinstance(fp_raw, list) else []
    for entry in fp_candidates:
        if isinstance(entry, dict):
            ident = _entry_identity(entry)
            if ident not in seen:
                seen.add(ident)
                rules.append(copy.deepcopy(entry))

    fm_raw = config.get("fallback_model")
    fm_candidates = [fm_raw] if isinstance(fm_raw, dict) else fm_raw if isinstance(fm_raw, list) else []
    for entry in fm_candidates:
        if isinstance(entry, dict):
            ident = _entry_identity(entry)
            if ident not in seen:
                seen.add(ident)
                e_copy = copy.deepcopy(entry)
                e_copy["_from_fallback_model"] = True
                rules.append(e_copy)

    return rules


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _normalize_provider_id(str(entry.get("provider") or "").strip()),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its order. Legacy
    ``fallback_model`` entries are appended afterwards unless they target the same
    provider/model/base_url route as an earlier entry. The returned list always contains fresh dict
    copies.
    """
    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key), config=config):
            identity = _entry_identity(entry)
            if identity not in seen:
                seen.add(identity)
                chain.append(entry)
    return chain


def scoped_fallback_chain(
    inherited: list[dict[str, Any]] | None, declared: Any, *, pinned: bool, owner: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Fallback chain for a route owner that can pin its own primary (delegated child, cron job).

    A pinned owner (explicit provider, endpoint or model) never borrows the *inherited* chain: the
    operator chose that route, and a chain entry is a different provider and usually a different
    model. Predictability beats liveness for an explicit pin. An unpinned owner inherits the chain
    when *declared* is absent/None. An explicit ``[]`` disables fallback either way; any other
    *declared* value is the owner's own chain, normalized by :func:`get_fallback_chain` (malformed
    entries are dropped; nothing usable left falls back to the pinned/inherited default).
    """
    default = None if pinned else (inherited or None)
    if declared is None:
        return default
    if declared == []:
        return None
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly() or {}
        except Exception as exc:
            logger.debug("Failed to load config for scoped_fallback_chain: %s", exc, exc_info=True)
            config = {}
    synth_cfg = {**config, "fallback_providers": declared}
    normalized = get_fallback_chain(synth_cfg)
    if not normalized:
        logger.warning("%s fallback_providers has no usable routes; using the %s default",
                       owner, "pinned" if pinned else "inherited")
    return normalized or default
