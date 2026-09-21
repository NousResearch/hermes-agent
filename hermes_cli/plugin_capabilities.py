"""Plugin capability declarations + consent state.

Every capability id maps 1:1 to a trust gate that already exists on the enforcing surface; ids
without an enforcing gate are deliberately never minted.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapabilitySpec:
    """One declarable capability and the legacy gate it maps to."""

    id: str
    legacy_path: Tuple[str, ...]  # deprecated boolean under plugins.entries.<id>, e.g. ("llm", "allow_model_override")
    description: str  # one-line risk description shown on the consent screen


# (id, legacy_path, description) — ONLY capabilities with an existing enforcing surface.
_CAPABILITY_ROWS = (
    ("tools.override", ("allow_tool_override",),
     "Replace built-in tools (e.g. shell_exec, write_file) — an "
     "override can intercept everything routed through that tool"),
    ("llm.provider_override", ("llm", "allow_provider_override"),
     "Run host-owned LLM calls against a provider other than your "
     "active one (uses your credentials)"),
    ("llm.model_override", ("llm", "allow_model_override"),
     "Choose which model host-owned LLM calls use (spend follows "
     "the chosen model)"),
    ("llm.agent_id_override", ("llm", "allow_agent_id_override"),
     "Attribute its LLM calls to a different agent id"),
    ("llm.profile_override", ("llm", "allow_profile_override"),
     "Run LLM calls under a different auth profile"),
    ("llm.task_override", ("llm", "allow_task_override"),
     "Route its LLM calls through the host's built-in auxiliary "
     "task lanes"),
    ("gateway.platform_actions", ("allow_platform_actions",),
     "Act on connected chat platforms as the gateway bot "
     "(add reactions, rename threads) via ctx.platform_actions"))
CAPABILITY_REGISTRY: Dict[str, CapabilitySpec] = {
    cid: CapabilitySpec(cid, path, desc) for cid, path, desc in _CAPABILITY_ROWS
}
VALID_CAPABILITY_IDS = frozenset(CAPABILITY_REGISTRY)

# Config keys under ``plugins.entries.<plugin_id>``.
GRANTED_KEY = "granted_capabilities"
CONSENT_KEY = "capabilities_consent"


def parse_declared_capabilities(raw: Any, plugin_name: str = "?") -> List[str]:
    """Normalize a manifest ``capabilities:`` value into known capability ids.

    Unknown ids are dropped with a warning: they can never be granted by this build, so hiding
    them from the consent screen is the fail-closed choice (the plugin must degrade gracefully).
    """
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            "Plugin %s: manifest 'capabilities' must be a list, got %s — ignoring",
            plugin_name, type(raw).__name__)
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            logger.warning("Plugin %s: ignoring non-string capability entry %r", plugin_name, item)
            continue
        cap = item.strip()
        if cap not in VALID_CAPABILITY_IDS:
            logger.warning(
                "Plugin %s: unknown capability %r (known: %s) — ignoring",
                plugin_name, cap, ", ".join(sorted(VALID_CAPABILITY_IDS)))
        elif cap not in out:
            out.append(cap)
    return out


def _known(capabilities: Iterable[str]) -> List[str]:
    """Deduplicated (order-preserving) subset of *capabilities* with a registry entry."""
    return [c for c in dict.fromkeys(capabilities) if c in VALID_CAPABILITY_IDS]


def capability_set_hash(capabilities: Iterable[str]) -> str:
    """Deterministic sha256 over a capability set (order-insensitive)."""
    canon = "\n".join(sorted(set(capabilities)))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ── Consent state (read side — fail closed on ANY error) ────────────────────────────────────

def _plugin_entry(plugin_id: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """``plugins.entries.<plugin_id>`` or ``{}`` — never raises (unreadable state = not granted)."""
    try:
        cfg: Any = config
        if cfg is None:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        entry = ((cfg.get("plugins") or {}).get("entries") or {}).get(plugin_id) or {}
        return entry if isinstance(entry, dict) else {}
    except Exception:
        return {}


def granted_capabilities(plugin_id: str, config: Optional[Mapping[str, Any]] = None) -> frozenset:
    """The set of capabilities the user has granted this plugin."""
    raw = _plugin_entry(plugin_id, config).get(GRANTED_KEY)
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(_known(c.strip() for c in raw if isinstance(c, str)))


def _legacy_gate_set(entry: Mapping[str, Any], spec: CapabilitySpec) -> bool:
    """True when the deprecated ``allow_*`` key for *spec* is truthy."""
    node: Any = entry
    for part in spec.legacy_path:
        if not isinstance(node, Mapping):
            return False
        node = node.get(part)
    return bool(node)


def plugin_capability_granted(plugin_id: str, capability: str, config: Optional[Mapping[str, Any]] = None) -> bool:
    """Canonical check: is *capability* live for *plugin_id*? True via ``granted_capabilities`` OR
    the deprecated-but-honored legacy ``allow_*`` key. Unknown ids / unreadable state -> False."""
    spec = CAPABILITY_REGISTRY.get(capability)
    if spec is None:
        logger.debug("capability check for unknown id %r (plugin %s) — denied", capability, plugin_id)
        return False
    entry = _plugin_entry(plugin_id, config)
    if capability in granted_capabilities(plugin_id, config={"plugins": {"entries": {plugin_id: entry}}}):
        allowed, evidence = True, "granted_capabilities"
    elif _legacy_gate_set(entry, spec):
        allowed, evidence = True, f"legacy key plugins.entries.{plugin_id}.{'.'.join(spec.legacy_path)} (deprecated)"
    else:
        allowed, evidence = False, "not granted"
    logger.info(  # audit trail for capability gate decisions
        "capability_check plugin=%s capability=%s decision=%s checked_by=plugin_capability_granted evidence=%s",
        plugin_id, capability, "allow" if allowed else "deny", evidence)
    return allowed


# ── Consent state (write side) ──────────────────────────────────────────────────────────────

def _child_dict(parent: dict, key: str) -> dict:
    """``parent[key]`` as a dict, replacing any non-dict value in place."""
    child = parent.setdefault(key, {})
    if not isinstance(child, dict):
        child = {}
        parent[key] = child
    return child


def _write_raw_config_values(updates: Mapping[Tuple[str, ...], Any]) -> None:
    """Change several settings in one locked, atomic raw-config mutation."""
    from hermes_cli import config as config_mod

    with config_mod._CONFIG_LOCK:
        if config_mod.is_managed():
            config_mod.managed_error("save configuration")
            return
        # Validate every destination before mutating the raw document so a later managed
        # leaf cannot leave an earlier capability or consent write behind.
        for path in updates:
            config_mod._exit_if_key_managed(".".join(path), "set")
        config_path = config_mod.get_config_path()
        raw = config_mod.require_readable_config_before_write(config_path)
        loaded = config_mod.load_config()
        for path, value in updates.items():
            entry = raw
            for segment in path[:-1]:
                entry = _child_dict(entry, segment)
            loaded_entry = loaded
            for segment in path:
                if not isinstance(loaded_entry, dict):
                    loaded_entry = None
                    break
                loaded_entry = loaded_entry.get(segment)
            entry[path[-1]] = config_mod._preserve_env_ref_templates(
                value, entry.get(path[-1]), loaded_entry)
        config_mod._write_user_config(config_path, raw)
        config_mod._secure_file(config_path)
        config_mod._RAW_CONFIG_CACHE.pop(str(config_path), None)
        config_mod._LOAD_CONFIG_CACHE.pop(str(config_path), None)
        config_mod._LAST_EXPANDED_CONFIG_BY_PATH[str(config_path)] = config_mod.load_config()


def _write_raw_config_value(path: Tuple[str, ...], value: Any) -> None:
    """Change one setting without canonicalizing unrelated user configuration."""
    _write_raw_config_values({path: value})


def record_consent(plugin_id: str, granted: Iterable[str], declared: Iterable[str]) -> None:
    """Persist a consent decision: ``granted_capabilities`` (union with prior grants), the consent
    record (hash of the declared set the user saw + UTC timestamp), and the legacy ``allow_*`` key
    for each grant so existing enforcement sites keep working unchanged."""
    from hermes_cli import config as config_mod

    with config_mod._CONFIG_LOCK:
        config = config_mod.load_config()
        entry = _child_dict(_child_dict(_child_dict(config, "plugins"), "entries"), plugin_id)
        previous = entry.get(GRANTED_KEY)
        merged = (list(previous) if isinstance(previous, list) else []) + _known(granted)
        granted_capabilities = sorted(_known(c for c in merged if isinstance(c, str)))
        consent = {
            "hash": capability_set_hash(_known(declared)),
            "granted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        updates: Dict[Tuple[str, ...], Any] = {
            ("plugins", "entries", plugin_id, GRANTED_KEY): granted_capabilities,
            ("plugins", "entries", plugin_id, CONSENT_KEY): consent,
        }
        # Bridge: mirror each grant into its legacy gate (enforcement sites still read allow_*).
        for cap in granted_capabilities:
            *parents, leaf = CAPABILITY_REGISTRY[cap].legacy_path
            updates[("plugins", "entries", plugin_id, *parents, leaf)] = True

        _write_raw_config_values(updates)

    logger.info(
        "capability_consent plugin=%s granted=%s declared_hash=%s", plugin_id,
        ",".join(granted_capabilities) or "(none)", consent["hash"][:12])


def consent_hash(plugin_id: str, config: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Return the stored consent hash, or None when absent/corrupt."""
    consent = _plugin_entry(plugin_id, config).get(CONSENT_KEY)
    h = consent.get("hash") if isinstance(consent, dict) else None
    return h if isinstance(h, str) and h else None


def pending_capabilities(
    plugin_id: str, declared: Iterable[str], config: Optional[Mapping[str, Any]] = None
) -> List[str]:
    """Declared-but-ungranted capabilities: everything at first consent, only the additions on an
    update re-consent (they must be re-consented before going live)."""
    granted = granted_capabilities(plugin_id, config)
    return [c for c in _known(declared) if c not in granted]


def declared_set_changed(
    plugin_id: str, declared: Iterable[str], config: Optional[Mapping[str, Any]] = None
) -> bool:
    """True when the declared set differs from what the user consented to (or never consented)."""
    stored = consent_hash(plugin_id, config)
    return stored is None or stored != capability_set_hash(_known(declared))
