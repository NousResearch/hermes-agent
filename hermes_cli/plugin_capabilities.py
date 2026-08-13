"""Plugin capability declarations + consent state (#64228).

Unifies the scattered per-plugin trust gates (``plugins.entries.<id>.allow_*``)
into one declared, diffable **capability model** with install/update-time
consent.

**This is NOT a sandbox.** In-process Python plugins remain trusted code — a
malicious plugin can import anything, monkey-patch core, and ignore all of
this. Capabilities govern the *host API surfaces* Hermes hands out (which
registrations succeed, which ``ctx`` methods are live) and give the user an
honest consent + audit trail. Actual isolation is a separate research track.

Canonical registry
------------------
Every capability id maps 1:1 to a trust gate that **already exists** on the
enforcing surface. We deliberately do not mint capability ids without an
enforcing gate:

===========================  ==================================================
Capability id                Legacy config gate (``plugins.entries.<id>.…``)
===========================  ==================================================
``tools.override``           ``allow_tool_override``
``llm.provider_override``    ``llm.allow_provider_override``
``llm.model_override``       ``llm.allow_model_override``
``llm.agent_id_override``    ``llm.allow_agent_id_override``
``llm.profile_override``     ``llm.allow_profile_override``
``llm.task_override``        ``llm.allow_task_override``
===========================  ==================================================

The legacy ``allow_*`` keys keep working verbatim (deprecated but honored):
a gate is open when the legacy key is true **or** the capability is granted.

Consent state
-------------
Stored under the plugin's config entry::

    plugins:
      entries:
        <plugin_id>:
          granted_capabilities: [tools.override]
          capabilities_consent:
            hash: "<sha256 of the declared capability set at consent time>"
            granted_at: "2026-08-12T00:00:00+00:00"

The hash records *what the user saw* when they consented. When an update
declares capabilities whose set hash differs, the additions stay ungranted
until the user re-consents (``hermes plugins update`` surfaces the diff).

Ground rule: everything defaults OFF. Any failure to read consent state
(missing config, corrupt YAML, wrong types) means **not granted**.
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
    # Path of the deprecated boolean under ``plugins.entries.<plugin_id>``,
    # e.g. ("allow_tool_override",) or ("llm", "allow_model_override").
    legacy_path: Tuple[str, ...]
    # One-line risk description shown on the consent screen.
    description: str


# Canonical registry — ONLY capabilities with an existing enforcing surface.
CAPABILITY_REGISTRY: Dict[str, CapabilitySpec] = {
    spec.id: spec
    for spec in (
        CapabilitySpec(
            id="tools.override",
            legacy_path=("allow_tool_override",),
            description=(
                "Replace built-in tools (e.g. shell_exec, write_file) — an "
                "override can intercept everything routed through that tool"
            ),
        ),
        CapabilitySpec(
            id="llm.provider_override",
            legacy_path=("llm", "allow_provider_override"),
            description=(
                "Run host-owned LLM calls against a provider other than your "
                "active one (uses your credentials)"
            ),
        ),
        CapabilitySpec(
            id="llm.model_override",
            legacy_path=("llm", "allow_model_override"),
            description=(
                "Choose which model host-owned LLM calls use (spend follows "
                "the chosen model)"
            ),
        ),
        CapabilitySpec(
            id="llm.agent_id_override",
            legacy_path=("llm", "allow_agent_id_override"),
            description="Attribute its LLM calls to a different agent id",
        ),
        CapabilitySpec(
            id="llm.profile_override",
            legacy_path=("llm", "allow_profile_override"),
            description="Run LLM calls under a different auth profile",
        ),
        CapabilitySpec(
            id="llm.task_override",
            legacy_path=("llm", "allow_task_override"),
            description=(
                "Route its LLM calls through the host's built-in auxiliary "
                "task lanes"
            ),
        ),
    )
}

VALID_CAPABILITY_IDS = frozenset(CAPABILITY_REGISTRY)

# Config keys under ``plugins.entries.<plugin_id>``.
GRANTED_KEY = "granted_capabilities"
CONSENT_KEY = "capabilities_consent"


# ---------------------------------------------------------------------------
# Declaration parsing
# ---------------------------------------------------------------------------

def parse_declared_capabilities(raw: Any, plugin_name: str = "?") -> List[str]:
    """Normalize a manifest ``capabilities:`` value into known capability ids.

    Unknown ids are dropped with a warning (forward compat: a plugin built
    for a newer Hermes may declare ids this build doesn't know; they can
    never be granted here, so hiding them from the consent screen is the
    fail-closed choice — the plugin must degrade gracefully).
    """
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            "Plugin %s: manifest 'capabilities' must be a list, got %s — ignoring",
            plugin_name, type(raw).__name__,
        )
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            logger.warning(
                "Plugin %s: ignoring non-string capability entry %r",
                plugin_name, item,
            )
            continue
        cap = item.strip()
        if cap in VALID_CAPABILITY_IDS:
            if cap not in out:
                out.append(cap)
        else:
            logger.warning(
                "Plugin %s: unknown capability %r (known: %s) — ignoring",
                plugin_name, cap, ", ".join(sorted(VALID_CAPABILITY_IDS)),
            )
    return out


def capability_set_hash(capabilities: Iterable[str]) -> str:
    """Deterministic sha256 over a capability set (order-insensitive)."""
    canon = "\n".join(sorted(set(capabilities)))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Consent state (read side — fail closed on ANY error)
# ---------------------------------------------------------------------------

def _plugin_entry(plugin_id: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """Return ``plugins.entries.<plugin_id>`` or ``{}`` — never raises."""
    try:
        cfg: Any = config
        if cfg is None:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get(plugin_id) or {}
        return entry if isinstance(entry, dict) else {}
    except Exception:
        # Ground rule: failure to read consent state = not granted.
        return {}


def granted_capabilities(
    plugin_id: str, config: Optional[Mapping[str, Any]] = None
) -> frozenset:
    """Return the set of capabilities the user has granted this plugin.

    Fail-closed: missing/corrupt state yields the empty set.
    """
    entry = _plugin_entry(plugin_id, config)
    raw = entry.get(GRANTED_KEY)
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(
        c.strip() for c in raw
        if isinstance(c, str) and c.strip() in VALID_CAPABILITY_IDS
    )


def _legacy_gate_set(entry: Mapping[str, Any], spec: CapabilitySpec) -> bool:
    """True when the deprecated ``allow_*`` key for *spec* is truthy."""
    node: Any = entry
    for part in spec.legacy_path:
        if not isinstance(node, Mapping):
            return False
        node = node.get(part)
    return bool(node) and node is not None


def plugin_capability_granted(
    plugin_id: str,
    capability: str,
    config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Canonical check: is *capability* live for *plugin_id*?

    True when EITHER:

    * the capability appears in ``granted_capabilities`` (consent flow), OR
    * the legacy ``allow_*`` config key is set (deprecated, still honored so
      existing configs keep working).

    Unknown capability ids and any failure to read state return ``False``
    (ground rule 4: fail closed).
    """
    spec = CAPABILITY_REGISTRY.get(capability)
    if spec is None:
        logger.debug(
            "capability check for unknown id %r (plugin %s) — denied",
            capability, plugin_id,
        )
        return False
    entry = _plugin_entry(plugin_id, config)
    if capability in granted_capabilities(plugin_id, config={"plugins": {"entries": {plugin_id: entry}}}):
        _log_capability_decision(plugin_id, capability, True, "granted_capabilities")
        return True
    if _legacy_gate_set(entry, spec):
        _log_capability_decision(
            plugin_id, capability, True,
            f"legacy key plugins.entries.{plugin_id}.{'.'.join(spec.legacy_path)} (deprecated)",
        )
        return True
    _log_capability_decision(plugin_id, capability, False, "not granted")
    return False


def _log_capability_decision(
    plugin_id: str, capability: str, allowed: bool, evidence: str
) -> None:
    """Audit line for capability gate decisions (the ``checked_by`` trail)."""
    logger.info(
        "capability_check plugin=%s capability=%s decision=%s checked_by=plugin_capability_granted evidence=%s",
        plugin_id, capability, "allow" if allowed else "deny", evidence,
    )


# ---------------------------------------------------------------------------
# Consent state (write side)
# ---------------------------------------------------------------------------

def record_consent(
    plugin_id: str,
    granted: Iterable[str],
    declared: Iterable[str],
) -> None:
    """Persist a consent decision for *plugin_id*.

    Writes ``granted_capabilities`` (union with any previously granted set),
    the consent record (hash of the *declared* set the user saw + UTC
    timestamp), and — so every existing enforcement site keeps working
    without changes — the corresponding legacy ``allow_*`` keys for each
    newly granted capability.
    """
    from hermes_cli.config import load_config, save_config

    granted_list = [c for c in dict.fromkeys(granted) if c in VALID_CAPABILITY_IDS]
    declared_list = [c for c in dict.fromkeys(declared) if c in VALID_CAPABILITY_IDS]

    config = load_config()
    plugins_cfg = config.setdefault("plugins", {})
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
        config["plugins"] = plugins_cfg
    entries = plugins_cfg.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins_cfg["entries"] = entries
    entry = entries.setdefault(plugin_id, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[plugin_id] = entry

    previous = entry.get(GRANTED_KEY)
    merged = list(previous) if isinstance(previous, list) else []
    for cap in granted_list:
        if cap not in merged:
            merged.append(cap)
    entry[GRANTED_KEY] = sorted(
        c for c in dict.fromkeys(merged)
        if isinstance(c, str) and c in VALID_CAPABILITY_IDS
    )
    entry[CONSENT_KEY] = {
        "hash": capability_set_hash(declared_list),
        "granted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Bridge: mirror each granted capability into its legacy gate so the
    # existing enforcement sites (which still read allow_*) honor the grant.
    for cap in entry[GRANTED_KEY]:
        spec = CAPABILITY_REGISTRY[cap]
        node = entry
        for part in spec.legacy_path[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[spec.legacy_path[-1]] = True

    save_config(config)
    logger.info(
        "capability_consent plugin=%s granted=%s declared_hash=%s",
        plugin_id, ",".join(entry[GRANTED_KEY]) or "(none)",
        entry[CONSENT_KEY]["hash"][:12],
    )


def consent_hash(plugin_id: str, config: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Return the stored consent hash, or None when absent/corrupt."""
    entry = _plugin_entry(plugin_id, config)
    consent = entry.get(CONSENT_KEY)
    if not isinstance(consent, dict):
        return None
    h = consent.get("hash")
    return h if isinstance(h, str) and h else None


def pending_capabilities(
    plugin_id: str,
    declared: Iterable[str],
    config: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Capabilities declared by the plugin but not yet granted.

    Used both at first consent (everything is pending) and on update
    re-consent: when a new version declares capabilities the granted set
    lacks, those additions are returned and must be re-consented before
    they go live. The stored consent hash tells whether the *declared* set
    changed since the user last saw it.
    """
    declared_list = [c for c in dict.fromkeys(declared) if c in VALID_CAPABILITY_IDS]
    granted = granted_capabilities(plugin_id, config)
    return [c for c in declared_list if c not in granted]


def declared_set_changed(
    plugin_id: str,
    declared: Iterable[str],
    config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when the declared set differs from what the user consented to.

    No stored consent at all counts as changed (never consented).
    """
    stored = consent_hash(plugin_id, config)
    if stored is None:
        return True
    return stored != capability_set_hash(
        c for c in declared if c in VALID_CAPABILITY_IDS
    )
