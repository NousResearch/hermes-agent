"""Host-owned hydration and application of registered session boolean policies."""

from __future__ import annotations

import logging

from hermes_cli.config import load_config_readonly
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def policy_entries():
    from hermes_cli.plugins import _ensure_plugins_discovered
    return _ensure_plugins_discovered()._session_policies


class SessionPolicyRuntime:
    """Narrow engine capability; no prompt, history, secrets, or DB access."""

    def __init__(self, agent):
        self._agent = agent

    @property
    def checkpoint_required(self) -> bool:
        return getattr(self._agent, "compression_checkpoint_required", False) is True

    def set_context_option(self, name: str, value: bool) -> bool:
        # Lossy micro compaction lacks a pre-compression checkpoint hook.
        if name == "micro_compact_enabled" and self.checkpoint_required:
            value = False
        setter = getattr(getattr(self._agent, "context_compressor", None), "set_session_option", None)
        options = getattr(self._agent, "_session_context_options_supported", None)
        if not isinstance(options, dict):
            options = {}
            self._agent._session_context_options_supported = options
        options[name] = False
        if callable(setter):
            options[name] = setter(name, value) is True
        return options[name]


def _config_value(config, entry):
    value = config
    for part in entry["config_path"].split("."):
        if not isinstance(value, dict) or part not in value:
            return entry["default"]
        value = value[part]
    return is_truthy_value(value, default=entry["default"])


def apply_session_policy(agent, key, entry, override, config) -> bool:
    override = override if type(override) is bool else None
    global_value = _config_value(config, entry)
    agent._session_policy_overrides = dict(getattr(agent, "_session_policy_overrides", {}))
    agent._session_policy_overrides[key] = override
    agent._session_policy_global_values = dict(getattr(agent, "_session_policy_global_values", {}))
    agent._session_policy_global_values[key] = global_value
    model_config = getattr(agent, "_session_init_model_config", None)
    if not isinstance(model_config, dict):
        model_config = {}
        agent._session_init_model_config = model_config
    if override is None:
        model_config.pop(key, None)
    else:
        model_config[key] = override
    supported = False
    try:
        supported = entry["apply"](SessionPolicyRuntime(agent),
                                   global_value if override is None else override) is True
    except Exception:
        logger.warning("Session policy %s saved but engine application failed", key, exc_info=True)
    return supported


def hydrate_session_policies(agent, session_db=None, session_id=None, *, reset=False):
    entries = policy_entries()
    if not entries or getattr(agent, "_persist_disabled", False):
        return
    db = session_db if session_db is not None else getattr(agent, "_session_db", None)
    sid = session_id if session_id is not None else getattr(agent, "session_id", "")
    try:
        config = load_config_readonly()
    except Exception:
        logger.warning("Session policy config unavailable; using last known profile defaults", exc_info=True)
        config = _cached_config(agent, entries)
    try:
        row = None if reset or db is None or not sid else db.get_session(sid)
    except Exception:
        logger.warning("Session policy hydration unavailable for %s; inheriting", sid, exc_info=True)
        row = None
    from hermes_state import SessionDB
    for key, entry in entries.items():
        apply_session_policy(agent, key, entry, SessionDB.session_policy_override(row, key), config)


def refresh_session_policies(agent):
    if getattr(agent, "_persist_disabled", False):
        return
    entries = policy_entries()
    if not entries:
        return
    overrides = getattr(agent, "_session_policy_overrides", {})
    globals_ = getattr(agent, "_session_policy_global_values", {})
    try:
        config = load_config_readonly() if any(overrides.get(key) is None for key in entries) else None
    except Exception:
        logger.warning("Session policy refresh retaining last effective values", exc_info=True)
        return
    for key, entry in entries.items():
        if config is not None:
            apply_session_policy(agent, key, entry, overrides.get(key), config)
        else:
            try:
                entry["apply"](SessionPolicyRuntime(agent), overrides.get(key, globals_.get(key, entry["default"])))
            except Exception:
                logger.warning("Session policy %s refresh failed", key, exc_info=True)


def session_has_policy_overrides(row) -> bool:
    from hermes_state import SessionDB
    return any(SessionDB.session_policy_override(row, key) is not None for key in policy_entries())


def context_option_supported(agent, name) -> bool:
    return getattr(agent, "_session_context_options_supported", {}).get(name, True) is not False


def _cached_config(agent, entries):
    config = {}
    globals_ = getattr(agent, "_session_policy_global_values", {})
    for key, entry in entries.items():
        target = config
        parts = entry["config_path"].split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = globals_.get(key, entry["default"])
    return config
