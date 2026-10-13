"""Config-section shape guards: ``hermes config set`` must not overwrite a whole section with a scalar,
and a non-mapping ``agent:`` section read from config.yaml must not crash the loader.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); ``set_config_value`` late-imports
``_guard_section_overwrite``, and ``hermes_cli.config`` re-exports the ``agent`` helpers.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

logger = logging.getLogger("hermes_cli.config")


def _drop_non_mapping_agent(user_config: Any) -> Any:
    """Drop a non-mapping ``agent:`` (``agent: 5``, ``agent: true``) from a parsed config.yaml.

    ``_deep_merge`` lets any non-None scalar replace the default ``agent`` mapping, and the
    max_turns normalization plus every ``config["agent"].get(...)`` reader then raise, so no
    command (``hermes doctor`` and ``hermes config`` included) could start. Ignore it like an
    empty ``agent:`` (None), which keeps the defaults; anything else is returned unchanged."""
    if not isinstance(user_config, dict) or isinstance(user_config.get("agent"), (dict, type(None))):
        return user_config
    logger.warning("config.yaml: 'agent' must be a mapping, got %s; ignoring it and using the defaults",
                   type(user_config["agent"]).__name__)
    return {key: value for key, value in user_config.items() if key != "agent"}


def _normalize_max_turns_config(config: dict[str, Any]) -> dict[str, Any]:
    """Move legacy root-level ``max_turns`` under ``agent``; the schema default is injected only
    when the user set max_turns somewhere (so save_config can otherwise omit it). A non-mapping
    ``agent`` value is treated as empty rather than passed to ``dict()``."""
    config = dict(config)
    agent_section = config.get("agent")
    agent_config = dict(agent_section) if isinstance(agent_section, dict) else {}
    if "max_turns" in config and "max_turns" not in agent_config:
        agent_config["max_turns"] = config["max_turns"]
    if agent_config or "agent" in config:  # a sparse save must not grow an `agent: {}` section
        config["agent"] = agent_config
    config.pop("max_turns", None)
    return config


def _guard_section_overwrite(key: str, value: Any, user_config: dict[str, Any], force: bool) -> str:
    """Refuse (or with ``force`` allow) a single-segment key overwriting a mapping with a scalar.
    Bare ``model`` is a documented shorthand — redirected to ``model.default`` so siblings survive;
    a list (or a mapping over an existing section) under it is refused without ``force``.
    Returns the (possibly redirected) key."""
    from hermes_cli.config import _exit_invalid
    existing = user_config.get(key)
    kind = "mapping" if isinstance(value, dict) else "list"
    # Containers under the model-id shorthand have no reader slot (#131435).
    if key == "model" and not force and (
            isinstance(value, list) or (isinstance(value, dict) and isinstance(existing, dict))):
        _exit_invalid(
            f"✗ Cannot set 'model' to a {kind} — "
            "the bare 'model' shorthand takes a model id, and a container value has no "
            "slot there.\n"
            "  Set the section's keys individually instead:\n"
            "    hermes config set model.provider <provider>\n"
            "    hermes config set model.default <model-id>\n"
            "  Or replace the whole section deliberately:\n"
            "    hermes config set --force model '{provider: <provider>, default: <model-id>}'")
    if "." in key or not isinstance(existing, dict):
        return key
    if key == "model":
        if force:
            what = f"the given {kind}" if isinstance(value, (dict, list)) else "a scalar"
            print(
                f"⚠ Replacing entire 'model' section with {what} "
                f"(discarding {len(existing)} existing sub-key(s))")
            return key
        print(
            f"✓ Redirecting bare 'model' to 'model.default' "
            f"(preserving {len(existing)} existing model sub-key(s))")
        return "model.default"
    if force:
        return key
    sub = [k for k in existing if isinstance(k, str)]
    err = [
        f"✗ Cannot set '{key}' to a scalar — '{key}' is a "
        f"configuration section with {len(sub)} sub-key(s)."]
    if sub:
        err.append(f"  Sub-keys: {', '.join(sub[:8])}")
        if len(sub) > 8:
            err.append(f"  ... and {len(sub) - 8} more")
    err += [
        "  Use a dotted path to set a specific leaf key:",
        f"    hermes config set {key}.<sub-key> <value>",
        "  Or use --force to replace the entire section:",
        f"    hermes config set --force {key} {value!r}"]
    print("\n".join(err), file=sys.stderr)
    sys.exit(1)
