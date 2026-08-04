"""Import-safe gateway startup configuration bridges.

This module must stay free of gateway/agent imports.  ``gateway.__init__``
imports session code, which can reach ``agent.redact`` before ``gateway.run``
begins executing, so import-time settings for the redactor must be prepared
here first.
"""

from __future__ import annotations

import os
from pathlib import Path


_BOOTSTRAPPED = False


def bootstrap_gateway_redaction() -> None:
    """Load dotenv then export config fallback redaction settings before agent imports.

    The order intentionally matches the normal startup contract: dotenv values
    (including managed dotenv values) are resolved first, and an explicit
    ``security.redact_secrets`` and ``security.redact_level`` in config.yaml
    fill their respective values only when dotenv or the inherited environment
    did not provide them. ``agent.redact`` snapshots both environment variables
    at import time, so this must run before importing gateway session or runner
    modules.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.env_loader import load_hermes_dotenv

        hermes_home = get_hermes_home()
        load_hermes_dotenv(
            hermes_home=hermes_home,
            project_env=Path(__file__).resolve().parents[1] / ".env",
        )
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        config_path = hermes_home / "config.yaml"
        if not config_path.exists():
            return
        config = _expand_env_vars(read_user_config_raw(config_path))
        if not isinstance(config, dict):
            return
        try:
            from hermes_cli import managed_scope

            config = managed_scope.apply_managed_overlay(config)
        except Exception:
            pass
        security = config.get("security", {})
        if not isinstance(security, dict):
            return
        for config_key, environment_key in (
            ("redact_secrets", "HERMES_REDACT_SECRETS"),
            ("redact_level", "HERMES_REDACT_LEVEL"),
        ):
            if (
                environment_key not in os.environ
                and security.get(config_key) is not None
            ):
                os.environ[environment_key] = str(security[config_key]).lower()
    except Exception:
        # Preserve the existing fail-open startup behavior: malformed config or
        # an unavailable optional dependency leaves agent.redact at its default.
        pass
