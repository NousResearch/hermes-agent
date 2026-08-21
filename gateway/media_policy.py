"""Shared config→env bridge for media-delivery policy.

``validate_media_delivery_path`` (gateway/platforms/base.py) reads its policy
from environment variables:

  - ``HERMES_MEDIA_DELIVERY_STRICT``    <- gateway.strict
  - ``HERMES_MEDIA_ALLOW_DIRS``         <- gateway.media_delivery_allow_dirs
  - ``HERMES_MEDIA_TRUST_RECENT_FILES`` <- gateway.trust_recent_files

Historically the config.yaml -> env translation ran ONLY in gateway startup
(gateway/run.py), so any process that delivers media without booting the
gateway — a manual ``hermes cron run`` in the CLI, ``hermes send``, a
standalone cron tick — filtered MEDIA paths under DIFFERENT policy than the
gateway's scheduled deliveries. In strict/allowlisted enterprise deployments
that divergence silently dropped attachments from manual cron runs while
scheduled runs delivered them (text is unaffected — only media goes through
path validation).

``apply_media_policy_env()`` is that same translation as a shared, idempotent
helper. Gateway startup calls it, and every standalone delivery entrypoint
calls it immediately before filtering media paths.

Precedence: an explicitly-set environment variable WINS over config.yaml.
This preserves both the operator contract (env overrides are how deployments
pin behavior) and gateway/run.py's historical shape (it only wrote the env
var when the config key was present; we additionally refuse to overwrite a
pre-existing env value so a shell-exported override survives).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_STRICT_ENV = "HERMES_MEDIA_DELIVERY_STRICT"
_ALLOW_DIRS_ENV = "HERMES_MEDIA_ALLOW_DIRS"
_TRUST_RECENT_ENV = "HERMES_MEDIA_TRUST_RECENT_FILES"


def _load_gateway_cfg(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config() or {}
        except Exception:
            return {}
    gateway_cfg = config.get("gateway", {})
    return gateway_cfg if isinstance(gateway_cfg, dict) else {}


def apply_media_policy_env(config: Optional[Dict[str, Any]] = None) -> None:
    """Bridge gateway media-policy settings from config.yaml into the env.

    Idempotent and env-wins: a variable already present in the environment is
    never overwritten, so gateway startup (which runs this same helper) and
    operator shell exports keep precedence. Never raises — a policy-bridge
    failure must not break delivery; the validator falls back to its
    defaults exactly as before.
    """
    try:
        gateway_cfg = _load_gateway_cfg(config)
        if not gateway_cfg:
            return

        strict = gateway_cfg.get("strict")
        if strict is not None and not os.environ.get(_STRICT_ENV):
            os.environ[_STRICT_ENV] = "1" if strict else "0"

        allow_dirs = gateway_cfg.get("media_delivery_allow_dirs")
        if allow_dirs and not os.environ.get(_ALLOW_DIRS_ENV):
            if isinstance(allow_dirs, str):
                allow_dirs_str = allow_dirs
            elif isinstance(allow_dirs, (list, tuple)):
                allow_dirs_str = os.pathsep.join(str(p) for p in allow_dirs if p)
            else:
                allow_dirs_str = ""
            if allow_dirs_str:
                os.environ[_ALLOW_DIRS_ENV] = allow_dirs_str

        trust_recent = gateway_cfg.get("trust_recent_files")
        if trust_recent is not None and not os.environ.get(_TRUST_RECENT_ENV):
            os.environ[_TRUST_RECENT_ENV] = "1" if trust_recent else "0"
    except Exception:  # noqa: BLE001 - policy bridge must never break delivery
        logger.debug("apply_media_policy_env failed", exc_info=True)
