"""Shared config→env bridge for media-delivery policy.

``validate_media_delivery_path`` reads ``HERMES_MEDIA_DELIVERY_STRICT`` (gateway.strict),
``HERMES_MEDIA_ALLOW_DIRS`` (gateway.media_delivery_allow_dirs) and
``HERMES_MEDIA_TRUST_RECENT_FILES`` (gateway.trust_recent_files).  Every delivery
entrypoint (gateway startup, ``hermes cron run``, ``hermes send``) calls
:func:`apply_media_policy_env` first so standalone paths filter under the gateway's
policy instead of silently dropping attachments in strict/allowlisted deployments.
An explicitly-set env var WINS over config.yaml, so shell overrides survive.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_FLAG_ENVS = (("strict", "HERMES_MEDIA_DELIVERY_STRICT"), ("trust_recent_files", "HERMES_MEDIA_TRUST_RECENT_FILES"))
_ALLOW_DIRS_ENV = "HERMES_MEDIA_ALLOW_DIRS"


def _load_gateway_cfg(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config() or {}
        except Exception:
            return {}
    gateway_cfg = config.get("gateway", {})
    return gateway_cfg if isinstance(gateway_cfg, dict) else {}


def _set_env_default(env: str, value: str) -> None:
    """Set ``env`` only when unset/empty and ``value`` is non-empty (env wins)."""
    if value and not os.environ.get(env):
        os.environ[env] = value


def _allow_dirs_str(allow_dirs: Any) -> str:
    if isinstance(allow_dirs, (list, tuple)):
        return os.pathsep.join(str(p) for p in allow_dirs if p)
    return allow_dirs if isinstance(allow_dirs, str) else ""


def apply_media_policy_env(config: Optional[Dict[str, Any]] = None) -> None:
    """Bridge gateway media-policy settings from config.yaml into the env.  Idempotent,
    env-wins, never raises — a bridge failure must not break delivery (validator defaults apply)."""
    try:
        gateway_cfg = _load_gateway_cfg(config)
        if not gateway_cfg:
            return
        for key, env in _FLAG_ENVS:
            flag = gateway_cfg.get(key)
            if flag is not None:
                _set_env_default(env, "1" if flag else "0")
        allow_dirs = gateway_cfg.get("media_delivery_allow_dirs")
        if allow_dirs:
            _set_env_default(_ALLOW_DIRS_ENV, _allow_dirs_str(allow_dirs))
    except Exception:  # noqa: BLE001 - policy bridge must never break delivery
        logger.debug("apply_media_policy_env failed", exc_info=True)
