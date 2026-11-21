"""Which deferred platform adapters a gateway config/env pass must import.

Bundled platform plugins register lazily (``PlatformRegistry.register_deferred``); importing all
~20 of them costs every cold gateway start about half a second (discord, telegram, google_chat
SDKs, telegram's i18n catalog) although a fresh profile configures none. The config passes in
``gateway.config_loader`` / ``gateway.config_env`` only need a platform's hooks
(``apply_yaml_config_fn``, ``env_enablement_fn``, ``is_connected``) when the profile could be
configuring that platform, so they materialize only names with *activation evidence*:

* a config block for it (top-level ``<name>:``, ``platforms.<name>``, ``gateway.platforms.<name>``,
  ``gateway.<name>``, or a legacy ``gateway.json`` row) — including a sibling block such as
  ``wecom_callback`` (registered by the ``wecom`` plugin);
* an env var (process env, the profile ``.env``, or the active secret scope) under the platform's
  prefix (``TELEGRAM_*``, ``GOOGLE_CHAT_*``) or named in its manifest ``requires_env``
  (``TWILIO_*`` for sms);
* a credential-pool record for it in the profile ``auth.json`` (Photon stores project creds there).

The rule over-approximates on purpose: a false positive costs one import, a false negative would
leave a configured platform down. Any lookup by name (``platform_registry.get``) still imports a
deferred adapter on demand.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


def _yaml_block_names(yaml_cfg: Any) -> set[str]:
    """Keys of every mapping a platform block may live in (``platform_section`` /
    ``merge_platform_sections`` read the same places): ``<name>:``, ``platforms.<name>``,
    ``gateway.<name>``, ``gateway.platforms.<name>``."""
    if not isinstance(yaml_cfg, dict):
        return set()
    gateway: dict = yaml_cfg["gateway"] if isinstance(yaml_cfg.get("gateway"), dict) else {}
    parents = (yaml_cfg, yaml_cfg.get("platforms"), gateway, gateway.get("platforms"))
    return {str(k) for parent in parents if isinstance(parent, dict)
            for k, v in parent.items() if isinstance(v, dict)}


def _env_names(home: Optional[Path]) -> set[str]:
    names = {k for k, v in os.environ.items() if str(v).strip()}
    try:
        from agent.secret_scope import current_secret_scope, load_env_file
        scope = current_secret_scope()
        if scope is not None:
            names.update(k for k, v in scope.items() if str(v).strip())
        if home is not None:
            names.update(k for k, v in load_env_file(home / ".env").items() if str(v).strip())
            # Values an external secret source (Bitwarden, 1Password) resolved for this home, and
            # the administrator-managed .env: both feed the profile scope the adapters read.
            from hermes_cli.env_loader import get_secret_source_values
            names.update(k for k, v in get_secret_source_values(home).items() if str(v).strip())
        from hermes_cli.managed_scope import load_managed_env
        names.update(k for k, v in load_managed_env().items() if str(v).strip())
    except Exception:
        logger.debug("platform activation: env scan degraded", exc_info=True)
    return names


def _credential_pool_names(home: Optional[Path]) -> set[str]:
    if home is None:
        return set()
    try:
        with open(home / "auth.json", encoding="utf-8-sig") as handle:
            pool = (json.load(handle) or {}).get("credential_pool") or {}
        return {str(k) for k in pool} if isinstance(pool, dict) else set()
    except FileNotFoundError:
        return set()
    except Exception:
        logger.debug("platform activation: auth.json unreadable", exc_info=True)
        return set()


def _matches(name: str, candidates: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(c.lower() == lowered or c.lower().startswith(lowered + "_") for c in candidates)


def _legacy_gateway_json_names(home: Path) -> set[str]:
    try:
        with open(home / "gateway.json", encoding="utf-8-sig") as handle:
            rows = (json.load(handle) or {}).get("platforms") or {}
        return {str(k) for k in rows} if isinstance(rows, dict) else set()
    except FileNotFoundError:
        return set()
    except (OSError, ValueError, AttributeError):
        # Unreadable or not a {"platforms": {...}} document: the YAML/env/pool evidence still applies.
        logger.debug("platform activation: gateway.json unreadable", exc_info=True)
        return set()


def activation_predicate(
    *, yaml_cfg: Any = None, platform_names: Iterable[str] = (), home: Optional[Path] = None,
    registry: Any = None,
) -> Callable[[str], bool]:
    """``name -> bool``: does this profile show evidence of configuring platform *name*?

    *yaml_cfg* is the loaded config.yaml layer (read from *home* when omitted), *platform_names*
    any already-merged platform keys (``GatewayConfig.platforms``). *home* defaults to the active
    Hermes home.
    """
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    if yaml_cfg is None and home is not None:
        try:
            from gateway.config_loader import read_yaml_layers
            yaml_cfg = read_yaml_layers(home)
        except Exception:
            logger.debug("platform activation: config.yaml unreadable", exc_info=True)
    blocks = _yaml_block_names(yaml_cfg) | {str(getattr(n, "value", n)) for n in platform_names}
    if home is not None:
        blocks |= _legacy_gateway_json_names(home)
    env = _env_names(home)
    upper_env = {e.upper() for e in env}
    pools = _credential_pool_names(home)

    def has_evidence(name: str) -> bool:
        if _matches(name, blocks) or _matches(name, pools):
            return True
        prefix = name.upper().replace("-", "_") + "_"
        if any(e.startswith(prefix) for e in upper_env):
            return True
        declared = registry.activation_env(name) if registry is not None else frozenset()
        return any(d.upper() in upper_env for d in declared)

    return has_evidence


def configured_plugin_entries(registry: Any, **evidence: Any) -> list:
    """Plugin entries for platforms this profile configures (see module docstring); already-loaded
    entries are always included. Never imports an adapter with no activation evidence."""
    if registry is None:
        return []
    wanted = activation_predicate(registry=registry, **evidence)
    return [e for e in registry.configured_entries(wanted) if e.source == "plugin"]
