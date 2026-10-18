"""Builtin (non-plugin) platform adapter table — FILE_LINES sibling of the gateway run facade.

Moved from ``gateway/run.py``. ``_instantiate_builtin_adapter`` gained the lazy self-heal for
#136990: when an adapter's requirements probe fails at gateway start, an installer named in
``_BUILTIN_ADAPTER_INSTALLERS`` gets one attempt to pip-install the missing extra via
``pm.ensure_import`` and rebind the adapter module's availability flag before the adapter is
skipped — the same active/passive split the plugin registry got in #79812 (probes stay passive;
only gateway start may install).
"""

import logging
from typing import Any, Optional

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)


def _builtin_adapter_import(module: str, adapter_name: str, requirement: str):
    """Lazy-import ``(adapter_cls, requirements_ok)`` from ``gateway.platforms.<module>``."""
    import importlib
    mod = importlib.import_module(f"gateway.platforms.{module}")
    return getattr(mod, adapter_name), getattr(mod, requirement)


# platform -> (module, adapter class, requirements probe, warning on probe failure).
_BUILTIN_ADAPTERS: dict[Platform, tuple[str, str, str, str]] = {
    Platform.API_SERVER: ("api_server", "APIServerAdapter", "check_api_server_requirements",
                          "API Server: aiohttp not installed"),
    Platform.WEBHOOK: ("webhook", "WebhookAdapter", "check_webhook_requirements",
                       "Webhook: aiohttp not installed"),
    Platform.MSGRAPH_WEBHOOK: ("msgraph_webhook", "MSGraphWebhookAdapter", "check_msgraph_webhook_requirements",
                               "MSGraph webhook: aiohttp not installed")}

# ACTIVE installers (#136990): adapter modules whose missing dependency is a pip-installable
# extra they can heal themselves. Unlisted adapters keep the skip-with-warning behaviour.
_BUILTIN_ADAPTER_INSTALLERS: dict[Platform, str] = {
    Platform.API_SERVER: "ensure_api_server_requirements",
    Platform.WEBHOOK: "ensure_webhook_requirements",
}


def _instantiate_builtin_adapter(platform: Platform, config: Any) -> Optional[BasePlatformAdapter]:
    """Instantiate a core (non-plugin) adapter, or None when its requirements are unmet/unknown."""
    spec = _BUILTIN_ADAPTERS.get(platform)
    if spec is None:
        return None
    module, adapter_name, requirement, warning = spec
    adapter_cls, requirements_ok = _builtin_adapter_import(module, adapter_name, requirement)
    if not (requirements_ok() if callable(requirements_ok) else requirements_ok):
        installer = _BUILTIN_ADAPTER_INSTALLERS.get(platform)
        if installer is not None:
            import importlib
            mod = importlib.import_module(f"gateway.platforms.{module}")
            if getattr(mod, installer)():
                _, requirements_ok = _builtin_adapter_import(module, adapter_name, requirement)
    if not (requirements_ok() if callable(requirements_ok) else requirements_ok):
        logger.warning(warning)
        return None
    return adapter_cls(config)
