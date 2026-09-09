"""
Compatibility Manifest for legacy Hermes modules and external plugins.
"""

from typing import Dict, Any


class CompatibilityManifest:
    """Tracks compatibility bridges and legacy module mappings."""

    COMPAT_MAPPINGS: Dict[str, str] = {
        "run_agent": "core.agent",
        "model_tools": "core.tools",
        "hermes_state": "core.memory",
    }

    @classmethod
    def get_target(cls, legacy_module: str) -> str:
        return cls.COMPAT_MAPPINGS.get(legacy_module, legacy_module)

    @classmethod
    def is_compatible(cls, module_name: str) -> bool:
        return module_name in cls.COMPAT_MAPPINGS or module_name.startswith("core.") or module_name.startswith("hermes_platform.")
