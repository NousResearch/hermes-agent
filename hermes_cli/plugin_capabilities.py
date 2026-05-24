"""Plugin capability-based permissions system.

Defines a capability model for plugins so they can declare what resources
they need access to, and the framework enforces those boundaries at
runtime.

Capability Model
----------------
Each plugin declares its required capabilities in its ``plugin.yaml``
manifest:

```yaml
capabilities:
  - tool_register        # can register new tools
  - filesystem_read      # can read files
  - filesystem_write     # can write files (requires user approval)
  - network              # can make outbound HTTP requests
  - subprocess           # can spawn subprocesses
  - env_read             # can read environment variables
  - config_read          # can read config.yaml
```

Enforcement is applied in ``hermes_cli/plugins.py`` at load time.
Plugins without a declared capability cannot perform the associated
action — the framework raises ``CapabilityDenied``.

Security
--------
- Capabilities are **opt-in**: a plugin with no declared capabilities
  gets zero permissions.
- ``filesystem_write`` and ``subprocess`` require explicit user approval
  on first use (persisted per-plugin).
- The ``--yolo`` flag does NOT bypass capability checks — this is a
  security boundary, not a UX preference.
"""

import logging
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Capability(Enum):
    """Resource capabilities a plugin may request."""
    TOOL_REGISTER = auto()       # Register new tools via ctx.register_tool
    FILESYSTEM_READ = auto()     # Read files from the filesystem
    FILESYSTEM_WRITE = auto()    # Write/modify files (requires approval)
    NETWORK = auto()             # Make outbound HTTP/HTTPS requests
    SUBPROCESS = auto()          # Spawn subprocesses
    ENV_READ = auto()            # Read environment variables
    CONFIG_READ = auto()         # Read config.yaml settings
    MEMORY_READ = auto()         # Read memory/provider state
    MEMORY_WRITE = auto()        # Write memory entries
    HOOK_REGISTER = auto()       # Register lifecycle hooks
    CLI_COMMAND = auto()         # Register CLI subcommands


# Human-readable descriptions for error messages
_CAPABILITY_DESCRIPTIONS: dict[Capability, str] = {
    Capability.TOOL_REGISTER: "register new tools",
    Capability.FILESYSTEM_READ: "read files from the filesystem",
    Capability.FILESYSTEM_WRITE: "write or modify files",
    Capability.NETWORK: "make outbound network requests",
    Capability.SUBPROCESS: "spawn subprocesses",
    Capability.ENV_READ: "read environment variables",
    Capability.CONFIG_READ: "read configuration settings",
    Capability.MEMORY_READ: "read memory state",
    Capability.MEMORY_WRITE: "write memory entries",
    Capability.HOOK_REGISTER: "register lifecycle hooks",
    Capability.CLI_COMMAND: "register CLI subcommands",
}

# Capabilities that require user approval on first use
_REQUIRES_APPROVAL = {
    Capability.FILESYSTEM_WRITE,
    Capability.SUBPROCESS,
    Capability.MEMORY_WRITE,
}


class CapabilityDenied(Exception):
    """Raised when a plugin attempts an action without the required capability."""

    def __init__(self, plugin_name: str, capability: Capability, action: str = ""):
        self.plugin_name = plugin_name
        self.capability = capability
        self.action = action or _CAPABILITY_DESCRIPTIONS.get(capability, "perform this action")
        super().__init__(
            f"Plugin '{plugin_name}' denied: requires '{capability.name}' "
            f"capability to {self.action}. Add it to the plugin's "
            f"'capabilities' list in plugin.yaml."
        )


class PluginCapabilities:
    """Tracks and enforces capabilities for a loaded plugin."""

    def __init__(self, plugin_name: str, declared: list[str]):
        self.plugin_name = plugin_name
        self._capabilities: set[Capability] = set()
        self._approval_granted: set[Capability] = set()
        self._parse_declared(declared)

    def _parse_declared(self, declared: list[str]) -> None:
        """Parse capability strings into Capability enums."""
        for cap_name in declared:
            cap_name_upper = cap_name.upper().replace("-", "_")
            try:
                cap = Capability[cap_name_upper]
                self._capabilities.add(cap)
            except KeyError:
                logger.warning(
                    "Plugin '%s' declared unknown capability '%s' — ignored",
                    self.plugin_name, cap_name,
                )

    def has(self, capability: Capability) -> bool:
        """Check if the plugin has the given capability."""
        return capability in self._capabilities

    def check(self, capability: Capability, *, require_approval: bool = True) -> None:
        """Enforce a capability check.

        Parameters
        ----------
        capability:
            The capability to check.
        require_approval:
            If True and the capability requires approval, check that
            approval was granted.

        Raises
        ------
        CapabilityDenied
            If the plugin does not have the required capability.
        """
        if not self.has(capability):
            raise CapabilityDenied(self.plugin_name, capability)

        if require_approval and capability in _REQUIRES_APPROVAL:
            if capability not in self._approval_granted:
                raise CapabilityDenied(
                    self.plugin_name, capability,
                    f"{_CAPABILITY_DESCRIPTIONS.get(capability, 'this action')} "
                    "(requires user approval)"
                )

    def grant_approval(self, capability: Capability) -> None:
        """Grant approval for a capability that requires it."""
        if capability in _REQUIRES_APPROVAL:
            self._approval_granted.add(capability)
            logger.info(
                "Plugin '%s' granted approval for '%s'",
                self.plugin_name, capability.name,
            )

    def get_capabilities(self) -> list[str]:
        """Return the list of declared capability names."""
        return [c.name.lower().replace("_", "-") for c in sorted(self._capabilities, key=lambda c: c.value)]

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation."""
        return {
            "plugin": self.plugin_name,
            "capabilities": self.get_capabilities(),
            "approval_granted": [c.name for c in self._approval_granted],
        }


def parse_capabilities_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Extract capabilities from a plugin manifest dict."""
    caps = manifest.get("capabilities", [])
    if isinstance(caps, list):
        return [str(c) for c in caps]
    if isinstance(caps, str):
        return [c.strip() for c in caps.split(",") if c.strip()]
    return []


def create_capabilities(
    plugin_name: str, manifest: dict[str, Any]
) -> PluginCapabilities:
    """Create a PluginCapabilities instance from a plugin manifest."""
    declared = parse_capabilities_from_manifest(manifest)
    return PluginCapabilities(plugin_name, declared)
