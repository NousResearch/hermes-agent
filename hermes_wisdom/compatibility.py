"""Deterministic local compatibility evaluation for Wisdom packages."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from packaging.version import InvalidVersion, Version

from hermes_cli import __version__ as HERMES_VERSION

from .contract import SystemSpecification


CompatibilityOutcome = Literal[
    "compatible", "compatible_after_setup", "partial", "blocked_pending_action"
]


@dataclass(frozen=True)
class LocalCapabilities:
    hermes_version: str
    os: str
    architecture: str
    model_capabilities: frozenset[str] = frozenset()
    context_window: int | None = None
    enabled_tools: dict[str, str | None] = field(default_factory=dict)
    plugins: dict[str, str | None] = field(default_factory=dict)
    credentials: frozenset[str] = frozenset()
    connections: frozenset[str] = frozenset()
    filesystem_readable: frozenset[str] = frozenset()
    filesystem_writable: frozenset[str] = frozenset()
    network_access: bool | None = None
    runtime: dict[str, bool] = field(default_factory=dict)
    hardware: frozenset[str] = frozenset()
    is_admin: bool = False


@dataclass(frozen=True)
class CompatibilityResult:
    outcome: CompatibilityOutcome
    satisfied: tuple[str, ...]
    setup_actions: tuple[str, ...]
    limitations: tuple[str, ...]
    blocked: tuple[str, ...]


def _normalized_os(value: str) -> str:
    lowered = value.strip().lower()
    return {"macos": "darwin", "mac": "darwin", "win32": "windows"}.get(
        lowered, lowered
    )


def _normalized_architecture(value: str) -> str:
    lowered = value.strip().lower()
    return {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
    }.get(lowered, lowered)


def _distribution_version(name: str) -> str | None:
    for candidate in {name, name.replace("_", "-"), name.rsplit("/", 1)[-1]}:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _path_capabilities(paths: list[str]) -> tuple[frozenset[str], frozenset[str]]:
    readable: set[str] = set()
    writable: set[str] = set()
    for raw in paths:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if "$" in expanded:
            continue
        candidate = Path(expanded)
        if candidate.exists() and os.access(candidate, os.R_OK):
            readable.add(raw)
        writable_target = candidate
        while (
            not writable_target.exists() and writable_target != writable_target.parent
        ):
            writable_target = writable_target.parent
        if writable_target.exists() and os.access(writable_target, os.W_OK):
            writable.add(raw)
    return frozenset(readable), frozenset(writable)


def _is_admin() -> bool:
    if os.name != "nt":
        return hasattr(os, "geteuid") and os.geteuid() == 0
    try:  # pragma: no cover - exercised by the Windows CI lane
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def detect_local_capabilities(
    spec: SystemSpecification | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> LocalCapabilities:
    """Inspect only capabilities named by the package's local specification.

    Detection is deliberately static and side-effect free: it reads the active
    profile configuration and credential presence, but never executes a
    binary, loads plugin code, probes a destination, or exposes secret values.
    """

    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}

    enabled_toolsets: set[str] = set()
    try:
        from hermes_cli.tools_config import _platform_toolset_summary

        enabled_toolsets.update(
            _platform_toolset_summary(config, ["cli"]).get("cli", set())
        )
    except Exception:
        configured = (config.get("platform_toolsets") or {}).get("cli", [])
        if isinstance(configured, list):
            enabled_toolsets.update(str(item) for item in configured)

    enabled_names = set(enabled_toolsets)
    try:
        from toolsets import resolve_toolset

        for toolset in enabled_toolsets:
            enabled_names.update(resolve_toolset(toolset, include_registry=False))
    except Exception:
        pass

    required_tools = spec.tools if spec else []
    enabled_tools: dict[str, str | None] = {}
    for requirement in required_tools:
        if requirement.name in enabled_names or shutil.which(requirement.name):
            enabled_tools[requirement.name] = _distribution_version(requirement.name)

    plugins: dict[str, str | None] = {}
    plugin_config = config.get("plugins")
    enabled_plugins = (
        plugin_config.get("enabled", []) if isinstance(plugin_config, dict) else []
    )
    enabled_plugin_names = {
        str(item) for item in enabled_plugins if isinstance(item, str)
    }
    for requirement in spec.plugins if spec else []:
        if requirement.id in enabled_plugin_names:
            plugins[requirement.id] = _distribution_version(requirement.id)

    credentials: set[str] = set()
    connections: set[str] = set()
    try:
        from hermes_cli.config import get_env_value

        for name in spec.credentials if spec else []:
            if get_env_value(name):
                credentials.add(name)

        messaging_env = {
            "telegram": "TELEGRAM_BOT_TOKEN",
            "discord": "DISCORD_BOT_TOKEN",
            "slack": "SLACK_BOT_TOKEN",
            "whatsapp": "WHATSAPP_ENABLED",
        }
        for name in spec.connections if spec else []:
            env_name = messaging_env.get(name.lower())
            if env_name and get_env_value(env_name):
                connections.add(name)
    except Exception:
        pass
    mcp_servers = config.get("mcp_servers")
    if isinstance(mcp_servers, dict):
        for name, value in mcp_servers.items():
            if not isinstance(value, dict) or value.get("enabled", True) is not False:
                connections.add(str(name))

    model_capabilities: set[str] = set()
    context_window: int | None = None
    model_config = config.get("model")
    if isinstance(model_config, dict):
        model = str(model_config.get("default") or model_config.get("name") or "")
        provider = str(model_config.get("provider") or "")
        configured_context = model_config.get("context_length")
        if isinstance(configured_context, int) and configured_context > 0:
            context_window = configured_context
        if provider and model:
            try:
                from agent.models_dev import get_model_capabilities

                model_info = get_model_capabilities(
                    provider, model, allow_network=False
                )
                if model_info:
                    if model_info.supports_tools:
                        model_capabilities.add("tools")
                    if model_info.supports_vision:
                        model_capabilities.add("vision")
                    if model_info.supports_reasoning:
                        model_capabilities.add("reasoning")
                    context_window = context_window or model_info.context_window
            except Exception:
                pass

    required_paths = []
    if spec:
        required_paths = [*spec.filesystem.read, *spec.filesystem.write]
    filesystem_readable, filesystem_writable = _path_capabilities(required_paths)

    terminal = config.get("terminal")
    terminal = terminal if isinstance(terminal, dict) else {}
    backend = str(terminal.get("backend") or "local").lower()
    network_access = not (
        backend == "docker" and not terminal.get("docker_network", True)
    )
    runtime = {
        "shell": shutil.which("sh") is not None or platform.system() == "Windows",
        "browser": "browser" in enabled_toolsets,
        "code": "code_execution" in enabled_toolsets,
        "sandbox": True,
    }
    return LocalCapabilities(
        hermes_version=HERMES_VERSION,
        os=_normalized_os(platform.system()),
        architecture=_normalized_architecture(platform.machine()),
        model_capabilities=frozenset(model_capabilities),
        context_window=context_window,
        enabled_tools=enabled_tools,
        plugins=plugins,
        credentials=frozenset(credentials),
        connections=frozenset(connections),
        filesystem_readable=filesystem_readable,
        filesystem_writable=filesystem_writable,
        network_access=network_access,
        runtime=runtime,
        hardware=frozenset({"cpu", _normalized_architecture(platform.machine())}),
        is_admin=_is_admin(),
    )


def _version_at_least(actual: str | None, minimum: str | None) -> bool:
    if minimum is None:
        return actual is not None
    if actual is None:
        return False
    try:
        return Version(actual) >= Version(minimum)
    except InvalidVersion:
        return actual == minimum


def evaluate(
    spec: SystemSpecification, local: LocalCapabilities
) -> CompatibilityResult:
    satisfied: list[str] = []
    setup: list[str] = []
    partial: list[str] = list(spec.known_limitations)
    blocked: list[str] = []
    if _version_at_least(local.hermes_version, spec.hermes.minimum_version):
        satisfied.append(f"Hermes >= {spec.hermes.minimum_version}")
    else:
        blocked.append(f"Hermes >= {spec.hermes.minimum_version}")
    if spec.platforms and _normalized_os(local.os) not in {
        _normalized_os(item) for item in spec.platforms
    }:
        blocked.append(f"platform in {', '.join(spec.platforms)}")
    if spec.architectures and _normalized_architecture(local.architecture) not in {
        _normalized_architecture(item) for item in spec.architectures
    }:
        blocked.append(f"architecture in {', '.join(spec.architectures)}")
    missing_caps = sorted(set(spec.model.capabilities) - set(local.model_capabilities))
    if missing_caps:
        partial.append("model capabilities: " + ", ".join(missing_caps))
    if spec.model.minimum_context_window and (
        local.context_window is None
        or local.context_window < spec.model.minimum_context_window
    ):
        partial.append(f"model context window >= {spec.model.minimum_context_window}")
    for requirement in spec.tools:
        present = requirement.name in local.enabled_tools
        actual = local.enabled_tools.get(requirement.name)
        label = requirement.name + (
            f">={requirement.minimum_version}" if requirement.minimum_version else ""
        )
        if requirement.requires_admin and not local.is_admin:
            blocked.append(f"administrator approval for tool {label}")
        elif not present or (
            requirement.minimum_version is not None
            and not _version_at_least(actual, requirement.minimum_version)
        ):
            setup.append(f"enable tool {label}")
    for requirement in spec.plugins:
        present = requirement.id in local.plugins
        actual = local.plugins.get(requirement.id)
        if not present or (
            requirement.minimum_version is not None
            and not _version_at_least(actual, requirement.minimum_version)
        ):
            (setup if requirement.required else partial).append(
                f"install plugin {requirement.id}"
            )
    setup.extend(
        f"configure credential {item}"
        for item in spec.credentials
        if item not in local.credentials
    )
    setup.extend(
        f"connect {item}" for item in spec.connections if item not in local.connections
    )
    setup.extend(
        f"grant filesystem read access to {item}"
        for item in spec.filesystem.read
        if item not in local.filesystem_readable
    )
    setup.extend(
        f"grant filesystem write access to {item}"
        for item in spec.filesystem.write
        if item not in local.filesystem_writable
    )
    if spec.network.destinations and local.network_access is not True:
        setup.extend(
            f"allow network access to {item}" for item in spec.network.destinations
        )
    elif spec.network.destinations:
        satisfied.append("network egress is enabled")
    for capability, required in spec.runtime.model_dump().items():
        if required and not local.runtime.get(capability, False):
            blocked.append(f"runtime capability {capability}")
    missing_hw = sorted(set(spec.hardware) - set(local.hardware))
    blocked.extend(f"hardware {item}" for item in missing_hw)
    if blocked:
        outcome: CompatibilityOutcome = "blocked_pending_action"
    elif setup:
        outcome = "compatible_after_setup"
    elif partial:
        outcome = "partial"
    else:
        outcome = "compatible"
    return CompatibilityResult(
        outcome, tuple(satisfied), tuple(setup), tuple(partial), tuple(blocked)
    )
