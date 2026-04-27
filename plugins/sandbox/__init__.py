"""Sandbox provider plugin discovery.

Sandbox providers are single-select execution backends selected via
``sandbox.provider`` or the legacy ``terminal.backend``/``TERMINAL_ENV`` bridge.
They are discovered under bundled ``plugins/sandbox/<name>/`` and user
``$HERMES_HOME/plugins/sandbox/<name>/`` directories.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_SANDBOX_PLUGINS_DIR = Path(__file__).parent


def _get_user_sandbox_plugins_dir() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "sandbox"
        return d if d.is_dir() else None
    except Exception:
        return None


def _is_sandbox_provider_dir(path: Path) -> bool:
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace")[:8192]
        return "register_sandbox_provider" in source or "SandboxProvider" in source
    except Exception:
        return False


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    seen: set[str] = set()
    dirs: List[Tuple[str, Path]] = []

    if _SANDBOX_PLUGINS_DIR.is_dir():
        for child in sorted(_SANDBOX_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if not (child / "__init__.py").exists():
                continue
            seen.add(child.name)
            dirs.append((child.name, child))

    user_dir = _get_user_sandbox_plugins_dir()
    if user_dir:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen or not _is_sandbox_provider_dir(child):
                continue
            dirs.append((child.name, child))

    return dirs


def find_provider_dir(name: str) -> Optional[Path]:
    bundled = _SANDBOX_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled

    user_dir = _get_user_sandbox_plugins_dir()
    if user_dir:
        user = user_dir / name
        if user.is_dir() and _is_sandbox_provider_dir(user):
            return user
    return None


def discover_sandbox_providers() -> List[Tuple[str, str, bool]]:
    """Return ``[(name, description, available), ...]``."""

    results: List[Tuple[str, str, bool]] = []
    for name, child in _iter_provider_dirs():
        desc = ""
        yaml_file = child / "plugin.yaml"
        if yaml_file.exists():
            try:
                import yaml

                with open(yaml_file, encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "") or ""
            except Exception:
                pass

        try:
            provider = _load_provider_from_dir(child)
            available = bool(provider and provider.is_available())
        except Exception:
            available = False
        results.append((name, desc, available))
    return results


def load_sandbox_provider(name: str) -> Optional["SandboxProvider"]:
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug("Sandbox provider '%s' not found", name)
        return None

    try:
        return _load_provider_from_dir(provider_dir)
    except Exception as exc:
        logger.warning("Failed to load sandbox provider '%s': %s", name, exc)
        return None


def _load_provider_from_dir(provider_dir: Path) -> Optional["SandboxProvider"]:
    name = provider_dir.name
    is_bundled = (
        _SANDBOX_PLUGINS_DIR in provider_dir.parents
        or provider_dir.parent == _SANDBOX_PLUGINS_DIR
    )
    module_name = f"plugins.sandbox.{name}" if is_bundled else f"_hermes_user_sandbox.{name}"
    init_file = provider_dir / "__init__.py"
    if not init_file.exists():
        return None

    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        for parent in ("plugins", "plugins.sandbox"):
            if parent in sys.modules:
                continue
            parent_path = Path(__file__).parent
            if parent == "plugins":
                parent_path = parent_path.parent
            parent_init = parent_path / "__init__.py"
            if not parent_init.exists():
                continue
            spec = importlib.util.spec_from_file_location(
                parent,
                str(parent_init),
                submodule_search_locations=[str(parent_path)],
            )
            if spec and spec.loader:
                parent_mod = importlib.util.module_from_spec(spec)
                sys.modules[parent] = parent_mod
                try:
                    spec.loader.exec_module(parent_mod)
                except Exception:
                    pass

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(init_file),
            submodule_search_locations=[str(provider_dir)],
        )
        if not spec or not spec.loader:
            return None

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod

        for sub_file in provider_dir.glob("*.py"):
            if sub_file.name == "__init__.py":
                continue
            full_sub_name = f"{module_name}.{sub_file.stem}"
            if full_sub_name in sys.modules:
                continue
            sub_spec = importlib.util.spec_from_file_location(full_sub_name, str(sub_file))
            if sub_spec and sub_spec.loader:
                sub_mod = importlib.util.module_from_spec(sub_spec)
                sys.modules[full_sub_name] = sub_mod
                try:
                    sub_spec.loader.exec_module(sub_mod)
                except Exception as exc:
                    logger.debug("Failed to load sandbox submodule %s: %s", full_sub_name, exc)

        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            logger.debug("Failed to exec sandbox module %s: %s", module_name, exc)
            sys.modules.pop(module_name, None)
            return None

    if hasattr(mod, "register"):
        collector = _ProviderCollector()
        try:
            mod.register(collector)
            if collector.provider:
                return collector.provider
        except Exception as exc:
            logger.debug("register() failed for sandbox provider %s: %s", name, exc)

    from agent.sandbox_provider import SandboxProvider

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if isinstance(attr, type) and issubclass(attr, SandboxProvider) and attr is not SandboxProvider:
            try:
                return attr()
            except Exception:
                pass
    return None


class _ProviderCollector:
    def __init__(self):
        self.provider = None

    def register_sandbox_provider(self, provider):
        self.provider = provider

    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass

    def register_memory_provider(self, *args, **kwargs):
        pass

    def register_context_engine(self, *args, **kwargs):
        pass


def discover_plugin_cli_commands() -> List[dict]:
    """Return CLI commands exposed by sandbox provider plugins.

    Unlike memory providers, sandbox setup commands must be visible before a
    provider is active, so this scans every discovered provider for ``cli.py``.
    """

    results: List[dict] = []
    for name, plugin_dir in _iter_provider_dirs():
        cli_file = plugin_dir / "cli.py"
        if not cli_file.exists():
            continue

        is_bundled = (
            _SANDBOX_PLUGINS_DIR in plugin_dir.parents
            or plugin_dir.parent == _SANDBOX_PLUGINS_DIR
        )
        module_name = (
            f"plugins.sandbox.{name}.cli"
            if is_bundled
            else f"_hermes_user_sandbox.{name}.cli"
        )
        try:
            if module_name in sys.modules:
                cli_mod = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, str(cli_file))
                if not spec or not spec.loader:
                    continue
                cli_mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = cli_mod
                spec.loader.exec_module(cli_mod)

            register_cli = getattr(cli_mod, "register_cli", None)
            if not callable(register_cli):
                continue

            help_text = f"Manage {name} sandbox provider"
            description = ""
            yaml_file = plugin_dir / "plugin.yaml"
            if yaml_file.exists():
                try:
                    import yaml

                    with open(yaml_file, encoding="utf-8") as f:
                        meta = yaml.safe_load(f) or {}
                    desc = meta.get("description", "") or ""
                    if desc:
                        help_text = desc
                        description = desc
                except Exception:
                    pass

            results.append(
                {
                    "name": name,
                    "help": help_text,
                    "description": description,
                    "setup_fn": register_cli,
                    "handler_fn": getattr(cli_mod, f"{name}_command", None),
                    "plugin": name,
                }
            )
        except Exception as exc:
            logger.debug("Failed to scan CLI for sandbox plugin '%s': %s", name, exc)
    return results
