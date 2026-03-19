"""
Cosmos Optional Module Registry.

Defines all optional capability modules and their dependencies.
Each module can be installed via Y/N prompts during setup.
"""

import os
import sys
import subprocess
import importlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """Configuration for an optional module."""
    name: str
    description: str
    package: str
    dependencies: List[str] = field(default_factory=list)
    size_mb: int = 5
    requires: List[str] = field(default_factory=list)
    env_vars: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    post_install: Optional[str] = None
    python_min_version: Optional[str] = None


OPTIONAL_MODULES: Dict[str, ModuleConfig] = {
    "bankr": ModuleConfig(
        name="Bankr Trading Engine",
        description="Crypto trading, DeFi, Polymarket via Bankr API",
        package="cosmos.integration.bankr",
        dependencies=["httpx"],
        size_mb=5,
        requires=["Bankr API key (BANKR_API_KEY)"],
        env_vars=["BANKR_API_KEY"],
    ),
    "x402": ModuleConfig(
        name="x402 Protocol",
        description="Micropayments via Bankr SDK + custom API monetization",
        package="cosmos.integration.x402",
        dependencies=["httpx", "web3"],
        size_mb=50,
        requires=["Bankr API key", "USDC on Base network"],
        depends_on=["bankr"],
    ),
    "nlp_tasks": ModuleConfig(
        name="Natural Language Tasks",
        description="Hey Farn, do this... style commands (routes crypto to Bankr)",
        package="cosmos.core.nlp",
        dependencies=[],
        size_mb=5,
        requires=[],
    ),
    "desktop": ModuleConfig(
        name="Windows Desktop Interface",
        description="Full GUI for local interaction with system tray",
        package="cosmos.desktop",
        dependencies=["PySide6", "keyboard", "darkdetect"],
        size_mb=150,
        requires=[],
    ),
    "browser_agent": ModuleConfig(
        name="Agentic Browser",
        description="Autonomous web browsing with Playwright",
        package="cosmos.agents.browser",
        dependencies=["browser-use", "playwright"],
        size_mb=300,
        requires=[],
        post_install="playwright install chromium",
        python_min_version="3.11",
    ),
    "ide": ModuleConfig(
        name="IDE with Terminal",
        description="Code editor with Monaco and integrated terminal",
        package="cosmos.ide",
        dependencies=["pygments", "pywinpty"],
        size_mb=20,
        requires=[],
    ),
    "ue5": ModuleConfig(
        name="Unreal Engine 5 Integration",
        description="UE5 automation and asset creation",
        package="cosmos.integration.ue5",
        dependencies=[],
        size_mb=5,
        requires=["Unreal Engine 5 installed separately"],
    ),
    "cad": ModuleConfig(
        name="CAD Integration",
        description="3D modeling with CadQuery/FreeCAD",
        package="cosmos.integration.cad",
        dependencies=["cadquery"],
        size_mb=200,
        requires=[],
    ),
}


def get_installed_modules() -> List[str]:
    """Return list of installed module keys."""
    installed = []
    for key, config in OPTIONAL_MODULES.items():
        try:
            importlib.import_module(config.package)
            installed.append(key)
        except ImportError:
            pass
    return installed


def is_module_available(module_key: str) -> bool:
    """Check if a module is installed and available."""
    if module_key not in OPTIONAL_MODULES:
        return False

    config = OPTIONAL_MODULES[module_key]

    # Check dependencies
    for dep_key in config.depends_on:
        if not is_module_available(dep_key):
            return False

    # Check if package can be imported
    try:
        importlib.import_module(config.package)
        return True
    except ImportError:
        return False


def check_env_vars(module_key: str) -> Dict[str, bool]:
    """Check if required environment variables are set."""
    if module_key not in OPTIONAL_MODULES:
        return {}

    config = OPTIONAL_MODULES[module_key]
    results = {}

    for var in config.env_vars:
        results[var] = bool(os.environ.get(var))

    return results


def prompt_modules() -> List[str]:
    """Interactive prompt for optional module selection."""
    print("\n" + "=" * 50)
    print("OPTIONAL MODULES")
    print("=" * 50 + "\n")

    selected = []

    for key, config in OPTIONAL_MODULES.items():
        deps = ", ".join(config.requires) if config.requires else "None"

        print(f"{config.name}")
        print(f"  {config.description}")
        print(f"  Size: ~{config.size_mb}MB | Requires: {deps}")

        if config.depends_on:
            print(f"  Depends on: {', '.join(config.depends_on)}")

        if config.python_min_version:
            print(f"  Python {config.python_min_version}+ required")

        choice = input(f"  Install? [Y/n]: ").strip().lower()
        if choice != 'n':
            # Check dependencies first
            missing_deps = [d for d in config.depends_on if d not in selected]
            if missing_deps:
                print(f"    -> Also selecting: {', '.join(missing_deps)}")
                selected.extend(missing_deps)
            selected.append(key)
        print()

    return list(set(selected))  # Remove duplicates


def install_modules(selected: List[str]) -> bool:
    """Install selected optional modules."""
    success = True

    for key in selected:
        if key not in OPTIONAL_MODULES:
            logger.warning(f"Unknown module: {key}")
            continue

        config = OPTIONAL_MODULES[key]
        print(f"\nInstalling {config.name}...")

        # Check Python version
        if config.python_min_version:
            version = tuple(map(int, config.python_min_version.split('.')))
            if sys.version_info[:2] < version:
                print(f"  Skipping: Requires Python {config.python_min_version}+")
                continue

        # Install dependencies
        if config.dependencies:
            print(f"  Installing: {', '.join(config.dependencies)}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", *config.dependencies],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
                success = False
                continue

        # Run post-install commands
        if config.post_install:
            print(f"  Running: {config.post_install}")
            result = subprocess.run(
                config.post_install,
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  Warning: {result.stderr}")

        print(f"  {config.name} installed successfully!")

    return success


def get_module_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all optional modules."""
    status = {}

    for key, config in OPTIONAL_MODULES.items():
        status[key] = {
            "name": config.name,
            "installed": is_module_available(key),
            "env_vars": check_env_vars(key),
            "dependencies_met": all(
                is_module_available(d) for d in config.depends_on
            ),
        }

    return status
