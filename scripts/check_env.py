#!/usr/bin/env python3
"""Environment checker for Hermes Agent.

Validates that all required and optional dependencies are available.
Run this to diagnose installation issues:

    python scripts/check_env.py
"""

import importlib
import os
import shutil
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version < (3, 10):
        return ("Python", "❌", f"{version.major}.{version.minor} (need 3.10+)")
    return ("Python", "✅", f"{version.major}.{version.minor}.{version.micro}")


def check_module(name: str, package: str = None):
    """Check if a Python module is installed."""
    package = package or name
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        return (package, "✅", version)
    except ImportError:
        return (package, "❌", "not installed")


def check_command(name: str, version_flag: str = "--version"):
    """Check if a command-line tool is available."""
    if shutil.which(name):
        return (name, "✅", "available")
    return (name, "❌", "not found in PATH")


def check_env_var(name: str, required: bool = True):
    """Check if an environment variable is set."""
    value = os.environ.get(name)
    if value:
        # Mask sensitive values
        if "key" in name.lower() or "secret" in name.lower():
            display = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        else:
            display = value[:20] + "..." if len(value) > 20 else value
        return (name, "✅", display)
    if required:
        return (name, "❌", "not set (required)")
    return (name, "⚠️", "not set (optional)")


def check_hermes_home():
    """Check Hermes home directory."""
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    if hermes_home.exists():
        skills_count = len(list(hermes_home.glob("skills/**/SKILL.md")))
        return ("HERMES_HOME", "✅", f"{hermes_home} ({skills_count} skills)")
    return ("HERMES_HOME", "⚠️", f"{hermes_home} (does not exist)")


def main():
    print("\n🔍 Hermes Agent Environment Check\n")
    print("=" * 60)
    
    # Python
    print("\n📦 Python Runtime")
    print("-" * 40)
    result = check_python_version()
    print(f"  {result[1]} {result[0]:<20} {result[2]}")
    
    # Core dependencies
    print("\n📦 Core Dependencies")
    print("-" * 40)
    core_deps = [
        ("openai", "openai"),
        ("prompt_toolkit", "prompt-toolkit"),
        ("rich", "rich"),
        ("yaml", "pyyaml"),
        ("requests", "requests"),
    ]
    for mod, pkg in core_deps:
        result = check_module(mod, pkg)
        print(f"  {result[1]} {result[0]:<20} {result[2]}")
    
    # Optional dependencies
    print("\n📦 Optional Dependencies")
    print("-" * 40)
    optional_deps = [
        ("edge_tts", "edge-tts"),
        ("elevenlabs", "elevenlabs"),
        ("playwright", "playwright"),
        ("ptyprocess", "ptyprocess"),
        ("websockets", "websockets"),
    ]
    for mod, pkg in optional_deps:
        result = check_module(mod, pkg)
        print(f"  {result[1]} {result[0]:<20} {result[2]}")
    
    # Command-line tools
    print("\n🔧 Command-Line Tools")
    print("-" * 40)
    tools = ["ffmpeg", "git", "docker", "playwright"]
    for tool in tools:
        result = check_command(tool)
        print(f"  {result[1]} {result[0]:<20} {result[2]}")
    
    # Environment variables
    print("\n🔐 Environment Variables")
    print("-" * 40)
    required_vars = [("OPENROUTER_API_KEY", True)]
    optional_vars = [
        ("OPENAI_API_KEY", False),
        ("ELEVENLABS_API_KEY", False),
        ("FIRECRAWL_API_KEY", False),
        ("ANTHROPIC_API_KEY", False),
    ]
    for var, req in required_vars + optional_vars:
        result = check_env_var(var, req)
        print(f"  {result[1]} {result[0]:<20} {result[2]}")
    
    # Hermes home
    print("\n🏠 Hermes Configuration")
    print("-" * 40)
    result = check_hermes_home()
    print(f"  {result[1]} {result[0]:<20} {result[2]}")
    
    print("\n" + "=" * 60)
    print("✨ Check complete!")
    print()


if __name__ == "__main__":
    main()
