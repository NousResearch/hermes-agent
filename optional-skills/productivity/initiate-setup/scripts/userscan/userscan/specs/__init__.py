"""Probe module loader. Add new module names here as you create them."""
from __future__ import annotations

import importlib

_MODULES = [
    "userscan.specs.host_identity",     # L0-ish identity + install age + locale + shell prefs
                                        # (+ security, network, hardware, health, usage on Windows)
    "userscan.specs.apps_dev",          # installed software (+ shared inventories)
    "userscan.specs.apps_dev_agents",   # AI agents
    "userscan.specs.apps_dev_devenv",   # dev environment
    "userscan.specs.browser_files",     # browsers
    "userscan.specs.browser_files_comms",    # comms and work
    "userscan.specs.browser_files_content",  # files and content
    "userscan.specs.usage_gaming",      # usage habits
    "userscan.specs.usage_gaming_games",     # gaming, media
    "userscan.specs.usage_gaming_hardware",  # hardware health
    "userscan.specs.derive",            # L3 insights (persona, fresh-vs-old)
    "userscan.specs.linux_system",      # Linux: host, identity, install_age, locale, shell_prefs, health, usage
    "userscan.specs.linux_system_security",  # Linux: security, network
    "userscan.specs.linux_system_hardware",  # Linux: hardware
    "userscan.specs.linux_apps",        # Linux: apps, ai_agents
    "userscan.specs.linux_apps_dev",    # Linux: dev
    "userscan.specs.linux_apps_browser",     # Linux: browser, comms_work
    "userscan.specs.linux_apps_files",  # Linux: files, gaming, media
    "userscan.specs.darwin_system",     # macOS: host, identity, install_age, locale, shell_prefs,
                                        # security, network, hardware, health, usage
    "userscan.specs.darwin_apps",       # macOS: apps, ai_agents
    "userscan.specs.darwin_apps_dev",   # macOS: dev
    "userscan.specs.darwin_apps_browser",    # macOS: browser, comms_work
    "userscan.specs.darwin_apps_files",  # macOS: files, gaming, media
]

for m in _MODULES:
    try:
        importlib.import_module(m)
    except ImportError:
        pass
