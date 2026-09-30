"""Probe module loader. Add new module names here as you create them."""
from __future__ import annotations

import importlib

_MODULES = [
    ".host_identity",          # L0-ish identity + install age + locale + shell prefs
                               # (+ security, network, hardware, health, usage on Windows)
    ".apps_dev",               # installed software (+ shared inventories)
    ".apps_dev_agents",        # AI agents
    ".apps_dev_devenv",        # dev environment
    ".browser_files",          # browsers
    ".browser_files_comms",    # comms and work
    ".browser_files_content",  # files and content
    ".usage_gaming",           # usage habits
    ".usage_gaming_games",     # gaming, media
    ".usage_gaming_hardware",  # hardware health
    ".derive",                 # L3 insights (persona, fresh-vs-old)
    ".linux_system",           # Linux: host, identity, install_age, locale, shell_prefs, health, usage
    ".linux_system_security",  # Linux: security, network
    ".linux_system_hardware",  # Linux: hardware
    ".linux_apps",             # Linux: apps, ai_agents
    ".linux_apps_dev",         # Linux: dev
    ".linux_apps_browser",     # Linux: browser, comms_work
    ".linux_apps_files",       # Linux: files, gaming, media
    ".darwin_system",          # macOS: host, identity, install_age, locale, shell_prefs,
                               # security, network, hardware, health, usage
    ".darwin_apps",            # macOS: apps, ai_agents
    ".darwin_apps_dev",        # macOS: dev
    ".darwin_apps_browser",    # macOS: browser, comms_work
    ".darwin_apps_files",      # macOS: files, gaming, media
]

for m in _MODULES:
    try:
        importlib.import_module(m, __name__)
    except ImportError:
        pass
