"""Probe module loader. Add new module names here as you create them."""
from __future__ import annotations

import importlib

_MODULES = [
    "userscan.specs.host_identity",     # L0-ish identity + install age + locale + shell prefs
                                        # (+ security, network, hardware, health, usage on Windows)
    "userscan.specs.apps_dev",          # installed software, AI agents, dev environment
    "userscan.specs.browser_files",     # browsers, files and content
    "userscan.specs.usage_gaming",      # usage habits, gaming, media, hardware health
    "userscan.specs.derive",            # L3 insights (persona, fresh-vs-old)
    "userscan.specs.linux_system",      # Linux: host, identity, install_age, locale, shell_prefs,
                                        # security, network, hardware, health, usage
    "userscan.specs.linux_apps",        # Linux: apps, ai_agents, dev, browser, comms_work, files,
                                        # gaming, media
    "userscan.specs.darwin_system",     # macOS: host, identity, install_age, locale, shell_prefs,
                                        # security, network, hardware, health, usage
    "userscan.specs.darwin_apps",       # macOS: apps, ai_agents, dev, browser, comms_work, files,
                                        # gaming, media
]

for m in _MODULES:
    try:
        importlib.import_module(m)
    except ImportError:
        pass
