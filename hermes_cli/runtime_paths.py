"""Frozen updater surface: pre-PM updaters import these names from here.

The implementation moved to ``pm.environments``; in-tree code imports it from there.
"""
from pm.environments import (  # noqa: F401
    activation_environment,
    dependency_home_root,
    install_state_dir,
    runtime_facts_path,
    selected_venv,
    site_packages,
    store_root,
)
