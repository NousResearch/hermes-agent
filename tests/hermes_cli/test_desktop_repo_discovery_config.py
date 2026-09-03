from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.web_server import CONFIG_SCHEMA


def test_desktop_repo_discovery_defaults_are_opt_in_by_root():
    desktop = DEFAULT_CONFIG["desktop"]

    assert desktop["repo_scan_enabled"] is True
    # Empty roots are deliberately safe: Desktop does not infer a home-wide
    # search. Users must explicitly configure roots or use session projects.
    assert desktop["repo_scan_roots"] == []
    assert desktop["repo_scan_exclude_paths"] == []


def test_desktop_repo_discovery_keys_are_in_generated_schema():
    assert CONFIG_SCHEMA["desktop.repo_scan_enabled"]["type"] == "boolean"
    assert CONFIG_SCHEMA["desktop.repo_scan_roots"]["type"] == "list"
    assert CONFIG_SCHEMA["desktop.repo_scan_exclude_paths"]["type"] == "list"
