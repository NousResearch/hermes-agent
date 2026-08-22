"""Tests for the ``quota:`` config section (issue #6567).

Verifies that the quota warning thresholds and suppression flag are present
in DEFAULT_CONFIG, survive load_config(), and are sourced without drift by
cli.load_cli_config().
"""
from pathlib import Path


def test_load_config_returns_quota_defaults():
    """load_config() deep-merges DEFAULT_CONFIG, so the four quota keys
    survive when no user config file is present in the (sandboxed) HERMES_HOME.
    """
    from hermes_cli.config import load_config

    cfg = load_config()
    quota = cfg["quota"]
    assert quota["warning_threshold"] == 80
    assert quota["strong_threshold"] == 90
    assert quota["critical_threshold"] == 95
    assert quota["suppress_warnings"] is False


def test_load_cli_config_quota_matches_default_config():
    """cli.load_cli_config() must source quota from the shared DEFAULT_CONFIG
    (not a second literal copy) to prevent drift, and must include all four
    keys.  When no config file exists the defaults dict is returned nearly
    verbatim (only env-var expansion and the managed overlay run afterward).
    """
    import cli
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    cli_cfg = cli.load_cli_config()
    expected = DEFAULT_CONFIG["quota"]

    # Drift guard: the CLI path must mirror DEFAULT_CONFIG exactly.
    assert cli_cfg["quota"] == expected

    # Explicit four-key check so a silent key-drop fails loudly.
    expected_keys = {
        "warning_threshold",
        "strong_threshold",
        "critical_threshold",
        "suppress_warnings",
    }
    assert set(cli_cfg["quota"].keys()) == expected_keys


def test_cli_config_yaml_example_documents_quota():
    """cli-config.yaml.example should document the quota: section."""
    example = Path(__file__).parent.parent / "cli-config.yaml.example"
    content = example.read_text(encoding="utf-8")
    assert "quota:" in content
