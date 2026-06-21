def test_default_evolution_config_is_disabled():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["evolution"] == {
        "enabled": False,
        "record_diff": True,
        "redact": True,
        "max_diff_chars": 20000,
    }
