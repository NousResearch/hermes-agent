"""Dashboard Settings schema entry for terminal.worker_memory_max_mb."""


def test_worker_memory_max_mb_is_a_described_free_text_field():
    """Dashboard Settings shows the worker memory limit as an editable text field
    (``auto`` or whole MiB), filed under Terminal, with the default from DEFAULT_CONFIG."""
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    entry = CONFIG_SCHEMA["terminal.worker_memory_max_mb"]
    assert entry["type"] == "string"
    assert "options" not in entry
    assert "MiB" in entry["description"] and "auto" in entry["description"]
    assert entry["category"] == CONFIG_SCHEMA["terminal.backend"]["category"]
    assert DEFAULT_CONFIG["terminal"]["worker_memory_max_mb"] == "auto"
