"""Interactive launch must survive the config file watcher init.

Regression for main@24fd22b94: ``_tui_init_run_state`` called
``utils.file_signature`` without importing it (the mixin imports lazily,
inside methods), so every interactive ``hermes`` launch crashed with
``NameError: name 'file_signature' is not defined`` — but only when a
config.yaml exists, because the call is behind ``_cfg_path.exists()``.
"""


def test_tui_init_run_state_resolves_file_signature_import(monkeypatch, tmp_path):
    from cli import HermesCLI

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # A real config.yaml makes _tui_init_run_state take the file_signature branch.
    (tmp_path / "config.yaml").write_text("model:\n  default: fixture\n")

    cli = HermesCLI(
        model="fixture",
        provider="openai-compat",
        api_key="fixture",
        base_url="http://127.0.0.1:1/v1",
    )
    # Crashes with NameError on unfixed main before doing anything else.
    cli._tui_init_run_state()
    assert cli._config_sig is not None
    assert cli._config_mcp_servers == {}
