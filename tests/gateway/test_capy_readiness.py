from scripts import capy_provider_fallback_check, capy_rl_readiness


def test_rl_readiness_reports_missing_credentials_without_values(monkeypatch):
    monkeypatch.setattr(capy_rl_readiness, "_load_dotenv", lambda: [])
    for name in (*capy_rl_readiness.REQUIRED_KEYS, *capy_rl_readiness.OPTIONAL_KEYS):
        monkeypatch.delenv(name, raising=False)
    status = capy_rl_readiness.run_smoke()
    rendered = capy_rl_readiness.render_text(status)
    assert status["status"] == "not_ready"
    assert status["smoke"]["skipped"] is True
    assert "TINKER_API_KEY: missing" in rendered
    assert "WANDB_API_KEY: missing" in rendered
    assert "[REDACTED]" not in rendered


def test_rl_readiness_redacts_smoke_output(monkeypatch):
    monkeypatch.setattr(capy_rl_readiness, "_load_dotenv", lambda: [])
    monkeypatch.setenv("TINKER_API_KEY", "tinker-secret-value")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret-value")
    monkeypatch.setattr(
        capy_rl_readiness,
        "_run_checked",
        lambda cmd, cwd, timeout=120: {
            "cmd": cmd,
            "exit_code": 0,
            "output_tail": "[REDACTED]",
        },
    )
    status = capy_rl_readiness.run_smoke()
    assert status["status"] == "ready"
    assert status["smoke"]["passed"] is True
    assert "tinker-secret-value" not in str(status)
    assert "wandb-secret-value" not in str(status)


def test_provider_fallback_collect_shape(monkeypatch):
    monkeypatch.setattr(capy_provider_fallback_check, "_http_ok", lambda url, timeout=1.0: url.endswith("/models"))
    monkeypatch.setattr(
        capy_provider_fallback_check,
        "_config_model",
        lambda: {"provider": "openai-codex", "model": "gpt-5.5"},
    )
    for name in ("OPENROUTER_API_KEY", "NOUS_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    data = capy_provider_fallback_check.collect()
    assert data["primary_expected"]["matches"] is True
    assert data["lmstudio"]["models_endpoint_ok"] is True
    assert all(value is False for value in data["optional_credentials"].values())
