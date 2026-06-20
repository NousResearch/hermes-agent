import json


def _write_workflow_login(home, *, base_url="https://cloud.example", token="tok-test"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "workflow-secrets.json").write_text(
        json.dumps({"kari": {"cloudBaseUrl": base_url, "token": token}}),
        encoding="utf-8",
    )


def test_model_options_include_kari_cloud_provider_when_workflow_login_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_workflow_login(tmp_path)

    from hermes_cli.inventory import build_models_payload, load_picker_context

    payload = build_models_payload(
        load_picker_context(),
        include_unconfigured=True,
        picker_hints=True,
        max_models=50,
    )

    kari = next((row for row in payload["providers"] if row["slug"] == "kari-cloud"), None)
    assert kari is not None
    assert kari["name"] == "Kari 云端"
    assert kari["models"] == ["极致", "性能", "DeepSeek"]
    assert kari["total_models"] == 3
    assert kari["authenticated"] is True


def test_runtime_provider_resolves_kari_cloud_from_workflow_login(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_workflow_login(tmp_path, base_url="https://cloud.example/", token="tok-runtime")

    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="kari-cloud", target_model="性能")

    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == "kari-cloud"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == "https://cloud.example/api/v1/kari/llm/v1"
    assert runtime["api_key"] == "tok-runtime"
    assert runtime["source"] == "custom_provider:Kari 云端"


def test_switch_model_accepts_kari_cloud_provider_from_picker_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_workflow_login(tmp_path)
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "not listed by live endpoint",
        },
    )

    from hermes_cli.inventory import load_picker_context
    from hermes_cli.model_switch import switch_model

    ctx = load_picker_context()
    result = switch_model(
        raw_input="性能",
        current_provider="",
        current_model="",
        explicit_provider="kari-cloud",
        user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers,
    )

    assert result.success is True
    assert result.target_provider == "kari-cloud"
    assert result.new_model == "性能"
    assert result.base_url == "https://cloud.example/api/v1/kari/llm/v1"
    assert result.api_key == "tok-test"
