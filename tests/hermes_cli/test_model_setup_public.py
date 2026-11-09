"""Public helpers a provider plugin's ``model_setup_flow`` calls (documented in model-provider-plugin.md)."""

from hermes_cli.plugin_validate_private_imports import private_import_findings


def test_persist_model_selection_writes_the_model_section(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: old\n  provider: openrouter\n  api_key: inline-secret\n")
    from hermes_cli.model_setup import persist_model_selection

    persist_model_selection("m-1", "fake-provider", base_url="https://api.example.test/v1",
                            api_mode="chat_completions")

    from hermes_cli.config import load_config

    model = load_config()["model"]
    assert (model["default"], model["provider"]) == ("m-1", "fake-provider")
    assert model["base_url"] == "https://api.example.test/v1" and model["api_mode"] == "chat_completions"
    assert "api_key" not in model  # endpoint credentials of the previous route never carry over


def test_note_setup_failure_reaches_the_setup_metric(monkeypatch):
    seen = []
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_setup.note_provider_setup_failure", seen.append)
    from hermes_cli.model_setup import note_setup_failure

    note_setup_failure("auth")
    assert seen == ["auth"]


def test_secret_source_label_names_an_external_source(monkeypatch):
    monkeypatch.setattr("hermes_cli.env_loader.get_secret_source",
                        lambda name: "bitwarden" if name == "FAKE_TOKEN" else None)
    from agent.secret_scope import secret_source_label

    assert secret_source_label("FAKE_TOKEN") == " (from Bitwarden)"
    assert secret_source_label("OTHER_TOKEN") == ""


def test_plugin_validate_treats_the_helpers_as_public(tmp_path):
    (tmp_path / "plugin.yaml").write_text("name: fake-provider\nkind: model-provider\n")
    (tmp_path / "__init__.py").write_text(
        "from hermes_cli.model_setup import persist_model_selection, note_setup_failure\n"
        "from agent.secret_scope import secret_source_label\n"
        "from hermes_cli.model_setup_flows_common import _persist_model\n")
    assert private_import_findings(tmp_path) == ["__init__.py:3 hermes_cli.model_setup_flows_common._persist_model"]
