"""Explicit custom routes survive unrelated providers that serve the same model."""

import json


def test_complete_custom_snapshot_survives_unrelated_entry_with_same_model(tmp_path, monkeypatch):
    import cli
    from hermes_cli import runtime_provider
    from hermes_state import SessionDB

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    saved_url = "https://explicit.example.test/v1"
    other_url = "https://configured.example.test/v1"
    (home / "config.yaml").write_text(
        "model:\n  default: ambient-model\n  provider: openai\n"
        "providers:\n  other:\n    base_url: " + other_url + "\n"
        "    default_model: shared-model\n"
        "custom_providers: []\n",
        encoding="utf-8",
    )
    # No managed server is part of this fixture; keep its discovery entirely local.
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint._state_endpoint", lambda: None)
    resolved = []

    def resolve(**kwargs):
        resolved.append(kwargs)
        if kwargs["requested"] == "custom:other":
            return {"base_url": other_url, "api_key": "synthetic-other-key",
                    "api_mode": "chat_completions"}
        return {"base_url": saved_url, "api_key": "synthetic-saved-key",
                "api_mode": "anthropic_messages"}

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    with SessionDB(home / "state.db") as db:
        db.create_session("saved", "cli", model="shared-model")
        db.update_session_model("saved", "shared-model", provider="custom",
                                base_url=saved_url, api_mode="anthropic_messages")
        row = db.get_session("saved")
        assert json.loads(row["model_config"])["gateway_runtime"]["base_url"] == saved_url

        resumed = object.__new__(cli.HermesCLI)
        resumed.model = "ambient-model"
        resumed.provider = resumed.requested_provider = "openai"
        resumed.base_url = "https://ambient.example.test/v1"
        resumed.api_mode = "chat_completions"
        resumed.api_key = "synthetic-ambient-key"
        resumed._credential_pool = None
        resumed._explicit_model_override = False
        resumed._explicit_base_url = resumed._explicit_api_key = None
        resumed.agent = None
        resumed._console_print = lambda text: None
        resumed._restore_session_model(row)

        assert (resumed.model, resumed.provider, resumed.base_url, resumed.api_mode) == (
            "shared-model", "custom", saved_url, "anthropic_messages")
        assert resumed.api_key == "synthetic-saved-key"
        assert all(call["requested"] != "custom:other" for call in resolved)
