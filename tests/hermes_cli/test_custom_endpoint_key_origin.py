"""A custom endpoint's stored key is only ever sent to the origin it was entered for.

Editing an endpoint to a new scheme/host/port with the key field left blank must not
hand the old secret (or hand-written ``extra_headers``) to the new address.
"""

import pytest

import hermes_cli.main as hermes_main


def _drive_cli_custom_flow(monkeypatch, url: str, secret_answers: list[str]) -> list[str]:
    """Run ``hermes model`` → Custom endpoint against the real temp HERMES_HOME; return the
    keys the endpoint probe was handed."""
    probed_keys: list[str] = []

    def _probe(api_key, base_url):
        probed_keys.append(api_key)
        return {"models": [], "probed_url": f"{base_url}/models", "resolved_base_url": None,
                "suggested_base_url": None, "used_fallback": False}

    monkeypatch.setattr("hermes_cli.models.probe_api_models", _probe)
    monkeypatch.setattr("hermes_cli.main_provider_setup._prompt_custom_api_mode_selection",
                        lambda *_a, **_k: "chat_completions")
    # URL, then model name / context length / display name.
    answers = iter([url, "probe-model", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    secrets = iter(secret_answers)
    monkeypatch.setattr("hermes_cli.secret_prompt.masked_secret_prompt", lambda _prompt="": next(secrets))
    monkeypatch.setattr("hermes_cli.model_setup_flows_custom._report_context_length_detection",
                        lambda *_a: None)
    hermes_main._model_flow_custom({})
    return probed_keys


@pytest.mark.parametrize(
    ("url", "secret_answers", "expected_key"),
    [
        # Same origin, different path: the stored key is still the right one.
        ("https://llm.old.example/v2", [""], "sk-old-origin-fixture"),
        # New host, blank key twice: saved keyless, the old key never leaves.
        ("https://llm.new.example/v1", ["", ""], ""),
        # New port on the same host is a new origin too.
        ("https://llm.old.example:8443/v1", ["", "sk-new-origin-fixture"], "sk-new-origin-fixture"),
    ],
)
def test_cli_blank_key_reuses_stored_key_only_on_its_own_origin(monkeypatch, url, secret_answers, expected_key):
    from hermes_cli.config import get_env_path, save_env_value

    save_env_value("OPENAI_API_KEY", "sk-old-origin-fixture")
    save_env_value("OPENAI_BASE_URL", "https://llm.old.example/v1")

    probed_keys = _drive_cli_custom_flow(monkeypatch, url, secret_answers)

    assert probed_keys == [expected_key]
    env_text = get_env_path().read_text(encoding="utf-8-sig")
    assert env_text.count("sk-old-origin-fixture") == (2 if expected_key == "sk-old-origin-fixture" else 1)


@pytest.mark.parametrize(("url", "kept"), [
    ("https://llm.new.example/v1", False),   # new host, saved keyless: the inline credential goes
    ("https://llm.old.example/v2", True),    # same origin: it is still that host's credential
])
def test_cli_keyless_save_drops_the_previous_hosts_model_credential(monkeypatch, url, kept):
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["model"] = {"provider": "custom", "default": "m", "base_url": "https://llm.old.example/v1",
                    "api_key": "sk-inline-old-fixture", "api": "sk-inline-old-fixture"}
    save_config(cfg)

    assert _drive_cli_custom_flow(monkeypatch, url, ["", ""]) == [""]

    model = load_config()["model"]
    assert model["base_url"] == url
    assert (model.get("api_key"), model.get("api")) == (
        ("sk-inline-old-fixture",) * 2 if kept else (None, None))


class TestDesktopEndpointEdit:
    @pytest.fixture(autouse=True)
    def _client(self, _isolate_hermes_home):
        from starlette.testclient import TestClient

        from hermes_cli.config import load_config, save_config, save_env_value
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

        cfg = load_config()
        cfg["providers"] = {"myep": {
            "name": "myep", "base_url": "https://llm.old.example/v1", "model": "m",
            "key_env": "HERMES_CUSTOM_MYEP_API_KEY", "key_cmd": "print-old-token",
            "extra_headers": {"X-Org-Token": "hdr-old-fixture"}, "models": {"m": {}},
        }}
        save_config(cfg)
        save_env_value("HERMES_CUSTOM_MYEP_API_KEY", "sk-old-origin-fixture")
        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _edit(self, base_url, **extra):
        return self.client.post("/api/providers/custom-endpoints", json={
            "id": "myep", "name": "myep", "base_url": base_url, "model": "m", "make_default": False, **extra})

    def test_new_origin_without_a_key_is_refused_and_nothing_changes(self):
        from hermes_cli.config import get_config_path, get_env_path, load_config

        saved = {path: path.read_bytes() for path in (get_config_path(), get_env_path())}
        response = self._edit("https://llm.new.example/v1")
        assert response.status_code == 400
        assert "API key for the new address" in response.json()["detail"]
        for path, data in saved.items():
            assert path.read_bytes() == data

        # Same origin, new path: the stored credentials stay.
        assert self._edit("https://llm.old.example/v2").status_code == 200
        entry = load_config()["providers"]["myep"]
        assert entry["key_env"] == "HERMES_CUSTOM_MYEP_API_KEY"
        assert entry["extra_headers"] == {"X-Org-Token": "hdr-old-fixture"}

    @pytest.mark.parametrize("old_credential", [{"api_key": "sk-old-origin-fixture"},
                                                {"key_env": "MY_OWN_LLM_KEY"}])
    def test_default_save_probe_never_reloads_the_old_hosts_key(self, monkeypatch, old_credential):
        """Explicit blank key + make_default: validation probes the new host keyless."""
        from hermes_cli.config import load_config, save_config, save_env_value

        cfg = load_config()
        entry = cfg["providers"]["myep"]
        for field in ("key_env", "key_cmd", "extra_headers"):
            entry.pop(field)
        entry.update(old_credential)
        save_config(cfg)
        save_env_value("MY_OWN_LLM_KEY", "sk-old-origin-fixture")
        probed: list[tuple[str, str]] = []

        def _probe(api_key, base_url, *_a, **_k):
            probed.append((api_key, base_url))
            return {"models": ["m"], "probed_url": f"{base_url}/models", "resolved_base_url": None,
                    "suggested_base_url": None, "used_fallback": False}

        monkeypatch.setattr("hermes_cli.models.probe_api_models", _probe)
        response = self._edit("https://llm.new.example/v1", api_key="", make_default=True)

        assert response.status_code == 200, response.text
        assert probed and all(base == "https://llm.new.example/v1" for _key, base in probed)
        assert all("sk-old-origin-fixture" not in str(key) for key, _base in probed)

    def test_new_origin_with_its_key_drops_every_old_credential(self):
        from hermes_cli.config import get_env_value, load_config

        assert self._edit("https://llm.new.example/v1", api_key="sk-new-origin-fixture").status_code == 200
        entry = load_config()["providers"]["myep"]
        assert entry["base_url"] == "https://llm.new.example/v1"
        assert "extra_headers" not in entry and "key_cmd" not in entry
        assert get_env_value(entry["key_env"]) == "sk-new-origin-fixture"
