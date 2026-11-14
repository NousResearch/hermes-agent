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

    @pytest.mark.parametrize("new_key", ["", "sk-new-dotenv-fixture"])
    def test_move_leaves_no_old_key_under_the_legacy_variable(self, new_key):
        """The guard leaves ``key_env`` to ``_apply_endpoint_key``, which clears or rewrites the row's
        own ``HERMES_CUSTOM_*`` variable; dropping it first left the old host's key in .env."""
        from hermes_cli.config import get_env_path, get_env_value
        old = get_env_value("HERMES_CUSTOM_MYEP_API_KEY")
        assert old
        assert self._edit("https://llm.new.example/v1", api_key=new_key).status_code == 200
        assert old not in get_env_path().read_text()

    def test_new_origin_with_its_key_drops_every_old_credential(self):
        from hermes_cli.config import get_env_value, load_config

        assert self._edit("https://llm.new.example/v1", api_key="sk-new-origin-fixture").status_code == 200
        entry = load_config()["providers"]["myep"]
        assert entry["base_url"] == "https://llm.new.example/v1"
        assert "extra_headers" not in entry and "key_cmd" not in entry
        assert get_env_value(entry["key_env"]) == "sk-new-origin-fixture"


@pytest.mark.parametrize(("address_field", "credential"), [
    ("api", {"key_env": "MYEP_KEY"}), ("url", {"key_env": "MYEP_KEY"}), ("baseUrl", {"key_env": "MYEP_KEY"}),
    ("base_url", {"apiKey": "sk-old-dotenv-fixture"}), ("base_url", {"keyEnv": "MYEP_KEY"}),
    ("base_url", {"apiKeyEnv": "MYEP_KEY"}),
])
def test_desktop_move_of_an_entry_addressed_by_api_or_url_drops_its_key(_isolate_hermes_home, address_field,
                                                                         credential):
    """Migrated v12 entries store the address under ``api``; hand-written ones may use ``url`` or the
    camelCase aliases the readers accept (``baseUrl``, ``apiKey``/``keyEnv``/``apiKeyEnv``)."""
    from starlette.testclient import TestClient

    from hermes_cli.config import load_config, save_config, save_env_value
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    cfg = load_config()
    cfg["providers"] = {"myep": {"name": "myep", address_field: "https://llm.old.example/v1", "model": "m",
                                 **credential, "models": {"m": {}}}}
    save_config(cfg)
    save_env_value("MYEP_KEY", "sk-old-dotenv-fixture")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    body = {"id": "myep", "name": "myep", "base_url": "https://llm.new.example/v1", "model": "m",
            "make_default": False}

    assert client.post("/api/providers/custom-endpoints", json=body).status_code == 400
    assert client.post("/api/providers/custom-endpoints", json={**body, "api_key": ""}).status_code == 200

    entry = load_config()["providers"]["myep"]
    assert entry["base_url"] == "https://llm.new.example/v1"
    assert not ({"api", "url", "baseUrl", "key_env", "api_key", *credential} & entry.keys())
    runtime = resolve_runtime_provider(requested="myep")
    assert runtime["base_url"] == "https://llm.new.example/v1"
    assert runtime.get("api_key") != "sk-old-dotenv-fixture"
    # Bare ``provider: custom`` at the new address must not find the old key on the entry either.
    cfg = load_config()
    cfg["model"] = {"provider": "custom", "base_url": "https://llm.new.example/v1", "default": "m"}
    save_config(cfg)
    assert resolve_runtime_provider().get("api_key") != "sk-old-dotenv-fixture"


def test_desktop_move_of_a_legacy_custom_providers_row_keeps_its_key_home(_isolate_hermes_home):
    from starlette.testclient import TestClient

    from hermes_cli.config import get_env_path, load_config, save_config
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    cfg = load_config()
    cfg.pop("providers", None)
    cfg["custom_providers"] = [{"name": "legacyep", "base_url": "https://llm.old.example/v1",
                                "api_key": "sk-legacy-row-fixture", "model": "m"}]
    save_config(cfg)
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    response = client.post("/api/providers/custom-endpoints", json={
        "id": "legacyep", "name": "legacyep", "base_url": "https://llm.new.example/v1", "model": "m",
        "make_default": False})

    assert response.status_code == 400
    assert "sk-legacy-row-fixture" not in (get_env_path().read_text(encoding="utf-8-sig")
                                   if get_env_path().exists() else "")
    assert not (load_config().get("providers") or {}).get("legacyep")


@pytest.mark.parametrize("pool_key", ["myep", "custom:myep"])
def test_desktop_move_never_resolves_a_pooled_key_recorded_for_the_old_origin(_isolate_hermes_home, pool_key):
    """A ``hermes auth add myep`` row lives in auth.json with the origin it was added for; after the
    entry moves, neither ``custom:myep`` nor a bare ``custom`` route may pick it at the new host."""
    from starlette.testclient import TestClient

    from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool
    from hermes_cli.config import load_config, save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    old, new, pooled = "https://llm.old.example/v1", "https://llm.new.example/v1", "sk-pooled-old-secret"
    cfg = load_config()
    cfg["providers"] = {"myep": {"name": "myep", "base_url": old, "model": "m", "models": {"m": {}}}}
    save_config(cfg)
    load_pool(pool_key).add_entry(PooledCredential(
        provider=pool_key, id="row1", label="k", auth_type=AUTH_TYPE_API_KEY, priority=0,
        source=SOURCE_MANUAL, access_token=pooled, base_url=old))
    # On its own origin the pooled key is the entry's credential.
    assert resolve_runtime_provider(requested="custom:myep")["api_key"] == pooled

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    edit = {"id": "myep", "name": "myep", "base_url": new, "model": "m", "make_default": False}
    # The pooled key counts as a stored credential: an untouched key field refuses the move...
    assert client.post("/api/providers/custom-endpoints", json=edit).status_code == 400
    # ...and an explicit blank moves the entry keyless, leaving the row bound to the old origin.
    assert client.post("/api/providers/custom-endpoints", json={**edit, "api_key": ""}).status_code == 200

    named = resolve_runtime_provider(requested="custom:myep")
    assert named["base_url"] == new and named.get("api_key") != pooled
    cfg = load_config()
    cfg["model"] = {"provider": "custom", "base_url": new, "default": "m"}
    save_config(cfg)
    bare = resolve_runtime_provider()
    assert bare["base_url"] == new and bare.get("api_key") != pooled


def test_dashboard_aux_slot_moved_to_a_new_origin_without_a_key_drops_the_old_key(monkeypatch):
    """Desktop Settings → Auxiliary: a slot re-pointed at another host with a blank key keeps no
    key entered for its previous host; a same-origin path change keeps it."""
    from hermes_cli.web_server_config import _apply_aux_assignment_sync
    monkeypatch.setattr("hermes_cli.config.save_config", lambda _cfg: None)
    old_key = "sk-aux-old-host-fixture"

    def _assign(new_url):
        cfg = {"auxiliary": {"vision": {"provider": "custom", "model": "m",
                                        "base_url": "https://llm.old.example/v1", "api_key": old_key}}}
        _apply_aux_assignment_sync(cfg, "custom", "m", "vision", new_url, "")
        return cfg["auxiliary"]["vision"].get("api_key")

    assert _assign("https://llm.old.example/v2") == old_key
    assert _assign("https://llm.new.example/v1") is None
