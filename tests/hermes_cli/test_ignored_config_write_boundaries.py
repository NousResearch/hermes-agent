"""The ignore flag refuses effective replacements, not raw edits; dashboard refusals stay typed."""
import pytest

from hermes_cli.config_backend import ConfigWriteError
from hermes_cli.config_worker_save import USER_CONFIG_IGNORED, USER_CONFIG_IGNORED_MESSAGE


@pytest.fixture
def ignored_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    (tmp_path / "config.yaml").write_text("model:\n  default: keep/me\nunknown_sibling: keep\n")
    return tmp_path


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    from hermes_cli import web_server
    result = TestClient(web_server.app, raise_server_exceptions=False)
    result.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return result


def test_raw_editor_still_saves_while_the_schema_editor_refuses(ignored_home, client):
    from hermes_cli.config import read_raw_config
    original = (ignored_home / "config.yaml").read_bytes()
    schema = client.put("/api/config", json={"config": {"display": {"skin": "mono"}}})
    assert schema.status_code == 409, schema.text
    assert (ignored_home / "config.yaml").read_bytes() == original
    raw = client.put("/api/config/raw", json={"yaml_text": "unknown_sibling: edited\n"})
    assert raw.status_code == 200, raw.text
    assert read_raw_config() == {"unknown_sibling": "edited"}


def test_schema_editor_refuses_before_parsing_broken_yaml(ignored_home, client):
    path = ignored_home / "config.yaml"
    path.write_text("model: [broken\n")
    original = path.read_bytes()
    response = client.put("/api/config", json={"config": {"display": {"skin": "mono"}}})
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == USER_CONFIG_IGNORED_MESSAGE
    assert path.read_bytes() == original


def test_moa_raw_preset_save_keeps_siblings_under_the_flag(ignored_home, client):
    from hermes_cli.config import read_raw_config
    response = client.put("/api/model/moa", json={
        "enabled": False, "reference_models": [{"provider": "custom", "model": "reference"}],
        "aggregator": {"provider": "custom", "model": "aggregate"},
    })
    assert response.status_code == 200, response.text
    assert read_raw_config()["unknown_sibling"] == "keep"
    assert read_raw_config()["moa"]["enabled"] is False


def test_voice_copy_preserves_raw_settings_under_the_flag(ignored_home, tmp_path):
    from hermes_cli.profile_provision import _mirror_voice_sections
    from tui_gateway.launch_profile_policy import profile_config_scope
    from hermes_cli.config import read_raw_config
    (ignored_home / "config.yaml").write_text("voice:\n  reply: true\n")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "config.yaml").write_text("unknown_sibling: keep\n")
    assert _mirror_voice_sections(destination) is True
    with profile_config_scope(destination):
        copied = read_raw_config()
    assert copied["voice"] == {"reply": True}
    assert copied["unknown_sibling"] == "keep"


def test_raw_endpoint_seed_and_url_restore_preserve_templates(ignored_home, monkeypatch):
    """Exercise both raw inheritance writes; model validation is the only replaced boundary."""
    from hermes_cli import config as config_mod, profile_provision
    from tui_gateway.launch_profile_policy import profile_config_scope
    monkeypatch.setenv("PROBE_URL", "https://example.invalid/v1")
    (ignored_home / "config.yaml").write_text(
        'model:\n  provider: custom\n  default: picked/model\n  base_url: "${PROBE_URL}"\n'
        'providers:\n  custom:\n    base_url: "${PROBE_URL}"\n')
    destination = ignored_home / "destination"
    destination.mkdir()
    (destination / "config.yaml").write_text("unknown_sibling: keep\n")
    selected = {"model": {"provider": "custom", "default": "picked/model",
                          "base_url": "https://example.invalid/v1"}}
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: selected)

    def validated_pin(path, provider, model, **kwargs):
        with profile_config_scope(path):
            raw = config_mod.read_user_config_raw()
            raw["model"] = {"provider": provider, "default": model, "base_url": kwargs["base_url"]}
            config_mod.atomic_config_write(path / "config.yaml", raw)

    monkeypatch.setattr(profile_provision, "write_profile_model", validated_pin)
    assert profile_provision.inherit_launch_model(destination) is True
    with profile_config_scope(destination):
        raw = config_mod.read_user_config_raw()
    assert raw["model"]["base_url"] == "${PROBE_URL}"
    assert raw["providers"]["custom"]["base_url"] == "${PROBE_URL}"
    assert raw["unknown_sibling"] == "keep"


@pytest.mark.parametrize("code,status", [(USER_CONFIG_IGNORED, 409), ("config_write_failed", 500)])
def test_uncaught_dashboard_save_is_mapped_and_backend_details_are_opaque(
    ignored_home, client, monkeypatch, caplog, code, status
):
    from hermes_cli.web_routers import dashboard_ui
    def fail(*args, **kwargs):
        raise ConfigWriteError("backend-private-detail", code=code)
    monkeypatch.setattr(dashboard_ui, "save_config", fail)
    response = client.put("/api/dashboard/theme", json={"name": "default"})
    assert response.status_code == status, response.text
    if status == 500:
        assert "backend-private-detail" not in response.text
        assert "backend-private-detail" in caplog.text
    else:
        assert response.json()["detail"] == "backend-private-detail"


@pytest.mark.parametrize("boundary", ["memory", "profiles", "http_failure"])
@pytest.mark.parametrize("code,status", [(USER_CONFIG_IGNORED, 409), ("config_write_failed", 500)])
def test_local_broad_catches_preserve_refusals_and_hide_backend_failures(boundary, code, status, caplog):
    from fastapi import HTTPException
    from hermes_cli.web_routers.memory_providers import _value_errors_as_http
    from hermes_cli.web_routers.profiles import _profile_errors
    from hermes_cli.web_routers._common import http_failure
    contexts = {
        "memory": lambda: _value_errors_as_http("failed %s", "probe", passthrough_http=False),
        "profiles": lambda: _profile_errors("failed"),
        "http_failure": lambda: http_failure("failed", 500, "legacy prefix"),
    }
    with pytest.raises(HTTPException) as error, contexts[boundary]():
        raise ConfigWriteError("backend-private-detail", code=code)
    assert error.value.status_code == status
    if status == 500:
        assert "backend-private-detail" not in error.value.detail
        assert "backend-private-detail" in caplog.text
    else:
        assert error.value.detail == "backend-private-detail"


@pytest.mark.parametrize("code,status", [(USER_CONFIG_IGNORED, 409), ("config_write_failed", 500)])
def test_local_model_route_does_not_turn_config_errors_into_502(ignored_home, client, monkeypatch, code, status):
    from hermes_cli.web_routers import local_models
    def fail():
        raise ConfigWriteError("backend-private-detail", code=code)
    monkeypatch.setitem(local_models._SERVER_ACTIONS, "stop", fail)
    response = client.post("/api/local-models/server", json={"action": "stop"})
    assert response.status_code == status, response.text
    if status == 500:
        assert "backend-private-detail" not in response.text


def test_refusal_message_is_surface_neutral():
    assert "hermes config set" not in USER_CONFIG_IGNORED_MESSAGE


def test_raw_save_scope_unwinds_before_a_defaults_save(ignored_home):
    from hermes_cli.config_worker_save import saving_raw_file
    from hermes_cli.config import load_config, read_raw_config, save_config
    with pytest.raises(ValueError), saving_raw_file():
        raw = read_raw_config()
        raw["unknown_sibling"] = "edited"
        save_config(raw)
        raise ValueError("leave scope")
    with pytest.raises(ConfigWriteError):
        save_config(load_config())


@pytest.mark.parametrize("limit,label", [(19, "19"), (None, "unlimited")])
def test_classic_config_renders_the_resolved_budget(ignored_home, capsys, limit, label):
    from datetime import datetime
    from cli import HermesCLI
    from hermes_cli.config_turn_limit import resolve_turn_limit
    resolved = resolve_turn_limit(limit)
    stand_in = HermesCLI.__new__(HermesCLI)
    stand_in.api_key, stand_in.agent, stand_in.model, stand_in.base_url = None, None, "test/model", "local"
    stand_in.max_turns, stand_in.enabled_toolsets, stand_in.verbose = resolved, [], False
    stand_in.session_start = datetime(2026, 1, 1)
    HermesCLI.show_config(stand_in)
    output = capsys.readouterr().out
    assert label in output
    if limit is None:
        assert str(resolved) not in output


@pytest.mark.parametrize("code,status", [(USER_CONFIG_IGNORED, 409), ("config_write_failed", 500)])
def test_mcp_local_catch_preserves_config_error_classification(
    ignored_home, client, monkeypatch, caplog, code, status
):
    from hermes_cli import mcp_config
    def fail(*args, **kwargs):
        raise ConfigWriteError("backend-private-detail", code=code)
    monkeypatch.setattr(mcp_config, "_save_mcp_server", fail)
    response = client.post("/api/mcp/servers", json={"name": "probe", "command": "echo", "args": []})
    assert response.status_code == status, response.text
    if status == 500:
        assert "backend-private-detail" not in response.text
        assert "backend-private-detail" in caplog.text


def test_local_model_background_error_is_opaque_and_logged(monkeypatch, caplog):
    from hermes_cli.web_routers import local_models
    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target
        def start(self):
            self.target()
    monkeypatch.setattr(local_models, "spawn_context_thread", lambda target, **kwargs: InlineThread(target))
    def fail():
        raise ConfigWriteError("backend-private-detail", code="config_write_failed")
    job = {"job_id": "probe", "status": "running"}
    local_models._spawn_job(job, "probe", fail)
    assert job["status"] == "error"
    assert "backend-private-detail" not in job["error"]
    assert "backend-private-detail" in caplog.text


@pytest.mark.parametrize("code,status", [(USER_CONFIG_IGNORED, 409), ("config_write_failed", 500)])
def test_kanban_mounted_route_uses_host_handler(ignored_home, monkeypatch, caplog, code, status):
    import importlib.util
    import sys
    from pathlib import Path
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    from hermes_cli import config, web_server
    path = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_refusal_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.add_exception_handler(ConfigWriteError, web_server.app.exception_handlers[ConfigWriteError])
    app.include_router(module.router, prefix="/api/plugins/kanban")
    def fail(*args, **kwargs):
        raise ConfigWriteError("backend-private-detail", code=code)
    monkeypatch.setattr(config, "save_config", fail)
    response = TestClient(app, raise_server_exceptions=False).put("/api/plugins/kanban/orchestration", json={})
    assert response.status_code == status, response.text
    if status == 500:
        assert "backend-private-detail" not in response.text
        assert "backend-private-detail" in caplog.text


@pytest.mark.parametrize("expected", [True, False])
def test_voice_copy_returns_a_warning_without_failing_profile_creation(ignored_home, monkeypatch, caplog, expected):
    from hermes_cli import profile_provision
    from hermes_cli.config_backend import ConfigLockedError
    from hermes_cli import auth
    error = ConfigLockedError("voice.reply", "managed") if expected else ConfigWriteError("backend-private-detail")
    def fail(*args):
        raise error
    monkeypatch.setattr(profile_provision, "_mirror_secret", lambda *args: False)
    monkeypatch.setattr(profile_provision, "_mirror_voice_sections", fail)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: None)
    warnings, mirrored = [], {}
    profile_provision.mirror_launch_credentials(ignored_home, mirrored, config_warnings=warnings)
    assert mirrored["voice"] is False
    assert len(warnings) == 1
    if expected:
        assert str(error) in warnings[0]
    else:
        assert "backend-private-detail" not in warnings[0]
        assert "backend-private-detail" in caplog.text


@pytest.mark.parametrize("kind", ["locked", "value", "conflict", "unexpected"])
def test_post_create_step_preserves_actionable_refusals(ignored_home, monkeypatch, kind):
    from hermes_cli import profile_provision, profiles
    from hermes_cli.config_backend import ConfigLockedError, ConfigValueError
    errors = {"locked": ConfigLockedError("model", "managed"), "value": ConfigValueError("invalid model"),
              "conflict": ConfigWriteError("reload before saving", code="config_version_conflict"),
              "unexpected": ConfigWriteError("backend-private-detail")}
    error = errors[kind]
    monkeypatch.setattr(profiles, "create_profile", lambda **kwargs: ignored_home)
    def fail(*args):
        raise error
    monkeypatch.setattr(profile_provision, "_pin_model", fail)
    request = profile_provision.ProfileCreateRequest(name="probe", clone_from="source", alias=False,
        mirror_credentials=False, provider="custom", model="model")
    result = profile_provision.provision_profile(request)
    assert result.path == ignored_home and result.model_set is False
    assert len(result.warnings) == 1
    if kind == "unexpected":
        assert "backend-private-detail" not in result.warnings[0]
    else:
        assert str(error) in result.warnings[0]


@pytest.mark.parametrize("key", ["", "replacement-value"])
def test_refused_endpoint_edit_does_not_touch_existing_credentials(ignored_home, client, key):
    from hermes_cli.config import save_env_value
    from hermes_cli.web_routers.config_env import custom_endpoint_key_env
    save_env_value(custom_endpoint_key_env("probe-endpoint"), "original-value")
    env_path, config_path = ignored_home / ".env", ignored_home / "config.yaml"
    before = env_path.read_bytes(), config_path.read_bytes()
    response = client.post("/api/providers/custom-endpoints", json={
        "id": "probe-endpoint", "name": "Probe endpoint", "model": "probe-model",
        "base_url": "https://example.invalid/v1", "api_key": key,
    })
    assert response.status_code == 409, response.text
    assert (env_path.read_bytes(), config_path.read_bytes()) == before


def test_refused_mcp_add_does_not_store_a_bearer_token(ignored_home, client):
    env = ignored_home / ".env"
    env.write_text("OTHER_KEY=keep\n")
    before = env.read_bytes()
    response = client.post("/api/mcp/servers", json={
        "name": "probe", "url": "https://example.invalid/mcp", "auth": "header",
        "bearer_token": "replacement-value",
    })
    assert response.status_code == 409, response.text
    assert env.read_bytes() == before


def test_refused_declared_memory_selection_does_not_write_provider_fields(ignored_home, monkeypatch):
    from hermes_cli.web_routers import memory_providers
    from plugins.memory.config_schema import ProviderConfigSchema, STORAGE_FLAT_JSON
    writes = []
    monkeypatch.setitem(memory_providers._STORAGE_WRITERS, STORAGE_FLAT_JSON,
                        lambda provider, values: writes.append(values))
    provider = ProviderConfigSchema(name="probe", label="Probe")
    with pytest.raises(ConfigWriteError):
        memory_providers._update_memory_provider_config(provider, {})
    assert writes == []


def test_refused_messaging_toggle_does_not_clear_a_live_token(ignored_home, client):
    env = ignored_home / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=original-value\n")
    before = env.read_bytes()
    response = client.put("/api/messaging/platforms/telegram", json={
        "enabled": True, "clear_env": ["TELEGRAM_BOT_TOKEN"],
    })
    assert response.status_code == 409, response.text
    assert env.read_bytes() == before
