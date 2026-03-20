from pathlib import Path

from hermes_cli.product_runtime import (
    ProductRuntimeRecord,
    _docker_run_command,
    _resolve_runtime_model_base_url,
    get_product_runtime_session,
    product_runtime_session_id,
    stage_product_runtime,
)


def test_product_runtime_session_id_is_stable():
    first = product_runtime_session_id("admin")
    second = product_runtime_session_id("admin")

    assert first == second
    assert first.startswith("product_admin_")


def test_stage_product_runtime_writes_soul_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    record = stage_product_runtime({"preferred_username": "admin", "name": "Admin User", "is_admin": True})

    soul_path = Path(record.hermes_home) / "SOUL.md"
    assert soul_path.exists()
    soul_text = soul_path.read_text(encoding="utf-8")
    assert "You are Hermes" in soul_text
    assert "Your currently enabled Hermes toolsets are: memory, session_search." in soul_text
    manifest = Path(record.manifest_file)
    assert manifest.exists()
    loaded = ProductRuntimeRecord.model_validate_json(manifest.read_text(encoding="utf-8"))
    assert loaded.user_id == "admin"
    assert loaded.runtime == "runsc"


def test_stage_product_runtime_uses_custom_soul_template(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    template_path = tmp_path / "custom-soul.md"
    template_path.write_text("Custom runtime identity", encoding="utf-8")

    from hermes_cli.product_config import load_product_config, save_product_config

    config = load_product_config()
    config["product"]["agent"]["soul_template_path"] = str(template_path)
    save_product_config(config)

    record = stage_product_runtime({"preferred_username": "admin"})
    soul_path = Path(record.hermes_home) / "SOUL.md"
    soul_text = soul_path.read_text(encoding="utf-8")
    assert "Custom runtime identity" in soul_text
    assert "Your currently enabled Hermes toolsets are: memory, session_search." in soul_text


def test_get_product_runtime_session_proxies_runtime(monkeypatch):
    record = ProductRuntimeRecord(
        user_id="admin",
        display_name="Admin",
        session_id="product_admin_123",
        container_name="runtime-admin",
        runtime="runsc",
        runtime_port=18091,
        runtime_root="/tmp/runtime",
        hermes_home="/tmp/runtime/hermes",
        workspace_root="/tmp/workspace",
        env_file="/tmp/runtime/runtime.env",
        manifest_file="/tmp/runtime/launch-spec.json",
        status="running",
    )

    monkeypatch.setattr("hermes_cli.product_runtime.ensure_product_runtime", lambda user, config=None: record)

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "session_id": "product_admin_123",
                "messages": [{"role": "assistant", "content": "hello"}],
                "runtime_mode": "product",
                "runtime_toolsets": ["memory", "session_search"],
            }

    monkeypatch.setattr("hermes_cli.product_runtime.httpx.get", lambda *args, **kwargs: _Response())

    payload = get_product_runtime_session({"preferred_username": "admin"})
    assert payload["session_id"] == "product_admin_123"
    assert payload["messages"][0]["content"] == "hello"


def test_runtime_model_base_url_rewrites_loopback_for_container_access():
    config = {"runtime": {"host_access_host": "host.docker.internal"}}

    rewritten = _resolve_runtime_model_base_url(config, "http://127.0.0.1:8080/v1")

    assert rewritten == "http://host.docker.internal:8080/v1"


def test_runtime_model_base_url_keeps_non_loopback_hosts():
    config = {"runtime": {"host_access_host": "host.docker.internal"}}

    rewritten = _resolve_runtime_model_base_url(config, "https://llm.example.internal/v1")

    assert rewritten == "https://llm.example.internal/v1"


def test_stage_product_runtime_writes_container_reachable_model_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.product_config import load_product_config, save_product_config

    config = load_product_config()
    config["models"]["default_route"]["base_url"] = "http://127.0.0.1:8080/v1"
    config["runtime"]["host_access_host"] = "host.docker.internal"
    save_product_config(config)

    record = stage_product_runtime({"preferred_username": "admin"})
    env_text = Path(record.env_file).read_text(encoding="utf-8")

    assert "OPENAI_BASE_URL=http://host.docker.internal:8080/v1" in env_text


def test_docker_run_command_adds_host_gateway_mapping():
    config = {
        "runtime": {
            "internal_port": 8091,
            "image": "hermes-product-local:dev",
            "host_access_host": "host.docker.internal",
        }
    }
    record = ProductRuntimeRecord(
        user_id="admin",
        display_name="Admin",
        session_id="product_admin_123",
        container_name="runtime-admin",
        runtime="runc",
        runtime_port=18091,
        runtime_root="/tmp/runtime",
        hermes_home="/tmp/runtime/hermes",
        workspace_root="/tmp/workspace",
        env_file="/tmp/runtime/runtime.env",
        manifest_file="/tmp/runtime/launch-spec.json",
        status="running",
    )

    command = _docker_run_command(record, config)

    assert "--add-host" in command
    assert "host.docker.internal:host-gateway" in command
