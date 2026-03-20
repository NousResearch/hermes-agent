from pathlib import Path

from hermes_cli.product_runtime import (
    ProductRuntimeRecord,
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
    assert "You are Hermes" in soul_path.read_text(encoding="utf-8")
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
    assert soul_path.read_text(encoding="utf-8") == "Custom runtime identity\n"


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
                "runtime_profile": "admin",
                "runtime_toolset": "mynah-tier1",
            }

    monkeypatch.setattr("hermes_cli.product_runtime.httpx.get", lambda *args, **kwargs: _Response())

    payload = get_product_runtime_session({"preferred_username": "admin"})
    assert payload["session_id"] == "product_admin_123"
    assert payload["messages"][0]["content"] == "hello"
