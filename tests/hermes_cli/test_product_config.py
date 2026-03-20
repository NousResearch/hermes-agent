from pathlib import Path

import yaml

from hermes_cli.product_config import (
    DEFAULT_PRODUCT_CONFIG,
    ensure_product_home,
    get_product_config_path,
    get_product_storage_root,
    get_product_users_root,
    initialize_product_config_file,
    load_product_config,
    resolve_runtime_defaults,
    save_product_config,
)


def test_ensure_product_home_creates_expected_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    ensure_product_home()

    assert get_product_storage_root().is_dir()
    assert get_product_users_root().is_dir()
    assert (get_product_storage_root() / "logs").is_dir()
    assert (get_product_storage_root() / "services").is_dir()


def test_load_product_config_returns_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()

    assert config["auth"]["provider"] == "pocket-id"
    assert config["auth"]["mode"] == "passkey"
    assert config["tools"]["hermes_toolsets"] == DEFAULT_PRODUCT_CONFIG["tools"]["hermes_toolsets"]


def test_save_product_config_roundtrip_and_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["product"]["brand"]["name"] = "Erni Agent"
    config["tools"]["hermes_toolsets"] = ["web", "memory"]
    save_product_config(config)

    reloaded = load_product_config()
    assert reloaded["product"]["brand"]["name"] == "Erni Agent"
    assert reloaded["tools"]["hermes_toolsets"] == ["web", "memory"]
    assert reloaded["auth"]["provider"] == "pocket-id"

    saved = yaml.safe_load(get_product_config_path().read_text(encoding="utf-8"))
    assert saved["product"]["brand"]["name"] == "Erni Agent"


def test_initialize_product_config_file_creates_product_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = initialize_product_config_file()

    assert get_product_config_path().exists()
    assert config["bootstrap"]["first_admin_username"] == "admin"


def test_resolve_runtime_defaults_reads_product_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["tools"]["hermes_toolsets"] = ["memory", "session_search"]
    config["models"]["default_route"]["model"] = "custom-model"
    save_product_config(config)

    defaults = resolve_runtime_defaults()

    assert defaults["runtime_mode"] == "product"
    assert defaults["runtime_toolsets"] == "memory,session_search"
    assert defaults["runtime_profile"] == "product"
    assert defaults["runtime_toolset"] == "memory"
    assert defaults["inference_model"] == "custom-model"
