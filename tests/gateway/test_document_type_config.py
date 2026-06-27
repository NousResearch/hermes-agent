"""Tests for config-driven document attachment allowlist overrides."""
import yaml

from gateway.config import Platform, load_gateway_config
from gateway.platforms.base import resolve_document_types


def test_gateway_document_types_load_and_propagate(tmp_path, monkeypatch):
    config_yaml = {
        "gateway": {
            "document_types": {
                "add": {".epub": "application/epub+zip"},
                "remove": [".doc"],
            }
        },
        "platforms": {
            "telegram": {
                "enabled": True,
                "token": "fake-token",
            }
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)

    cfg = load_gateway_config()
    extra = cfg.platforms[Platform.TELEGRAM].extra

    assert extra["_global_document_types"] == config_yaml["gateway"]["document_types"]
    effective = resolve_document_types(extra)
    assert effective[".epub"] == "application/epub+zip"
    assert ".doc" not in effective


def test_platform_document_types_override_global_config(tmp_path, monkeypatch):
    config_yaml = {
        "gateway": {
            "document_types": {
                "add": {".foo": "application/x-global"},
                "remove": [".pdf"],
            }
        },
        "platforms": {
            "telegram": {
                "enabled": True,
                "token": "fake-token",
                "document_types": {
                    "add": {
                        ".foo": "application/x-platform",
                        ".pdf": "application/pdf",
                    },
                    "remove": [".doc"],
                },
            }
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config_yaml), encoding="utf-8")
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)

    cfg = load_gateway_config()
    extra = cfg.platforms[Platform.TELEGRAM].extra
    effective = resolve_document_types(extra)

    assert effective[".foo"] == "application/x-platform"
    assert effective[".pdf"] == "application/pdf"
    assert ".doc" not in effective
