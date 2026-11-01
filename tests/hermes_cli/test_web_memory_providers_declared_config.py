"""The declared settings surface must show the same effective values the runtime
honors: config.yaml's ``memory.<provider>`` block wins over the flat
``<provider>/config.json``, while the flat file still fills keys the yaml omits."""

import json

import pytest

import hermes_cli.config as config_mod
from hermes_cli.web_routers import memory_providers as mp
from plugins.memory.hindsight.config_schema import CONFIG_SCHEMA


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config_mod, "load_env", lambda: {})
    return tmp_path


def _field(payload, key):
    return next(f for f in payload["fields"] if f["key"] == key)


def test_declared_surface_reads_memory_block_from_config_yaml(hermes_home):
    (hermes_home / "config.yaml").write_text(
        "memory:\n  hindsight:\n    mode: local_external\n    api_url: http://localhost:8888\n",
        encoding="utf-8",
    )

    payload = mp._declared_provider_payload(CONFIG_SCHEMA)

    assert _field(payload, "mode")["value"] == "local_external"
    assert _field(payload, "api_url")["value"] == "http://localhost:8888"


def test_config_yaml_beats_flat_json_but_json_still_fills_gaps(hermes_home):
    provider_dir = hermes_home / "hindsight"
    provider_dir.mkdir()
    (provider_dir / "config.json").write_text(
        json.dumps({"mode": "cloud", "api_url": "https://json.example", "bank_id": "json-bank"}),
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "memory:\n  hindsight:\n    mode: local_external\n",
        encoding="utf-8",
    )

    payload = mp._declared_provider_payload(CONFIG_SCHEMA)

    assert _field(payload, "mode")["value"] == "local_external"
    assert _field(payload, "api_url")["value"] == "https://json.example"
    assert _field(payload, "bank_id")["value"] == "json-bank"
