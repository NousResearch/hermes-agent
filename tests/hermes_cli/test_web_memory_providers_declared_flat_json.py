"""The declared panel reads the provider's own flat config file.

Bundled providers keep their flat config at ``$HERMES_HOME/<name>.json`` — the file their own
``save_config`` writes and their loader reads (``mem0.json``, ``supermemory.json``). The declared
payload read only ``<name>/config.json``, the historical dashboard location, so a provider that
declared a schema rendered an empty form while its real values sat on disk.
"""

import json

import hermes_cli.web_routers.memory_providers as mp
from plugins.memory.config_schema import get_provider_config_schema


def _payload(tmp_path, monkeypatch, *, env=None, config=None):
    monkeypatch.setattr(mp, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(mp, "load_config", lambda: config or {})
    monkeypatch.setattr(mp, "load_env", lambda: env or {})
    schema = get_provider_config_schema("mem0")
    assert schema is not None
    return mp._declared_provider_payload(schema)


def _mem0_schema():
    schema = get_provider_config_schema("mem0")
    assert schema is not None
    return schema


def _point_mem0_at(tmp_path, monkeypatch):
    """Both the router and mem0's own loader resolve this temp home."""
    import hermes_constants

    monkeypatch.setattr(mp, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(mp, "load_config", lambda: {})
    monkeypatch.setattr(mp, "load_env", lambda: {})
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)


def _field(payload, key):
    return next(field for field in payload["fields"] if field["key"] == key)


def test_values_come_from_the_providers_own_json_file(tmp_path, monkeypatch):
    (tmp_path / "mem0.json").write_text(json.dumps({
        "mode": "platform",
        "host": "http://localhost:8000",
        "user_id": "zhangrun",
        "agent_id": "macbook",
        "sync_max_chars": 6000,
    }), encoding="utf-8")

    payload = _payload(tmp_path, monkeypatch)

    assert _field(payload, "host")["value"] == "http://localhost:8000"
    assert _field(payload, "user_id")["value"] == "zhangrun"
    assert _field(payload, "agent_id")["value"] == "macbook"
    assert str(_field(payload, "sync_max_chars")["value"]) == "6000"


def test_historical_dashboard_path_still_reads(tmp_path, monkeypatch):
    (tmp_path / "mem0").mkdir()
    (tmp_path / "mem0" / "config.json").write_text(json.dumps({"user_id": "legacy"}), encoding="utf-8")

    assert _field(_payload(tmp_path, monkeypatch), "user_id")["value"] == "legacy"


def test_provider_file_wins_and_config_yaml_is_the_fallback(tmp_path, monkeypatch):
    # A provider that never overrode save_config is written to config.memory.<name>.
    config = {"memory": {"mem0": {"user_id": "yaml-user", "agent_id": "yaml-agent"}}}
    assert _field(_payload(tmp_path, monkeypatch, config=config), "user_id")["value"] == "yaml-user"

    (tmp_path / "mem0.json").write_text(json.dumps({"user_id": "zhangrun"}), encoding="utf-8")
    payload = _payload(tmp_path, monkeypatch, config=config)

    assert _field(payload, "user_id")["value"] == "zhangrun"
    # Keys absent from the provider's own file still fall back.
    assert _field(payload, "agent_id")["value"] == "yaml-agent"


def test_secret_is_never_echoed_and_reports_set_from_env(tmp_path, monkeypatch):
    payload = _payload(tmp_path, monkeypatch, env={"MEM0_API_KEY": "mem0-secret"})
    api_key = _field(payload, "api_key")
    assert api_key["value"] == ""
    assert api_key["is_set"] is True

    payload = _payload(tmp_path, monkeypatch, env={})
    assert _field(payload, "api_key")["is_set"] is False


def test_declared_save_reaches_the_file_mem0s_own_loader_reads(tmp_path, monkeypatch):
    """A declared save lands where the runtime reads, then reads back through mem0 itself.

    Writing ``$HERMES_HOME/<name>/config.json`` while mem0 loads ``$HERMES_HOME/mem0.json`` left
    the panel confirming a value the provider never used.
    """
    (tmp_path / "mem0.json").write_text(
        json.dumps({"host": "http://old:8000", "user_id": "before"}), encoding="utf-8"
    )
    _point_mem0_at(tmp_path, monkeypatch)

    mp._write_provider_flat(
        _mem0_schema(),
        {"host": "http://new:8000", "user_id": "zhangrun", "sync_max_chars": "6000"},
    )

    from plugins.memory.mem0 import _load_config

    loaded = _load_config()
    assert loaded["host"] == "http://new:8000"
    assert loaded["user_id"] == "zhangrun"
    assert loaded["sync_max_chars"] == 6000
    # The runtime's file is the only one written: no shadow copy for the panel to show instead.
    assert not (tmp_path / "mem0" / "config.json").exists()


def test_declared_save_merges_and_keeps_unsubmitted_keys(tmp_path, monkeypatch):
    (tmp_path / "mem0.json").write_text(
        json.dumps({"user_id": "keep", "agent_id": "keep-too"}), encoding="utf-8"
    )
    _point_mem0_at(tmp_path, monkeypatch)

    mp._write_provider_flat(_mem0_schema(), {"host": "http://localhost:8000"})

    assert json.loads((tmp_path / "mem0.json").read_text(encoding="utf-8")) == {
        "user_id": "keep",
        "agent_id": "keep-too",
        "host": "http://localhost:8000",
    }


def test_provider_without_save_config_writes_the_config_section(tmp_path, monkeypatch):
    """A provider with no ownership seam keeps the legacy writer's target."""
    from agent.memory_provider import MemoryProvider

    class _NoOverride:
        # Class-level, like a real provider: the override check reads type(provider).save_config.
        save_config = MemoryProvider.save_config

    _point_mem0_at(tmp_path, monkeypatch)
    monkeypatch.setattr(mp, "_load_memory_provider", lambda name: _NoOverride())
    saved: dict = {}
    monkeypatch.setattr(mp, "save_config", lambda cfg: saved.update(cfg))

    mp._write_provider_flat(_mem0_schema(), {"user_id": "from-panel"})

    assert saved["memory"]["mem0"]["user_id"] == "from-panel"
    assert not (tmp_path / "mem0.json").exists()
