"""BOM-tolerant reads of user-editable plugin/auth JSON files.

Windows GUI editors (Notepad, PowerShell ``>``) prepend a UTF-8 BOM when
saving JSON. ``json.loads`` hard-fails on a leading BOM ("Unexpected UTF-8
BOM (decode using utf-8-sig)"), and every loader here swallows the exception
and silently falls back to defaults — so a user who edited mem0.json /
honcho.json / hindsight config.json / supermemory.json in Notepad lost their
entire config with no error. Same class as the merged #81967 sweep (auth
store, .env, memory files); these plugin-config readers were the missed
sibling sites. Ported alongside earendil-works/pi#8337's BOM normalization.

Each test writes the file with a real BOM (utf-8-sig encoding) and asserts
the loader still returns the configured values.
"""

import json

import pytest


def _write_bom_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8-sig")
    # Sanity: the BOM must actually be on disk for the test to mean anything.
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_mem0_load_config_tolerates_bom(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path
    )
    _write_bom_json(tmp_path / "mem0.json", {"agent_id": "bom-agent"})

    from plugins.memory.mem0 import _load_config

    assert _load_config()["agent_id"] == "bom-agent"


def test_supermemory_load_config_tolerates_bom(tmp_path):
    _write_bom_json(
        tmp_path / "supermemory.json", {"container_tag": "bom-tag"}
    )

    from plugins.memory.supermemory import _load_supermemory_config

    assert _load_supermemory_config(str(tmp_path))["container_tag"] == "bom-tag"


def test_hindsight_load_config_tolerates_bom(tmp_path, monkeypatch):
    (tmp_path / "hindsight").mkdir()
    _write_bom_json(
        tmp_path / "hindsight" / "config.json", {"mode": "bom-mode"}
    )
    import plugins.memory.hindsight as hs

    monkeypatch.setattr(hs, "get_hermes_home", lambda: tmp_path)
    assert hs._load_config()["mode"] == "bom-mode"


def test_honcho_cli_read_config_tolerates_bom(tmp_path, monkeypatch):
    import plugins.memory.honcho.cli as hcli

    cfg_path = tmp_path / "honcho.json"
    monkeypatch.setattr(hcli, "_config_path", lambda: cfg_path)
    _write_bom_json(cfg_path, {"workspace": "bom-ws"})
    assert hcli._read_config()["workspace"] == "bom-ws"


def test_honcho_client_from_global_config_tolerates_bom(tmp_path):
    from plugins.memory.honcho.client import HonchoClientConfig

    cfg_path = tmp_path / "honcho.json"
    _write_bom_json(cfg_path, {"workspace": "bom-ws", "enabled": True})
    cfg = HonchoClientConfig.from_global_config(config_path=cfg_path)
    assert cfg.workspace_id == "bom-ws"
    assert cfg.explicitly_configured is True


def test_qwen_cli_tokens_tolerates_bom(tmp_path, monkeypatch):
    import hermes_cli.auth as auth_mod

    creds = tmp_path / "oauth_creds.json"
    _write_bom_json(
        creds,
        {"access_token": "tok", "expiry_date": 4102444800000},
    )
    monkeypatch.setattr(auth_mod, "_qwen_cli_auth_path", lambda: creds)
    data = auth_mod._read_qwen_cli_tokens()
    assert data["access_token"] == "tok"


def test_plain_utf8_still_parses(tmp_path):
    """utf-8-sig reads plain UTF-8 unchanged — no regression for normal files."""
    from plugins.memory.supermemory import _load_supermemory_config

    (tmp_path / "supermemory.json").write_text(
        json.dumps({"container_tag": "plain-tag"}), encoding="utf-8"
    )
    assert (
        _load_supermemory_config(str(tmp_path))["container_tag"] == "plain-tag"
    )
