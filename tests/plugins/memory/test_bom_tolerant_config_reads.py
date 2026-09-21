"""User-editable plugin/auth JSON survives a Windows-editor BOM.

Notepad and PowerShell ``>`` prepend U+FEFF when saving; ``json.loads`` rejects it
("Unexpected UTF-8 BOM") and every loader below degrades to defaults, so a user who
edited mem0.json / honcho.json / hindsight config.json / supermemory.json lost the
whole config with no error (Qwen CLI creds raised ``qwen_auth_read_failed``). Same
class as the auth-store/.env sweep; these were the missed sibling readers.
"""

import json
from pathlib import Path

import pytest


def _write_bom_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8-sig")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # the BOM must really be on disk
    return path


def _via_shared_reader(p: Path, monkeypatch) -> dict:
    from utils import read_json_or_empty  # mem0 / hindsight / honcho CLI all read through this
    return read_json_or_empty(p)


def _via_supermemory(p: Path, monkeypatch) -> dict:
    from plugins.memory.supermemory import _load_supermemory_config
    return _load_supermemory_config(str(p.parent))


def _via_hindsight_runtime_deps(p: Path, monkeypatch) -> dict:
    import hermes_cli.memory_setup as ms
    monkeypatch.setattr(ms, "get_hermes_home", lambda: p.parent.parent)  # <home>/hindsight/config.json
    deps = ms._provider_pip_dependencies("hindsight", ["hindsight-client"])
    return {"workspace": "bom-ws" if "hindsight-all" in deps else None, "container_tag": None}


def _via_honcho_client(p: Path, monkeypatch) -> dict:
    from plugins.memory.honcho.client import HonchoClientConfig
    cfg = HonchoClientConfig.from_global_config(config_path=p)
    return {"workspace": cfg.workspace_id, "container_tag": None}


@pytest.mark.parametrize("filename, loader", [
    ("mem0.json", _via_shared_reader),
    ("supermemory.json", _via_supermemory),
    ("honcho.json", _via_honcho_client),
    ("hindsight/config.json", _via_hindsight_runtime_deps),
])
def test_plugin_config_json_tolerates_bom(tmp_path, monkeypatch, filename, loader):
    target = tmp_path / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_bom_json(target, {"workspace": "bom-ws", "container_tag": "bom-tag", "enabled": True, "mode": "local_embedded"})
    loaded = loader(target, monkeypatch)
    assert "bom" in str(loaded.get("workspace") or loaded.get("container_tag"))


def test_qwen_cli_tokens_tolerate_bom(tmp_path, monkeypatch):
    import hermes_cli.auth as auth_mod

    creds = _write_bom_json(tmp_path / "oauth_creds.json", {"access_token": "tok", "expiry_date": 4102444800000})
    monkeypatch.setattr(auth_mod, "_qwen_cli_auth_path", lambda: creds)
    assert auth_mod._read_qwen_cli_tokens()["access_token"] == "tok"
