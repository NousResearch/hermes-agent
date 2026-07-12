"""MVP method handler tests against tmp vault fixtures (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.projects_db import connect_closing, create_project
from tests.gateway.brain_rpc.conftest import make_auth, make_request


@pytest.mark.asyncio
async def test_brain_ping(dispatcher):
    res = await dispatcher.handle(
        make_request("brain.ping", params={"echo": "hi"})
    )
    assert res["ok"] is True
    assert res["type"] == "brain_rpc_result"
    assert res["contract_version"] == 1
    assert res["result"]["pong"] is True
    assert res["result"]["echo"] == "hi"
    assert res["result"]["instance_id"] == "inst_test"
    assert res["meta"]["hermes_profile"] == "contributor"


@pytest.mark.asyncio
async def test_brain_ping_echo_too_long(dispatcher):
    res = await dispatcher.handle(
        make_request("brain.ping", params={"echo": "x" * 65})
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_brain_health(dispatcher, vault_root):
    res = await dispatcher.handle(make_request("brain.health"))
    assert res["ok"] is True
    assert res["result"]["status"] in {"ok", "degraded", "unavailable"}
    assert res["result"]["checks"]["vault_root_readable"] is True
    assert res["result"]["profile"] == "contributor"
    assert res["result"]["versions"]["brain_rpc"] == 1


@pytest.mark.asyncio
async def test_vault_list(dispatcher):
    res = await dispatcher.handle(
        make_request("vault.list", params={"path": "/Projects/Legal", "limit": 50})
    )
    assert res["ok"] is True
    names = {e["name"] for e in res["result"]["entries"]}
    assert "memo.md" in names
    assert "exhibits" in names
    for e in res["result"]["entries"]:
        assert "content" not in e
        assert e["kind"] in {"file", "directory"}


@pytest.mark.asyncio
async def test_vault_list_pagination(dispatcher):
    res1 = await dispatcher.handle(
        make_request("vault.list", params={"path": "/Projects/Legal", "limit": 1})
    )
    assert res1["ok"] is True
    assert len(res1["result"]["entries"]) == 1
    assert res1["result"]["truncated"] is True
    cursor = res1["result"]["next_cursor"]
    res2 = await dispatcher.handle(
        make_request(
            "vault.list",
            params={"path": "/Projects/Legal", "limit": 1, "cursor": cursor},
        )
    )
    assert res2["ok"] is True
    assert len(res2["result"]["entries"]) == 1
    assert res1["result"]["entries"][0]["name"] != res2["result"]["entries"][0]["name"]


@pytest.mark.asyncio
async def test_vault_stat(dispatcher):
    res = await dispatcher.handle(
        make_request("vault.stat", params={"path": "/Projects/Legal/memo.md"})
    )
    assert res["ok"] is True
    assert res["result"]["kind"] == "file"
    assert res["result"]["name"] == "memo.md"
    assert res["result"]["size_bytes"] > 0


@pytest.mark.asyncio
async def test_vault_stat_not_found(dispatcher):
    res = await dispatcher.handle(
        make_request("vault.stat", params={"path": "/Projects/Legal/missing.md"})
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_vault_read(dispatcher):
    res = await dispatcher.handle(
        make_request("vault.read", params={"path": "/Projects/Legal/memo.md"})
    )
    assert res["ok"] is True
    assert res["result"]["encoding"] == "utf-8"
    assert "Memo" in res["result"]["content"]
    assert res["result"]["kind"] == "file"


@pytest.mark.asyncio
async def test_vault_read_payload_too_large(dispatcher, vault_root):
    big = vault_root / "Projects" / "Legal" / "big.bin"
    big.write_bytes(b"x" * 200)
    res = await dispatcher.handle(
        make_request(
            "vault.read",
            params={"path": "/Projects/Legal/big.bin", "max_bytes": 50},
        )
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "payload_too_large"


@pytest.mark.asyncio
async def test_vault_read_directory_invalid(dispatcher):
    res = await dispatcher.handle(
        make_request("vault.read", params={"path": "/Projects/Legal"})
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_admin_can_read_secrets(dispatcher):
    res = await dispatcher.handle(
        make_request(
            "vault.read",
            params={"path": "/Secrets/keys.txt"},
            auth=make_auth(profile="admin", path_prefixes=[]),
        )
    )
    assert res["ok"] is True
    assert res["result"]["content"] == "do-not-read"


@pytest.mark.asyncio
async def test_settings_snapshot_redacts_secrets(dispatcher, tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  model: test-model\n"
        "mcp_servers:\n  demo:\n    transport: stdio\n"
        "    env:\n      API_KEY: super-secret\n"
        "    api_key: should-not-appear\n",
        encoding="utf-8",
    )
    skills = home / "skills" / "demo-skill"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    res = await dispatcher.handle(make_request("settings.snapshot"))
    assert res["ok"] is True
    blob = json.dumps(res["result"])
    assert "super-secret" not in blob
    assert "should-not-appear" not in blob
    assert "api_key" not in blob
    assert res["result"]["source"] == "hermes"
    assert any(s["id"] == "demo-skill" for s in res["result"]["skills"])
    assert any(m["model"] == "test-model" for m in res["result"]["models"])
    assert any(s["name"] == "demo" for s in res["result"]["mcp_servers"])


@pytest.mark.asyncio
async def test_projects_list_missing_db(dispatcher, tmp_path, monkeypatch):
    home = tmp_path / "empty_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    res = await dispatcher.handle(make_request("projects.list"))
    assert res["ok"] is True
    assert res["result"]["projects"] == []
    assert res["result"]["source"] == "none"


@pytest.mark.asyncio
async def test_projects_list_with_db(dispatcher, tmp_path, monkeypatch):
    home = tmp_path / "proj_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = home / "projects.db"
    with connect_closing(db_path=db) as conn:
        create_project(
            conn,
            name="Legal",
            slug="legal",
            folders=["/Projects/Legal"],
            primary_path="/Projects/Legal",
        )
    res = await dispatcher.handle(make_request("projects.list"))
    assert res["ok"] is True
    assert res["result"]["source"] == "hermes_projects_db"
    assert len(res["result"]["projects"]) == 1
    assert res["result"]["projects"][0]["slug"] == "legal"


@pytest.mark.asyncio
async def test_method_not_found(dispatcher):
    res = await dispatcher.handle(make_request("vault.write"))
    assert res["ok"] is False
    assert res["error"]["code"] == "method_not_found"


@pytest.mark.asyncio
async def test_version_unsupported(dispatcher):
    res = await dispatcher.handle(
        make_request("brain.ping", contract_version=99)
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "version_unsupported"


@pytest.mark.asyncio
async def test_disabled_feature_gate(dispatcher, monkeypatch):
    monkeypatch.setenv("BRAIN_RPC_ENABLED", "0")
    res = await dispatcher.handle(make_request("brain.ping"))
    assert res["ok"] is False
    assert res["error"]["code"] == "unavailable"
