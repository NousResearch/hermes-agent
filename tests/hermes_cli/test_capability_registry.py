import os
from pathlib import Path

import yaml


def _home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "profiles" / "cto" / "mcp-tokens").mkdir(parents=True)
    (home / "profiles" / "cmo").mkdir(parents=True)
    (home / "config.yaml").write_text("toolsets: [hermes-cli]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    return home


def test_capability_registry_finds_disabled_vercel_with_redacted_token_presence(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    cto = home / "profiles" / "cto"
    (cto / "config.yaml").write_text(yaml.safe_dump({
        "toolsets": ["hermes-cli"],
        "mcp_servers": {"vercel": {"url": "https://mcp.vercel.com", "enabled": False, "auth": "oauth"}},
    }), encoding="utf-8")
    (cto / "mcp-tokens" / "vercel.json").write_text("super-secret-token", encoding="utf-8")

    from hermes_cli.capability_registry import find_capability

    result = find_capability("mcp:vercel", requester_profile="cmo", include_disabled=True)
    assert result.profiles
    cap = result.profiles[0]
    assert cap.profile == "cto"
    assert cap.configured is True
    assert cap.enabled is False
    assert cap.credential_present is True
    assert cap.credential_check == "disabled"
    assert "super-secret-token" not in str(result.to_dict())


def test_composio_vercel_route_beats_disabled_native_vercel(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "mcp_servers": {"composio": {"url": "https://connect.composio.dev/mcp"}},
    }), encoding="utf-8")
    cto = home / "profiles" / "cto"
    (cto / "config.yaml").write_text(yaml.safe_dump({
        "mcp_servers": {"vercel": {"enabled": False, "auth": "oauth"}},
    }), encoding="utf-8")
    (cto / "mcp-tokens" / "vercel.json").write_text("native-secret", encoding="utf-8")

    from hermes_cli.capability_registry import find_capability

    result = find_capability("mcp:vercel", requester_profile="cmo", include_disabled=True)
    assert result.recommendation["best_profile"] == "default"
    best = result.profiles[0]
    assert best.kind == "composio"
    assert best.source == "mcp:composio/toolkit:vercel"
    assert best.executable is True
    assert "native-secret" not in str(result.to_dict())


def test_workload_aware_ranking_prefers_less_busy_executor(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    for profile in ("cto", "coo"):
        p = home / "profiles" / profile
        (p / "mcp-tokens").mkdir(parents=True, exist_ok=True)
        (p / "config.yaml").write_text(yaml.safe_dump({
            "mcp_servers": {"vercel": {"enabled": True, "auth": "oauth"}},
        }), encoding="utf-8")
        (p / "mcp-tokens" / "vercel.json").write_text("secret", encoding="utf-8")

    from hermes_cli import kanban_db as kb
    from hermes_cli.capability_registry import find_capability

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="busy", assignee="cto", initial_status="running")
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
        conn.commit()

    result = find_capability("mcp:vercel", requester_profile="cmo", include_disabled=False, max_concurrency=2)
    assert result.profiles[0].profile == "coo"
    cto = next(c for c in result.profiles if c.profile == "cto")
    assert cto.workload.running_count == 1
    assert "running_count=1" in cto.rank_reasons
