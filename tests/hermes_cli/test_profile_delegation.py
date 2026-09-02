import yaml


def _home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    for profile in ("cto", "coo", "cmo"):
        (home / "profiles" / profile / "mcp-tokens").mkdir(parents=True, exist_ok=True)
    return home


def _enable_cap(home, profile, server="vercel"):
    p = home / "profiles" / profile
    (p / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {server: {"enabled": True, "auth": "oauth"}}}), encoding="utf-8")
    (p / "mcp-tokens" / f"{server}.json").write_text("secret", encoding="utf-8")


def test_auto_executor_selection_and_delegated_read_completes(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    _enable_cap(home, "cto")

    from hermes_cli import kanban_db as kb
    from hermes_cli.profile_delegation import ProfileDelegationRequest, delegate_to_profile

    def spawn(task, workspace, board):
        deleg = kb.get_profile_delegation_by_task(conn, task.id)
        kb.complete_task(
            conn,
            task.id,
            summary="Vercel inspected.",
            result="No secrets.",
            metadata={"profile_delegation": {
                "delegation_id": deleg["id"],
                "capability": "mcp:vercel",
                "risk": "READ",
                "status": "completed",
                "structured_result": {"project": "ConnectMe", "status": "ok"},
                "redaction": {"secrets_returned": False},
            }},
        )
        return 12345

    with kb.connect_closing() as conn:
        req = ProfileDelegationRequest(
            profile=None,
            task="Inspect Vercel project ConnectMe status",
            required_capability="mcp:vercel",
            requester_profile="cmo",
            timeout_seconds=1,
            max_concurrency=2,
        )
        result = delegate_to_profile(req, spawn_fn=spawn)

    assert result.status == "completed"
    assert result.executor_profile == "cto"
    assert result.result["project"] == "ConnectMe"
    assert result.result["status"] == "ok"
    assert result.ranking[0]["profile"] == "cto"


def test_composio_route_is_embedded_in_worker_prompt(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "mcp_servers": {"composio": {"url": "https://connect.composio.dev/mcp"}},
    }), encoding="utf-8")

    from hermes_cli.profile_delegation import ProfileDelegationRequest, build_delegation_worker_body, select_executor

    chosen, ranking, reason = select_executor(required_capability="mcp:vercel", requester_profile="cmo")
    assert chosen.profile == "default"
    assert chosen.kind == "composio"
    body = build_delegation_worker_body(
        ProfileDelegationRequest(profile=None, task="Inspect Vercel through Composio", required_capability="mcp:vercel", requester_profile="cmo"),
        "pd_test",
        chosen.profile,
        chosen,
    )
    assert "Use Composio for this capability" in body
    assert "Do not try to enable or use the native Vercel MCP" in body


def test_consequential_write_is_blocked_before_task_spawn(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    _enable_cap(home, "cto")

    from hermes_cli import kanban_db as kb
    from hermes_cli.profile_delegation import ProfileDelegationRequest, delegate_to_profile

    spawned = []
    def spawn(task, workspace, board):
        spawned.append(task.id)
        return 1

    req = ProfileDelegationRequest(
        profile=None,
        task="Deploy ConnectMe to production on Vercel",
        required_capability="mcp:vercel",
        requester_profile="cmo",
        risk="CONSEQUENTIAL_WRITE",
        timeout_seconds=0,
    )
    result = delegate_to_profile(req, spawn_fn=spawn)
    assert result.status == "blocked_approval"
    assert not spawned
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_busy_executor_ranking_uses_running_count(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    _enable_cap(home, "cto")
    _enable_cap(home, "coo")

    from hermes_cli import kanban_db as kb
    from hermes_cli.profile_delegation import ProfileDelegationRequest, select_executor

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="busy", assignee="cto", initial_status="running")
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
        conn.commit()
    chosen, ranking, reason = select_executor(required_capability="mcp:vercel", requester_profile="cmo", max_concurrency=2)
    assert chosen.profile == "coo"
    assert any(r["profile"] == "cto" and r["workload"]["running_count"] == 1 for r in ranking)
    assert reason == "auto_selected_workload_aware_executor"


def test_executor_failure_returns_structured_error(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    _enable_cap(home, "cto")

    from hermes_cli import kanban_db as kb
    from hermes_cli.profile_delegation import ProfileDelegationRequest, delegate_to_profile

    def spawn(task, workspace, board):
        kb.block_task(conn, task.id, reason="Vercel MCP authentication failed")
        return 222

    with kb.connect_closing() as conn:
        result = delegate_to_profile(ProfileDelegationRequest(
            profile="cto", task="Inspect Vercel", required_capability="mcp:vercel", requester_profile="cmo", timeout_seconds=1,
        ), spawn_fn=spawn)
    assert result.status == "failed"
    assert "Vercel MCP authentication failed" in (result.error or "")
