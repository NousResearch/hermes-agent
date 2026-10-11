"""Agent-origin admission is opt-in and fails closed for protected profiles."""
import json

from hermes_cli.profile_invocation_acl import permits
from tools import bot_mode_dm, bot_mode_probe


def test_policy_sources_and_malformed_config(tmp_path):
    root = tmp_path / ".hermes"
    root.mkdir()
    assert permits(None, "forge", root=root)  # legacy installs are unchanged
    (root / "config.yaml").write_text(
        "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker, forge-reviewer]\n"
        "    forge-worker: [forge, forge-worker, forge-reviewer]\n"
    )
    assert not permits(None, "forge", root=root)
    assert not permits("researcher", "forge", root=root)
    assert permits("forge-worker", "forge", root=root)
    assert permits("researcher", "ordinary", root=root)
    (root / "config.yaml").write_text("bot_mode:\n  invocation_acl: [forge]\n")
    assert not permits("forge", "forge", root=root)


def test_message_agent_blocks_default_to_forge_but_allows_forge_worker(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    for name in ("forge", "forge-worker"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "profile.yaml").write_text("ui_meta:\n  hermes-bots:\n    shape: cloud\n")
    class Agent:
        session_id = "session-acl"
        _session_title_hint = None
        _bot_mode_protocol = True

        def __init__(self, path):
            self._session_db = type("DB", (), {
                "db_path": str(path / "state.db"),
                "get_session_title": lambda _self, _sid: "Bot Chat",
            })()
    (home / "config.yaml").write_text(
        "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker]\n"
        "    forge-worker: [forge, forge-worker]\n"
    )
    bot_mode_probe._reset_cache_for_tests()
    calls = []
    monkeypatch.setattr(bot_mode_dm, "_start_delivery", lambda *a, **kw: calls.append(a) or '{"status":"queued"}')
    try:
        denied = json.loads(bot_mode_dm.message_agent_tool(target="forge", message="hi", agent=Agent(home)))
        assert "denied" in denied["error"]
        assert not calls
        forge_home = home / "profiles" / "forge"
        forge_agent = Agent(forge_home)
        allowed = json.loads(bot_mode_dm.message_agent_tool(target="forge-worker", message="hi", agent=forge_agent))
        assert allowed["status"] == "queued"
        assert len(calls) == 1
    finally:
        bot_mode_probe._reset_cache_for_tests()


def test_dispatch_rechecks_queued_card_creator(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch, profile_invocation_acl as acl
    from hermes_cli import profiles
    root = tmp_path / ".hermes"
    root.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(type(root), "home", lambda: tmp_path)
    monkeypatch.setattr(acl, "install_root", lambda: root)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    kb.init_db()
    with kbc.connect() as conn:
        outsider = kb.create_task(conn, title="outsider", assignee="forge", created_by="researcher")
        spoofed_operator = kb.create_task(conn, title="spoofed operator", assignee="forge", created_by="operator")
        spoofed_dashboard = kb.create_task(conn, title="spoofed dashboard", assignee="forge", created_by="dashboard")
        # CLI/database creator labels are display-only, even if they spell an allowed profile.
        spoofed_forge = kb.create_task(conn, title="spoofed Forge", assignee="forge", created_by="forge-worker")
        root.joinpath("config.yaml").write_text(
            "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker, forge-reviewer]\n")
        result = dispatch.dispatch_once(conn, dry_run=True)
        assert outsider in result.skipped_nonspawnable
        assert spoofed_operator in result.skipped_nonspawnable
        assert spoofed_dashboard in result.skipped_nonspawnable
        assert spoofed_forge in result.skipped_nonspawnable


def test_protected_dispatch_rechecks_current_target_and_review_lane(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch, profile_invocation_acl as acl
    from hermes_cli import profiles
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(type(root), "home", lambda: tmp_path)
    monkeypatch.setattr(acl, "install_root", lambda: root)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    kb.init_db()
    with kbc.connect() as conn:
        legacy = kb.create_task(conn, title="legacy Forge", assignee="ordinary", created_by="forge-worker")
        reviewed = kb.create_task(conn, title="review from CLI", assignee="forge-reviewer",
                                  created_by="forge-worker")
        conn.execute("UPDATE tasks SET assignee = 'forge' WHERE id = ?", (legacy,))
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (reviewed,))
        root.joinpath("config.yaml").write_text(
            "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker]\n"
            "    forge-reviewer: [forge, forge-worker]\n")
        decision = dispatch.dispatch_once(conn, dry_run=True)
        assert legacy in decision.skipped_nonspawnable
        assert reviewed in decision.skipped_nonspawnable


def test_manual_cli_claim_cannot_claim_protected_target(tmp_path, monkeypatch):
    import argparse
    from hermes_cli import kanban, kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import profile_invocation_acl as acl
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(type(root), "home", lambda: tmp_path)
    monkeypatch.setattr(acl, "install_root", lambda: root)
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="claimed?", assignee="forge", created_by="forge-worker")
    root.joinpath("config.yaml").write_text(
        "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker]\n")
    assert kanban._cmd_claim(argparse.Namespace(task_id=tid, ttl=300)) == 1
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def test_native_admission_cannot_be_reassigned_to_a_new_protected_target(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch, profile_invocation_acl as acl, profiles
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(type(root), "home", lambda: tmp_path)
    monkeypatch.setattr(acl, "install_root", lambda: root)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="originally ordinary", assignee="ordinary", created_by="forge-worker")
        kb._append_event(conn, tid, acl.NATIVE_KANBAN_ADMISSION,
                         {"origin": "native_kanban_tool", "source_profile": "forge-worker",
                          "target_profile": "ordinary", "lane": "ready"})
        kb.reassign_task(conn, tid, "forge")
        root.joinpath("config.yaml").write_text(
            "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker]\n")
        assert tid in dispatch.dispatch_once(conn, dry_run=True).skipped_nonspawnable


def test_acl_denied_review_does_not_reserve_ready_capacity(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch, profile_invocation_acl as acl, profiles
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(type(root), "home", lambda: tmp_path)
    monkeypatch.setattr(acl, "install_root", lambda: root)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(dispatch, "review_dispatch_enabled", lambda: True)
    kb.init_db()
    with kbc.connect() as conn:
        reviewed = kb.create_task(conn, title="unattested review", assignee="forge-reviewer")
        ordinary = kb.create_task(conn, title="ordinary ready", assignee="ordinary")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (reviewed,))
        root.joinpath("config.yaml").write_text(
            "bot_mode:\n  invocation_acl:\n    forge-reviewer: [forge, forge-worker]\n")
        result = dispatch.dispatch_once(conn, dry_run=True, max_spawn=1)
        assert ordinary in [tid for tid, *_ in result.spawned]
        assert reviewed not in [tid for tid, *_ in result.spawned]


def test_historical_review_admission_cannot_authorize_cli_replay(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch, profile_invocation_acl as acl, profiles
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(type(root), "home", lambda: tmp_path)
    monkeypatch.setattr(acl, "install_root", lambda: root)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(dispatch, "review_dispatch_enabled", lambda: True)
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review replay", assignee="forge-worker")
        assert kb.request_review(conn, tid, reviewer="forge-reviewer")
        first_review_event = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
            "ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()["id"]
        kb._append_event(conn, tid, acl.NATIVE_KANBAN_ADMISSION,
                         {"origin": "native_kanban_tool", "source_profile": "forge-worker",
                          "target_profile": "forge-reviewer", "lane": "review",
                          "review_event_id": first_review_event})
        assert kb.reopen_review_task(conn, tid)
        assert kb.reassign_task(conn, tid, "ordinary")
        assert kb.request_review(conn, tid, reviewer="forge-reviewer")
        root.joinpath("config.yaml").write_text(
            "bot_mode:\n  invocation_acl:\n    forge-reviewer: [forge-worker]\n")
        assert tid in dispatch.dispatch_once(conn, dry_run=True).skipped_nonspawnable
