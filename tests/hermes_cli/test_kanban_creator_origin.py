"""All explicit creator paths share durable lineage without graph coupling."""
import pytest


@pytest.mark.parametrize("surface", ["db", "builtin", "cli"])
def test_creator_origin_survives_without_dependency_parent(tmp_path, monkeypatch, capsys, surface):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kn
    from hermes_cli.kanban_db_graph import decompose_triage_task

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        owner = kb.create_task(conn, title="owner", session_id="durable", triage=True)
        kn.add_notify_sub(conn, task_id=owner, platform="telegram", chat_id="chat",
                         delivery_mode="wake", notifier_profile="default")
        if surface == "builtin":
            tid = decompose_triage_task(conn, owner, root_assignee="default",
                                       children=[{"title": "child"}])[0]
        elif surface == "db":
            tid = kb.create_task(conn, title="child", creator_task_id=owner)
        else:
            import json
            import argparse
            from hermes_cli.kanban import kanban_command
            from hermes_cli.kanban_parser import build_parser
            parser = argparse.ArgumentParser()
            build_parser(parser.add_subparsers())
            monkeypatch.setenv("HERMES_KANBAN_TASK", owner)
            assert kanban_command(parser.parse_args(["kanban", "create", "child", "--json"])) == 0
            tid = json.loads(capsys.readouterr().out)["id"]
        assert kb.get_task(conn, tid).session_id == "durable"
        subs = kn.list_notify_subs(conn, tid)
        assert len(subs) == 1 and subs[0]["delivery_mode"] == "wake"
        assert not conn.execute("SELECT 1 FROM task_links WHERE child_id = ?", (tid,)).fetchone()
        # No ambient identity guessing in the storage API.
        plain = kb.create_task(conn, title="plain", session_id="explicit")
        assert kb.get_task(conn, plain).session_id == "explicit"
        assert not kn.list_notify_subs(conn, plain)
