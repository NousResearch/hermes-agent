from __future__ import annotations

from hermes_cli import kanban_db as kb
from proactive.clawops_intake import (
    auto_publish_preapproved,
    create_clawops_task,
    infer_clawops_metadata,
    resolve_clawops_assignee,
    subscribe_clawops_task,
)
from proactive.hubops_routing import route_clawops_objective


def test_resolve_clawops_assignee_prefers_env(monkeypatch):
    monkeypatch.setenv("HERMES_CLAWOPS_ASSIGNEE", "ops-runtime")

    assert resolve_clawops_assignee({"clawops": {"default_assignee": "config-agent"}}) == "ops-runtime"


def test_resolve_clawops_assignee_falls_back_to_default_profile(monkeypatch):
    monkeypatch.delenv("HERMES_CLAWOPS_ASSIGNEE", raising=False)

    assert resolve_clawops_assignee({}) == "default"


def test_create_clawops_task_writes_task_and_created_event(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_CLAWOPS_ASSIGNEE", "clawops-test")

    task = create_clawops_task(
        "verify proactive runtime queue",
        source={"platform": "telegram", "chat_id": "chat-1", "user_id": "kj"},
    )

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)
        events = kb.list_events(conn, task.task_id)

    assert row is not None
    assert row.title == "ClawOps: verify proactive runtime queue"
    assert row.assignee == "clawops-test"
    assert row.status == "ready"
    assert row.created_by == "hermes-clawops-intake"
    assert "Hermes remains the primary agent" in row.body
    assert "platform: telegram" in row.body
    assert [event.kind for event in events] == ["created"]


def test_hubops_routing_selects_dev_worker_from_yaml():
    envelope = route_clawops_objective(
        "修正 Hermes bridge health check",
        project="hub_ops",
        task_type="devops",
        risk_level="low",
    )

    assert envelope["status"] == "routed"
    assert envelope["assignment"]["assigned_worker"] == "clawops.dev"
    assert envelope["assignment"]["approval_required"] is False
    assert envelope["assignment"]["timeout_seconds"] == 1800
    assert envelope["approval_checklist"] == "DevOps and Integration"


def test_hubops_routing_selects_secondhand_agent_and_browser_worker():
    envelope = route_clawops_objective(
        "繼續追加 Facebook 社團群組發佈，再10個",
        project="secondhand_commerce",
        task_type="browser_publish",
        risk_level="medium",
        approved=True,
    )

    assert envelope["status"] == "routed"
    assert envelope["agent_assignment"]["assigned_agent"] == "secondhand_commerce"
    assert envelope["assignment"]["assigned_worker"] == "clawops.browser"
    assert envelope["assignment"]["runtime_profile"] == "clawops-browser"
    assert "browser_upload_files" in envelope["assignment"]["allowed_tools"]
    assert envelope["approval_checklist"] == "External Browser Publish"


def test_hubops_routing_blocks_unapproved_high_risk_work():
    envelope = route_clawops_objective(
        "部署 OpenClaw bridge 到 production",
        project="hub_ops",
        task_type="devops",
        risk_level="high",
        approved=False,
    )

    assert envelope["status"] == "blocked"
    assert envelope["assignment"]["assigned_worker"] == "clawops.dev"
    assert "approval" in envelope["blocked_reason"].lower()


def test_create_clawops_task_embeds_hubops_assignment_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    task = create_clawops_task(
        "修正 Hermes bridge health check",
        source={
            "project": "hub_ops",
            "task_type": "devops",
            "risk_level": "low",
            "approved": "false",
        },
    )

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)

    assert row is not None
    assert row.assignee == "clawops-dev"
    assert "HubOps routing:" in row.body
    assert "assigned_worker: clawops.dev" in row.body
    assert "runtime_profile: clawops-dev" in row.body
    assert "approval_checklist: DevOps and Integration" in row.body


def test_facebook_clawops_task_declares_browser_upload_capabilities(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    task = create_clawops_task(
        "請繼續 #7 咖啡器材新舊交流團的刊登流程，只允許點 Next，不要送出刊登",
        source={"platform": "telegram", "chat_id": "chat-1"},
    )

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)

    assert row is not None
    assert "External browser capability contract:" in row.body
    assert "browser_upload_files" in row.body
    assert "BROWSER_CDP_URL" in row.body
    assert "kanban_block" in row.body
    assert "Post/Publish/Submit" in row.body
    assert "stop before Post/Publish/Submit" in row.body
    assert "Approved-copy auto-publish" not in row.body
    assert "Hermes must not perform this browser UI work directly" in row.body


def test_facebook_preapproved_copy_task_allows_auto_publish(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    task = create_clawops_task(
        "Facebook 社團刊登：之前發佈文案 Hermes 已經傳給我確認過了，後續自動發佈",
        source={
            "platform": "telegram",
            "chat_id": "chat-1",
            "auto_publish_preapproved": "true",
        },
    )

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)

    assert row is not None
    assert "Approved-copy auto-publish" in row.body
    assert "may click final Post/Publish/Submit without asking KJ again" in row.body
    assert "visible Facebook content differs from the confirmed copy/assets" in row.body
    assert "capture URL/screenshot/page state" in row.body
    assert "stop before Post/Publish/Submit" not in row.body


def test_facebook_preapproved_copy_task_uses_browser_capable_runtime_not_openclaw_dry_run(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    task = create_clawops_task(
        "繼續追加 Facebook 社團群組發佈，再10個。之前發佈文案 Hermes 已經傳給我確認過；後續自動發佈",
        source={
            "platform": "telegram",
            "chat_id": "chat-1",
            "auto_publish_preapproved": "true",
            "previous_copy_confirmed": "true",
        },
    )

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)

    assert row is not None
    assert "browser-capable Hermes/ClawOps runtime" in row.body
    assert "This task must be executed by ClawOps/OpenClaw" not in row.body
    assert "ClawOps/OpenClaw may execute only delegated work" not in row.body
    assert "Hermes must not perform this browser UI work directly" not in row.body
    assert "Do not route this task through the OpenClaw dry-run bridge" in row.body


def test_natural_language_secondhand_facebook_publish_routes_to_browser_worker(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    task = create_clawops_task(
        "繼續追加 Facebook 社團群組發佈，再10個。之前發佈文案 Hermes 已經傳給 KJ 確認過；後續自動發佈",
        source={"platform": "telegram", "chat_id": "chat-1"},
    )

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)

    assert row is not None
    assert row.assignee == "clawops-browser"
    assert "project: secondhand_commerce" in row.body
    assert "task_type: browser_publish" in row.body
    assert "assigned_agent: secondhand_commerce" in row.body
    assert "assigned_worker: clawops.browser" in row.body
    assert "runtime_profile: clawops-browser" in row.body
    assert "browser_upload_files" in row.body


def test_infer_clawops_metadata_covers_known_project_agents():
    cases = [
        ("請規劃 Hahow 課程大綱", "hahow_course", "course_design"),
        ("請規劃課程招生行銷活動", "course_marketing", "campaign"),
        ("二手咖啡機 Facebook 社團發佈", "secondhand_commerce", "browser_publish"),
        ("ingrids SEO 內容規劃", "ingrids_marketing", "product_marketing"),
        ("修正 OpenClaw bridge health check", "hub_ops", "devops"),
    ]

    for objective, project, task_type in cases:
        inferred = infer_clawops_metadata(objective, source={})
        assert inferred["project"] == project
        assert inferred["task_type"] == task_type


def test_auto_publish_preapproved_requires_explicit_signal():
    assert auto_publish_preapproved("Facebook 社團刊登，文案已確認，請自動發佈") is True
    assert auto_publish_preapproved(
        "Facebook 社團刊登",
        source={"copy_approved": "approved"},
    ) is True
    assert auto_publish_preapproved("Facebook 社團刊登，請先檢查") is False


def test_non_browser_clawops_task_does_not_add_browser_capability_contract(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    task = create_clawops_task("summarize local runtime logs")

    with kb.connect_closing(db_path) as conn:
        row = kb.get_task(conn, task.task_id)

    assert row is not None
    assert "External browser capability contract:" not in row.body


def test_subscribe_clawops_task_writes_notify_subscription(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    task = create_clawops_task("watch terminal update path", assignee="clawops-test")

    subscribed = subscribe_clawops_task(
        task.task_id,
        platform="telegram",
        chat_id="chat-1",
        thread_id="thread-1",
        user_id="kj",
        notifier_profile="main",
    )

    with kb.connect_closing(db_path) as conn:
        subs = kb.list_notify_subs(conn, task.task_id)

    assert subscribed is True
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["thread_id"] == "thread-1"
    assert subs[0]["user_id"] == "kj"
    assert subs[0]["notifier_profile"] == "main"


def test_decomposed_clawops_children_inherit_root_subscription(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    task = create_clawops_task("coordinate multi-step publish flow", assignee="default")

    subscribed = subscribe_clawops_task(
        task.task_id,
        platform="telegram",
        chat_id="chat-1",
        user_id="kj",
        notifier_profile="main",
    )
    assert subscribed is True

    with kb.connect_closing(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id = ?",
            (task.task_id,),
        )
        child_ids = kb.decompose_triage_task(
            conn,
            task.task_id,
            root_assignee="default",
            children=[
                {"title": "Draft Marketplace listing", "assignee": "default"},
                {
                    "title": "Compile final report",
                    "assignee": "default",
                    "parents": [0],
                },
            ],
            author="auto-decomposer",
        )
        assert child_ids is not None

        root_subs = kb.list_notify_subs(conn, task.task_id)
        first_child_subs = kb.list_notify_subs(conn, child_ids[0])
        second_child_subs = kb.list_notify_subs(conn, child_ids[1])

    assert len(root_subs) == 1
    assert len(first_child_subs) == 1
    assert len(second_child_subs) == 1
    assert first_child_subs[0]["platform"] == "telegram"
    assert first_child_subs[0]["chat_id"] == "chat-1"
    assert first_child_subs[0]["user_id"] == "kj"
    assert first_child_subs[0]["notifier_profile"] == "main"
    assert second_child_subs[0]["chat_id"] == "chat-1"
