import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("HERMES_WORKFLOW_STATE_PATH", str(tmp_path / "active_workflows.json"))


def test_record_outbound_success_can_be_read_back():
    from gateway.outbound_log import get_outbound, list_outbounds, record_outbound

    outbound_id = record_outbound(
        platform="feishu",
        chat_id="oc_success",
        chat_type="group",
        source_message_id="om_inbound",
        reply_to_message_id="om_parent",
        send_type="feishu_real_send",
        workflow_id="wf_1",
        task_id="task_1",
        to_role="wz",
        target_role="PM",
        target_profile="pm",
        content="hello from hermes",
        send_success=True,
        feishu_message_id="om_sent",
        raw_response_summary='{"success": true}',
    )

    row = get_outbound(outbound_id)

    assert row is not None
    assert row["outbound_id"] == outbound_id
    assert row["platform"] == "feishu"
    assert row["chat_id"] == "oc_success"
    assert row["chat_type"] == "group"
    assert row["source_message_id"] == "om_inbound"
    assert row["reply_to_message_id"] == "om_parent"
    assert row["send_type"] == "feishu_real_send"
    assert row["workflow_id"] == "wf_1"
    assert row["task_id"] == "task_1"
    assert row["to_role"] == "wz"
    assert row["target_role"] == "PM"
    assert row["target_profile"] == "pm"
    assert row["send_success"] is True
    assert row["real_sent"] is True
    assert row["feishu_message_id"] == "om_sent"
    assert row["content_preview"] == "hello from hermes"
    assert len(row["content_hash"]) == 64

    rows = list_outbounds(chat_id="oc_success", send_success=True)
    assert [item["outbound_id"] for item in rows] == [outbound_id]


def test_record_outbound_failure_can_be_read_back():
    from gateway.outbound_log import get_outbound, list_outbounds, record_outbound

    outbound_id = record_outbound(
        platform="feishu",
        chat_id="oc_failure",
        send_type="feishu_real_send",
        content="message that failed",
        send_success=False,
        error="[400] bad request",
        raw_response_summary='{"code": 400}',
    )

    row = get_outbound(outbound_id)

    assert row is not None
    assert row["send_success"] is False
    assert row["real_sent"] is False
    assert row["error"] == "[400] bad request"
    assert row["feishu_message_id"] is None

    rows = list_outbounds(chat_id="oc_failure", send_success=False)
    assert [item["outbound_id"] for item in rows] == [outbound_id]


def test_feishu_send_success_records_message_id():
    from gateway.config import PlatformConfig
    from gateway.outbound_log import list_outbounds
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())

    class _MessageAPI:
        def create(self, request):
            return SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id="om_logged_success"),
            )

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI()))
    )

    async def _direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
        result = asyncio.run(adapter.send(chat_id="oc_logged", content="hello log"))

    assert result.success is True
    assert result.message_id == "om_logged_success"

    rows = list_outbounds(chat_id="oc_logged")
    assert len(rows) == 1
    assert rows[0]["send_success"] is True
    assert rows[0]["feishu_message_id"] == "om_logged_success"
    assert rows[0]["content_preview"] == "hello log"
    assert rows[0]["send_type"] == "feishu_real_send"


def test_feishu_send_failure_records_error():
    from gateway.config import PlatformConfig
    from gateway.outbound_log import list_outbounds
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())

    class _MessageAPI:
        def create(self, request):
            return SimpleNamespace(
                success=lambda: False,
                code=400,
                msg="bad request",
            )

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI()))
    )

    async def _direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
        result = asyncio.run(adapter.send(chat_id="oc_logged_failure", content="bad log"))

    assert result.success is False
    assert result.error == "[400] bad request"

    rows = list_outbounds(chat_id="oc_logged_failure")
    assert len(rows) == 1
    assert rows[0]["send_success"] is False
    assert rows[0]["error"] == "[400] bad request"
    assert rows[0]["content_preview"] == "bad log"


def test_feishu_send_result_is_unchanged_when_outbound_log_fails():
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())

    class _MessageAPI:
        def create(self, request):
            return SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id="om_send_still_ok"),
            )

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI()))
    )

    async def _direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct),
        patch("gateway.platforms.feishu.record_outbound", side_effect=RuntimeError("db down")),
    ):
        result = asyncio.run(adapter.send(chat_id="oc_unaffected", content="still sends"))

    assert result.success is True
    assert result.message_id == "om_send_still_ok"


def test_feishu_send_success_updates_task_state_real_sent():
    from gateway.config import PlatformConfig
    from gateway.outbound_log import list_outbounds
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import add_pending_task, create_task, create_workflow, get_active_workflow

    workflow = create_workflow(
        profile_id="director_group",
        title="恐怖片",
        dispatcher_role="pm",
        current_node="waiting_for_dd",
        next_action="wait_for_dd_return",
    )
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="dd",
        task_type="direction_decision",
        deliverable="方向判断",
        instruction="做个恐怖片",
        return_to="pm",
    )
    add_pending_task(workflow["workflow_id"], task)

    adapter = FeishuAdapter(PlatformConfig())

    class _MessageAPI:
        def create(self, request):
            return SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id="om_task_sent"),
            )

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI()))
    )

    metadata = {
        "profile_id": "director_group",
        "workflow_id": workflow["workflow_id"],
        "task_id": task["task_id"],
        "to_role": "dd",
        "target_role": "dd",
        "target_profile": "dd",
        "task_type": "direction_decision",
        "deliverable": "方向判断",
        "return_to": "pm",
        "send_mode": "real_send",
    }

    async def _direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
        result = asyncio.run(
            adapter.send(
                chat_id="oc_task_state",
                content="总导演\n任务：做个恐怖片\n交付：方向判断\n完成后回给：pm",
                metadata=metadata,
            )
        )

    assert result.success is True
    rows = list_outbounds(chat_id="oc_task_state")
    assert rows[0]["workflow_id"] == workflow["workflow_id"]
    assert rows[0]["task_id"] == task["task_id"]
    assert rows[0]["to_role"] == "dd"
    assert rows[0]["real_sent"] is True

    active = get_active_workflow("director_group")
    assert active["pending_tasks"][0]["real_sent"] is True
    assert active["pending_tasks"][0]["message_id"] == "om_task_sent"
    assert active["pending_tasks"][0]["sent_message_id"] == "om_task_sent"
    assert active["pending_tasks"][0]["status"] == "sent"
    assert active["pending_tasks"][0]["outbound_id"] == rows[0]["outbound_id"]
    assert [item["status"] for item in active["pending_tasks"][0]["status_history"]] == ["created", "sent"]


def test_feishu_send_failure_marks_task_failed():
    from gateway.config import PlatformConfig
    from gateway.outbound_log import list_outbounds
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import add_pending_task, create_task, create_workflow, get_active_workflow

    workflow = create_workflow(
        profile_id="director_group",
        title="恐怖片",
        dispatcher_role="pm",
        current_node="waiting_for_dd",
        next_action="wait_for_dd_return",
    )
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="dd",
        task_type="direction_decision",
        deliverable="方向判断",
        instruction="做个恐怖片",
        return_to="pm",
    )
    add_pending_task(workflow["workflow_id"], task)

    adapter = FeishuAdapter(PlatformConfig())

    class _MessageAPI:
        def create(self, request):
            return SimpleNamespace(success=lambda: False, code=500, msg="boom")

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI())))

    metadata = {
        "profile_id": "director_group",
        "workflow_id": workflow["workflow_id"],
        "task_id": task["task_id"],
        "to_role": "dd",
        "target_role": "dd",
        "target_profile": "dd",
        "task_type": "direction_decision",
        "deliverable": "方向判断",
        "return_to": "pm",
        "send_mode": "real_send",
    }

    async def _direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
        result = asyncio.run(
            adapter.send(
                chat_id="oc_task_failed",
                content="总导演\n任务：做个恐怖片\n交付：方向判断\n完成后回给：pm",
                metadata=metadata,
            )
        )

    assert result.success is False
    rows = list_outbounds(chat_id="oc_task_failed")
    assert rows[0]["send_success"] is False
    assert rows[0]["task_id"] == task["task_id"]

    active = get_active_workflow("director_group")
    assert active["pending_tasks"] == []
    assert active["blocked_tasks"][0]["task_id"] == task["task_id"]
    assert active["blocked_tasks"][0]["status"] == "failed"
    assert "boom" in active["blocked_tasks"][0]["error"]


def test_feishu_validator_block_marks_task_failed_and_logs_blocked():
    from gateway.config import PlatformConfig
    from gateway.outbound_log import list_outbounds
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import add_pending_task, create_task, create_workflow, get_active_workflow

    workflow = create_workflow(
        profile_id="director_group",
        title="恐怖片",
        dispatcher_role="pm",
        current_node="waiting_for_dd",
    )
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="dd",
        task_type="direction_decision",
        deliverable="方向判断",
        instruction="做个恐怖片",
        return_to="pm",
    )
    add_pending_task(workflow["workflow_id"], task)

    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace())))
    metadata = {
        "profile_id": "director_group",
        "workflow_id": workflow["workflow_id"],
        "task_id": task["task_id"],
        "to_role": "dd",
        "target_role": "dd",
        "target_profile": "dd",
        "task_type": "direction_decision",
        "return_to": "pm",
        "send_mode": "real_send",
    }

    result = asyncio.run(
        adapter.send(
            chat_id="oc_blocked",
            content="总导演\n任务：做个恐怖片\n完成后回给：pm",
            metadata=metadata,
        )
    )

    assert result.success is False
    rows = list_outbounds(chat_id="oc_blocked")
    assert rows[0]["send_type"] == "blocked_by_validator"
    assert rows[0]["send_success"] is False
    assert rows[0]["real_sent"] is False

    active = get_active_workflow("director_group")
    assert active["blocked_tasks"][0]["task_id"] == task["task_id"]
    assert active["blocked_tasks"][0]["status"] == "failed"
    assert "MISSING_DELIVERABLE" in active["blocked_tasks"][0]["error"]


def test_feishu_relay_success_marks_task_relay_enqueued():
    from gateway.config import PlatformConfig
    from gateway.outbound_log import list_outbounds
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import add_pending_task, create_task, create_workflow, get_active_workflow

    workflow = create_workflow(profile_id="director_group", title="多角色", dispatcher_role="pm")
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="wz",
        task_type="parallel_specialist_dispatch",
        deliverable="专项意见",
        instruction="写作意见",
        return_to="pm",
    )
    add_pending_task(workflow["workflow_id"], task)

    adapter = FeishuAdapter(PlatformConfig())
    adapter._relay_profile_name = "pm"
    adapter._bot_name = "PM"
    metadata = {
        "workflow_id": workflow["workflow_id"],
        "task_id": task["task_id"],
        "to_role": "wz",
        "target_role": "WZ",
        "target_profile": "wz",
        "chat_type": "group",
    }

    with patch("gateway.platforms.feishu._relay_enqueue", return_value=True):
        asyncio.run(
            adapter._enqueue_outbound_role_relays(
                chat_id="oc_relay",
                dispatches=[{"profile_name": "wz", "role_code": "WZ", "instruction": "写作意见"}],
                sent_message_id="om_sent",
                metadata=metadata,
            )
        )

    active = get_active_workflow("director_group")
    assert active["pending_tasks"][0]["status"] == "relay_enqueued"
    assert active["pending_tasks"][0]["relay_target_profile"] == "wz"
    rows = list_outbounds(chat_id="oc_relay", send_type="local_relay")
    assert len(rows) == 1
    assert rows[0]["send_success"] is True
    assert rows[0]["task_id"] == task["task_id"]


def test_feishu_workflow_delivery_relay_preserves_returned_state_and_metadata():
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import (
        add_pending_task,
        create_task,
        create_workflow,
        get_active_workflow,
        record_task_result,
    )

    workflow = create_workflow(profile_id="director_group", title="武侠", dispatcher_role="pm")
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="dd",
        task_type="direction_decision",
        deliverable="方向判断",
        instruction="跑一段武侠片",
        return_to="pm",
    )
    add_pending_task(workflow["workflow_id"], task)
    record_task_result(
        workflow_id=workflow["workflow_id"],
        task_id=task["task_id"],
        result_text="武侠要走雨夜、旧债。",
        completed_by_role="dd",
    )

    adapter = FeishuAdapter(PlatformConfig())
    adapter._relay_profile_name = "dd"
    adapter._bot_name = "DD"
    metadata = {
        "workflow_id": workflow["workflow_id"],
        "task_id": task["task_id"],
        "to_role": "pm",
        "target_role": "pm",
        "target_profile": "pm",
        "relay_type": "workflow_task_return",
        "workflow_delivery": "true",
        "from_role": "dd",
        "upstream_task_id": task["task_id"],
        "upstream_summary": "武侠要走雨夜、旧债。",
    }

    captured = {}

    def _capture_relay(**kwargs):
        captured.update(kwargs)
        return True

    with patch("gateway.platforms.feishu._relay_enqueue", side_effect=_capture_relay):
        asyncio.run(
            adapter._enqueue_outbound_role_relays(
                chat_id="oc_relay_delivery",
                dispatches=[{"profile_name": "pm", "role_code": "PM", "instruction": "PM，DD 已返回"}],
                sent_message_id="om_delivery",
                metadata=metadata,
            )
        )

    assert captured["relay_type"] == "workflow_task_return"
    assert captured["metadata"]["upstream_summary"] == "武侠要走雨夜、旧债。"
    active = get_active_workflow("director_group")
    assert active["completed_tasks"][0]["status"] == "returned_to_pm"


def test_feishu_relay_failure_marks_task_failed():
    from gateway.config import PlatformConfig
    from gateway.outbound_log import list_outbounds
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import add_pending_task, create_task, create_workflow, get_active_workflow

    workflow = create_workflow(profile_id="director_group", title="多角色", dispatcher_role="pm")
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="wz",
        task_type="parallel_specialist_dispatch",
        deliverable="专项意见",
        instruction="写作意见",
        return_to="pm",
    )
    add_pending_task(workflow["workflow_id"], task)

    adapter = FeishuAdapter(PlatformConfig())
    adapter._relay_profile_name = "pm"
    adapter._bot_name = "PM"
    metadata = {"workflow_id": workflow["workflow_id"], "task_id": task["task_id"], "to_role": "wz"}

    with patch("gateway.platforms.feishu._relay_enqueue", return_value=False):
        asyncio.run(
            adapter._enqueue_outbound_role_relays(
                chat_id="oc_relay_fail",
                dispatches=[{"profile_name": "wz", "role_code": "WZ", "instruction": "写作意见"}],
                sent_message_id="om_sent",
                metadata=metadata,
            )
        )

    active = get_active_workflow("director_group")
    assert active["pending_tasks"] == []
    assert active["blocked_tasks"][0]["status"] == "failed"
    assert "failed to enqueue relay" in active["blocked_tasks"][0]["error"]
    rows = list_outbounds(chat_id="oc_relay_fail", send_type="local_relay")
    assert rows[0]["send_success"] is False


def test_feishu_relay_envelope_marks_task_received():
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import add_pending_task, create_task, create_workflow, get_active_workflow

    workflow = create_workflow(profile_id="director_group", title="多角色", dispatcher_role="pm")
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="wz",
        task_type="parallel_specialist_dispatch",
        deliverable="专项意见",
        instruction="写作意见",
        return_to="pm",
    )
    add_pending_task(workflow["workflow_id"], task)

    adapter = FeishuAdapter(PlatformConfig())
    adapter._relay_profile_name = "wz"
    adapter._is_duplicate = lambda key: False
    adapter._add_ack_reaction = AsyncMock()
    adapter.get_chat_info = AsyncMock(return_value={"name": "导演组"})
    adapter._handle_message_with_guards = AsyncMock()

    asyncio.run(
        adapter._process_relay_envelope(
            {
                "chat_id": "oc_relay_received",
                "text": "写作意见",
                "sender_profile": "pm",
                "sender_display_name": "PM",
                "message_id": "om_origin",
                "workflow_id": workflow["workflow_id"],
                "task_id": task["task_id"],
            }
        )
    )

    active = get_active_workflow("director_group")
    assert active["pending_tasks"][0]["status"] == "received"
    assert active["pending_tasks"][0]["relay_message_id"] == "om_origin"


def test_plain_feishu_chat_does_not_create_task_state():
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.task_state import load_state

    adapter = FeishuAdapter(PlatformConfig())

    class _MessageAPI:
        def create(self, request):
            return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_plain"))

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=_MessageAPI())))

    async def _direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("gateway.platforms.feishu.asyncio.to_thread", side_effect=_direct):
        result = asyncio.run(adapter.send(chat_id="oc_plain", content="普通聊天"))

    assert result.success is True
    assert load_state()["workflows"] == []
