from gateway.dispatcher import dispatch_pm_message, dispatch_workflow_delivery
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.message_renderer import sanitize_user_visible_message
from gateway.message_validator import validate_message_events
from gateway.task_state import (
    add_pending_task,
    create_task,
    create_workflow,
    get_active_workflow,
    reset_state,
)
from gateway.session import SessionSource
from gateway.workflow_profile import load_workflow_profile
from gateway.platforms.base import MessageEvent, MessageType
from types import SimpleNamespace


def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WORKFLOW_STATE_PATH", str(tmp_path / "active_workflows.json"))
    reset_state()


def _director_profile():
    return load_workflow_profile("director_group")


def test_rough_creative_request_dispatches_to_direction_decision(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)

    result = dispatch_pm_message("PM 做个恐怖片", profile=_director_profile())

    assert result["intent"] == "rough_creative_request"
    assert result["matched_capability"] == "direction_decision"
    assert result["matched_role"] == "dd"
    assert result["task_count"] == 1
    assert result["message_event_count"] == 1
    assert result["next_state"] == "waiting_for_dd"
    assert result["validator"] == "pass"
    assert get_active_workflow("director_group")["workflow_id"] == result["workflow"]["workflow_id"]
    active = get_active_workflow("director_group")
    task = active["pending_tasks"][0]
    assert task["status"] == "send_pending"
    assert task["task_id"].startswith("task_")
    assert task["deliver_to_role"] == "pm"
    assert task["reviewer_role"] == ""
    assert task["deliver_after_review_to_role"] == "pm"
    assert [item["status"] for item in task["status_history"]] == ["created", "validated", "send_pending"]
    event = result["message_events"][0]
    assert "task_id" not in event["content"]
    assert "waiting_for_dd" not in event["content"]
    assert event["metadata"]["task_id"] == task["task_id"]
    assert event["metadata"]["deliver_to_role"] == "pm"
    assert event["metadata"]["reviewer_role"] == ""


def test_continue_requires_active_workflow_and_uses_next_action(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)
    profile = _director_profile()

    missing = dispatch_pm_message("PM 继续", profile=profile)

    assert missing["result"] == "need_judgement"
    assert missing["reason"] == "没有明确当前工作流，不能盲目继续"

    create_workflow(
        profile_id="director_group",
        title="恐怖片",
        dispatcher_role="pm",
        current_node="waiting_for_dd",
        next_action="send_to_sy",
    )

    continued = dispatch_pm_message("PM 继续", profile=profile)

    assert continued["result"] == "continue_dispatch"
    assert continued["next_action"] == "send_to_sy"
    assert continued["validator"] == "pass"


def test_multi_role_separate_dispatch_creates_independent_events(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)
    profile = _director_profile()

    result = dispatch_pm_message("PM 分别发给 wz / art / by / fx，不要写在一条信息里", profile=profile)

    assert result["target_roles"] == ["wz", "art", "by", "fx"]
    assert len(result["tasks"]) == 4
    assert len(result["message_events"]) == 4
    assert result["merged_message"] is False
    assert result["validator"] == "pass"
    assert validate_message_events(result["message_events"], profile)["ok"] is True


def test_renderer_hides_internal_command_lines():
    text = sanitize_user_visible_message("tool.started\nhermes -p wz chat\n已派给 wz")

    assert text == "已派给 wz"


def test_renderer_filters_terminal_command_only_message():
    text = sanitize_user_visible_message(
        "terminal command: cd /Users/me/.hermes/hermes-agent && python -m pytest tests/gateway/test_x.py",
        fallback=None,
    )

    assert text is None


def test_renderer_filters_tool_args_json():
    text = sanitize_user_visible_message(
        '{"current_tool": "terminal", "args": {"command": "python -m pytest"}}',
        fallback=None,
    )

    assert text is None


def test_renderer_filters_traceback():
    text = sanitize_user_visible_message(
        'Traceback (most recent call last):\n  File "/Users/me/.hermes/x.py", line 1, in <module>\nRuntimeError: boom',
        fallback=None,
    )

    assert text is None


def test_renderer_preserves_human_status():
    text = sanitize_user_visible_message("已派给 WZ\n当前等待 FX 返回", fallback=None)

    assert text == "已派给 WZ\n当前等待 FX 返回"


def test_renderer_preserves_ordinary_final_response():
    text = sanitize_user_visible_message("这是普通聊天回复：PM 可以先判断恐怖片方向。", fallback=None)

    assert text == "这是普通聊天回复：PM 可以先判断恐怖片方向。"


def test_dispatcher_ignores_pm_without_workflow_profile(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)

    result = dispatch_pm_message("PM 做个恐怖片", profile=None)

    assert result == {"handled": False}
    assert get_active_workflow() is None


def test_gateway_resolves_workflow_profile_only_for_bound_chat():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={
                    "workflow_profiles": {
                        "oc_director_group": "director_group",
                    }
                },
            )
        }
    )

    unbound_source = SessionSource(platform=Platform.FEISHU, chat_id="oc_plain", chat_type="group")
    bound_source = SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group")

    assert runner._resolve_workflow_profile_for_source(unbound_source) is None
    assert runner._resolve_workflow_profile_for_source(bound_source).profile_id == "director_group"


def test_gateway_restores_stripped_feishu_dispatcher_mention(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    _isolated_state(tmp_path, monkeypatch)
    runner = object.__new__(GatewayRunner)
    profile = _director_profile()
    event = SimpleNamespace(
        text="做一段武打片",
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
        metadata={"mentioned_names": ["PM"], "feishu_bot_mentioned": True},
    )

    dispatch_text = runner._workflow_dispatch_text_for_event(event, profile)
    result = dispatch_pm_message(dispatch_text, profile=profile)

    assert dispatch_text == "pm 做一段武打片"
    assert result["handled"] is True
    assert result["intent"] == "rough_creative_request"
    assert result["matched_role"] == "dd"


def test_gateway_n_command_starts_workflow_without_pm_mention(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    _isolated_state(tmp_path, monkeypatch)
    runner = object.__new__(GatewayRunner)
    event = MessageEvent(
        text="/n 做一段武打片",
        message_type=MessageType.COMMAND,
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
    )

    dispatch_text = runner._workflow_dispatch_text_for_event(event, _director_profile())
    result = dispatch_pm_message(dispatch_text, profile=_director_profile())

    assert dispatch_text == "pm 做一段武打片"
    assert result["handled"] is True
    assert result["intent"] == "rough_creative_request"
    assert result["matched_role"] == "dd"


def test_gateway_n_command_without_profile_does_not_dispatch(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    _isolated_state(tmp_path, monkeypatch)
    runner = object.__new__(GatewayRunner)
    event = MessageEvent(
        text="/n 做一段武打片",
        message_type=MessageType.COMMAND,
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_plain", chat_type="group"),
    )

    dispatch_text = runner._workflow_dispatch_text_for_event(event, _director_profile())

    assert dispatch_pm_message(dispatch_text, profile=None) == {"handled": False}


def test_gateway_n_command_is_registered_for_gateway():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    assert "n" in GATEWAY_KNOWN_COMMANDS
    assert resolve_command("n").gateway_only is True


def test_gateway_does_not_restore_non_dispatcher_mention():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = SimpleNamespace(
        text="做一段武打片",
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
        metadata={"mentioned_names": ["CJ"], "feishu_bot_mentioned": True},
    )

    assert runner._workflow_dispatch_text_for_event(event, _director_profile()) == "做一段武打片"


def test_gateway_does_not_restore_dispatcher_name_without_bot_mention():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = SimpleNamespace(
        text="做一段武打片",
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
        metadata={"mentioned_names": ["PM"]},
    )

    assert runner._workflow_dispatch_text_for_event(event, _director_profile()) == "做一段武打片"


def test_gateway_does_not_restore_dispatcher_mention_outside_feishu():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = SimpleNamespace(
        text="做一段武打片",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="group"),
        metadata={"mentioned_names": ["PM"]},
    )

    assert runner._workflow_dispatch_text_for_event(event, _director_profile()) == "做一段武打片"


def test_executor_return_updates_task_state_and_records_summary(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)
    profile = _director_profile()
    result = dispatch_pm_message("PM 跑一段武侠片", profile=profile)
    workflow_id = result["workflow"]["workflow_id"]
    task_id = result["tasks"][0]["task_id"]

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._get_current_profile_name = lambda: "dd"
    delivered = []

    async def _capture_delivery(**kwargs):
        delivered.append(kwargs)
        return True

    runner._workflow_enqueue_delivery_relay = _capture_delivery
    event = SimpleNamespace(raw_message={"workflow_id": workflow_id, "task_id": task_id}, message_id="om_dd")
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group")

    import asyncio

    asyncio.run(
        runner._workflow_complete_role_turn(
            event=event,
            source=source,
            profile=profile,
            response="这段武侠片要走冷峻、克制的江湖气质，冲突放在师徒诀别。",
        )
    )

    active = get_active_workflow("director_group")
    assert active["pending_tasks"] == []
    completed = active["completed_tasks"][0]
    assert completed["task_id"] == task_id
    assert completed["status"] == "returned_to_pm"
    assert completed["completed_by_role"] == "dd"
    assert completed["returned_to_role"] == "pm"
    assert "冷峻" in completed["result_summary"]
    assert completed["completed_at"]
    assert delivered[0]["relay_type"] == "workflow_task_return"
    assert delivered[0]["to_role"] == "pm"
    assert delivered[0]["task_id"] == task_id


def test_executor_return_without_reviewer_auto_delivers_to_pm(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)
    profile = _director_profile()
    result = dispatch_pm_message("PM 跑一段武侠片", profile=profile)
    task = result["tasks"][0]

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._get_current_profile_name = lambda: "dd"
    delivered = []

    async def _capture_delivery(**kwargs):
        delivered.append(kwargs)
        return True

    runner._workflow_enqueue_delivery_relay = _capture_delivery

    import asyncio

    asyncio.run(
        runner._workflow_complete_role_turn(
            event=SimpleNamespace(raw_message={"workflow_id": task["workflow_id"], "task_id": task["task_id"]}, message_id="om_dd"),
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
            profile=profile,
            response="方向判断：武侠要走雨夜、旧债、克制爆发。",
        )
    )

    assert len(delivered) == 1
    assert delivered[0]["to_role"] == "pm"
    assert delivered[0]["from_role"] == "dd"
    assert "总导演已经回来了" in delivered[0]["text"]
    assert "武侠要走雨夜" in delivered[0]["upstream_summary"]


def test_executor_return_to_pm_uses_internal_relay_not_visible_send(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)
    profile = _director_profile()
    result = dispatch_pm_message("PM 跑一段武侠片", profile=profile)
    task = result["tasks"][0]

    from gateway.run import GatewayRunner

    class _Adapter:
        def __init__(self):
            self.sent = []

        async def send(self, chat_id, content, metadata=None):
            self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata or {}})
            return SimpleNamespace(success=True, message_id="om_delivery")

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner._get_current_profile_name = lambda: "dd"
    runner.adapters = {Platform.FEISHU: adapter}
    delivered = []

    async def _capture_delivery(**kwargs):
        delivered.append(kwargs)
        return True

    runner._workflow_enqueue_delivery_relay = _capture_delivery

    import asyncio

    asyncio.run(
        runner._workflow_complete_role_turn(
            event=SimpleNamespace(raw_message={"workflow_id": task["workflow_id"], "task_id": task["task_id"]}, message_id="om_dd"),
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
            profile=profile,
            response="方向判断：武侠要走雨夜、旧债、克制爆发。",
        )
    )

    assert adapter.sent == []
    assert len(delivered) == 1
    assert delivered[0]["relay_type"] == "workflow_task_return"
    assert delivered[0]["to_role"] == "pm"
    assert delivered[0]["task_id"] == task["task_id"]
    assert "武侠要走雨夜" in delivered[0]["text"]


def test_executor_return_with_reviewer_waits_for_review_then_returns_pm(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)
    profile = _director_profile()
    workflow = create_workflow(profile_id="director_group", title="武侠片", dispatcher_role="pm")
    task = create_task(
        workflow_id=workflow["workflow_id"],
        to_role="wz",
        task_type="action_design",
        deliverable="动作结构",
        instruction="拆动作结构",
        return_to="pm",
        reviewer_role="cj",
        deliver_to_role="pm",
    )
    add_pending_task(workflow["workflow_id"], task)
    from gateway import task_state

    task_state.mark_task_received(workflow_id=workflow["workflow_id"], task_id=task["task_id"])

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    delivered = []

    async def _capture_delivery(**kwargs):
        delivered.append(kwargs)
        return True

    runner._workflow_enqueue_delivery_relay = _capture_delivery

    import asyncio

    runner._get_current_profile_name = lambda: "wz"
    asyncio.run(
        runner._workflow_complete_role_turn(
            event=SimpleNamespace(raw_message={"workflow_id": workflow["workflow_id"], "task_id": task["task_id"]}, message_id="om_wz"),
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
            profile=profile,
            response="动作结构先压住，最后一招爆发。",
        )
    )

    active = get_active_workflow("director_group")
    pending = active["pending_tasks"][0]
    assert pending["status"] == "waiting_review"
    assert pending["result_summary"].startswith("动作结构")
    assert delivered[-1]["relay_type"] == "workflow_review_request"
    assert delivered[-1]["to_role"] == "cj"

    runner._get_current_profile_name = lambda: "cj"
    asyncio.run(
        runner._workflow_complete_role_turn(
            event=SimpleNamespace(raw_message={"workflow_id": workflow["workflow_id"], "task_id": task["task_id"]}, message_id="om_cj"),
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
            profile=profile,
            response="通过，连续性成立。",
        )
    )

    active = get_active_workflow("director_group")
    assert active["pending_tasks"] == []
    reviewed = active["completed_tasks"][0]
    assert reviewed["status"] == "returned_to_pm"
    assert reviewed["reviewed_by_role"] == "cj"
    assert reviewed["review_passed"] is True
    assert delivered[-1]["relay_type"] == "workflow_task_return"
    assert delivered[-1]["to_role"] == "pm"


def test_pm_followup_dispatch_carries_upstream_summary_to_next_role(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)
    profile = _director_profile()
    result = dispatch_pm_message("PM 跑一段武侠片", profile=profile)
    workflow_id = result["workflow"]["workflow_id"]
    task_id = result["tasks"][0]["task_id"]

    from gateway import task_state

    task_state.record_task_result(
        workflow_id=workflow_id,
        task_id=task_id,
        result_text="总导演方向：雨夜、旧债、克制爆发。",
        completed_by_role="dd",
    )

    followup = dispatch_workflow_delivery(
        {
            "relay_type": "workflow_task_return",
            "workflow_id": workflow_id,
            "task_id": task_id,
            "upstream_task_id": task_id,
            "upstream_summary": "总导演方向：雨夜、旧债、克制爆发。",
            "from_role": "dd",
            "to_role": "pm",
        },
        profile=profile,
    )

    assert followup["handled"] is True
    assert followup["intent"] == "followup_from_upstream"
    assert followup["tasks"][0]["to_role"] == "wz"
    assert followup["tasks"][0]["upstream_role"] == "dd"
    assert "旧债" in followup["tasks"][0]["upstream_summary"]
    assert "旧债" in followup["message_events"][0]["content"]
    assert "接着做动作设计" not in followup["message_events"][0]["content"]


def test_plain_role_chat_without_active_task_is_not_treated_as_delivery(tmp_path, monkeypatch):
    _isolated_state(tmp_path, monkeypatch)

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._get_current_profile_name = lambda: "dd"
    delivered = []

    async def _capture_delivery(**kwargs):
        delivered.append(kwargs)
        return True

    runner._workflow_enqueue_delivery_relay = _capture_delivery

    import asyncio

    asyncio.run(
        runner._workflow_complete_role_turn(
            event=SimpleNamespace(raw_message={}, message_id="om_plain"),
            source=SessionSource(platform=Platform.FEISHU, chat_id="oc_director_group", chat_type="group"),
            profile=_director_profile(),
            response="我只是普通发言。",
        )
    )

    assert delivered == []
    assert get_active_workflow("director_group") is None
