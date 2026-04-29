from gateway.message_renderer import (
    render_dispatch_result,
    render_review_request_message,
    render_task_completion_delivery,
    render_task_message,
    render_workflow_template,
    sanitize_user_visible_message,
)
from gateway.message_validator import validate_message_event
from gateway.workflow_profile import load_workflow_profile


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


def test_renderer_uses_safe_fallback_for_internal_only_message():
    text = sanitize_user_visible_message(
        '{"current_tool": "terminal", "iteration": 1, "args": {"command": "python -m pytest"}}',
        fallback="正在处理，请稍候。",
    )

    assert text == "正在处理，请稍候。"


def test_renderer_preserves_human_status():
    text = sanitize_user_visible_message("已派给 WZ\n当前等待 FX 返回", fallback=None)

    assert text == "已派给 WZ\n当前等待 FX 返回"


def test_renderer_preserves_ordinary_final_response():
    text = sanitize_user_visible_message("这是普通聊天回复：PM 可以先判断恐怖片方向。", fallback=None)

    assert text == "这是普通聊天回复：PM 可以先判断恐怖片方向。"


def _director_profile():
    return load_workflow_profile("director_group")


def _task(**overrides):
    payload = {
        "task_id": "task_xxx",
        "to_role": "dd",
        "task_type": "direction_decision",
        "instruction": "做一段爱情片",
        "deliverable": "方向判断",
        "return_to": "pm",
        "reviewer_role": "",
        "deliver_to_role": "pm",
        "deliver_after_review_to_role": "pm",
        "upstream_role": "",
        "upstream_summary": "",
        "deliverable_format": "方向判断 + 下一步建议",
        "status": "send_pending",
    }
    payload.update(overrides)
    return payload


def test_human_dispatch_hides_task_id_and_state_codes():
    profile = _director_profile()
    result = {
        "intent": "rough_creative_request",
        "profile": profile,
        "tasks": [_task()],
        "next_state": "waiting_for_dd",
    }

    text = render_dispatch_result(result)

    assert "task_xxx" not in text
    assert "waiting_for_dd" not in text
    assert "relay_enqueued" not in text
    assert "交给总导演" in text
    assert "定一下方向" in text
    assert "做完后直接回 PM" in text


def test_human_task_message_for_dd_is_natural_language():
    profile = _director_profile()
    role = profile.get_role("dd")

    text = render_task_message(profile, role, _task())

    assert "task_id" not in text
    assert "任务：" not in text
    assert "当前状态" not in text
    assert "总导演，这单先麻烦你定方向" in text
    assert "用户这次的需求是：做一段爱情片" in text
    assert "这一步不用别人验收，做完后直接回 PM" in text


def test_human_task_message_with_reviewer_explains_review_flow():
    profile = _director_profile()
    role = profile.get_role("wz")

    text = render_task_message(
        profile,
        role,
        _task(
            to_role="wz",
            task_type="action_design",
            instruction="基于总导演方向拆动作结构",
            deliverable="动作结构",
            reviewer_role="cj",
            deliver_to_role="pm",
            deliver_after_review_to_role="pm",
        ),
    )

    assert "reviewer_role" not in text
    assert "deliver_after_review_to_role" not in text
    assert "先交给场记看一下" in text
    assert "场记确认没问题后，再回到 PM" in text
    assert "退回你这边改一轮" in text


def test_human_task_message_carries_upstream_summary_without_field_name():
    profile = _director_profile()
    role = profile.get_role("wz")

    text = render_task_message(
        profile,
        role,
        _task(
            to_role="wz",
            task_type="action_design",
            upstream_role="dd",
            upstream_summary="这段不要拍成撒糖恋爱，要拍关系被看见的那一刻。",
            instruction="拆肢体冲突结构",
            deliverable="动作结构",
        ),
    )

    assert "upstream_summary" not in text
    assert "总导演已经给过一轮判断" in text
    assert "这段不要拍成撒糖恋爱" in text
    assert "基于这个结果继续" in text


def test_debug_dispatch_can_show_task_id():
    profile = _director_profile()
    result = {
        "profile": profile,
        "tasks": [_task()],
        "next_state": "waiting_for_dd",
    }

    text = render_dispatch_result(result, display_mode="debug")

    assert "task_id=task_xxx" in text
    assert "next_state=waiting_for_dd" in text


def test_completion_delivery_first_line_only_addresses_target_role():
    profile = _director_profile()
    text = render_task_completion_delivery(
        profile,
        _task(),
        from_role="dd",
        to_role="pm",
        result_summary="武侠要走雨夜、旧债。",
    )

    assert text.splitlines()[0] == "总导演已经回来了，我先把结果收一下。"
    assert "武侠要走雨夜、旧债" in text
    assert validate_message_event(
        {
            "content": text,
            "to_role": "pm",
            "task_type": "workflow_task_delivery",
            "deliverable": "方向判断",
            "return_to": "pm",
            "metadata": {"send_mode": "real_send"},
        },
        profile,
    )["ok"] is True


def test_review_request_first_line_only_addresses_reviewer():
    profile = _director_profile()
    text = render_review_request_message(
        profile,
        _task(
            to_role="wz",
            task_type="action_design",
            deliverable="动作结构",
            reviewer_role="cj",
            deliver_to_role="pm",
        ),
        from_role="wz",
        reviewer_role="cj",
        result_summary="动作先压住，最后一招爆发。",
    )

    assert text.splitlines()[0] == "场记，这一步已经做完了，先麻烦你看一下。"
    assert "武指的结果大意是" in text
    assert validate_message_event(
        {
            "content": text,
            "to_role": "cj",
            "task_type": "workflow_review_request",
            "deliverable": "动作结构",
            "return_to": "pm",
            "metadata": {"send_mode": "real_send", "reviewer_role": "cj"},
        },
        profile,
    )["ok"] is True


def test_sanitize_filters_status_codes_when_no_safe_text_remains():
    text = sanitize_user_visible_message("waiting_for_dd\nrelay_enqueued", fallback=None)

    assert text is None


def test_workflow_reply_templates_support_required_event_types():
    required = {
        "dispatch_to_role",
        "role_result_returned",
        "next_handoff",
        "review_request",
        "review_rejected",
        "workflow_waiting",
        "workflow_completed",
    }

    for event_type in required:
        text = render_workflow_template(
            event_type,
            {
                "from_role_name": "总导演",
                "target_role_name": "武指",
                "task_goal": "拆动作",
                "focus_points": "- 节奏",
                "upstream_summary": "雨夜、旧债、克制爆发。",
                "reviewer_name": "场记",
                "deliver_to_name": "PM",
                "next_role_name": "武指",
                "review_flow": "这一步不用别人验收，做完后直接回 PM。",
            },
        )

        assert text
