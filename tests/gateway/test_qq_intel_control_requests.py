from gateway.config import Platform
from gateway.session import SessionSource

from gateway.qq_intel_control_requests import (
    extract_qq_worker_name,
    match_qq_intel_control_request,
)


def _make_source(
    *,
    chat_type: str = "dm",
    user_id: str = "179033731",
    chat_id: str = "179033731",
) -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_match_qq_intel_control_request_returns_hire_worker_args():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="招一个情报员钢镚，去 726109087 这个群刺探情报，每天私聊向我汇报。",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=lambda body: "钢镚",
        looks_like_worker_context=lambda body: True,
        known_worker_names=[],
        target_extractor=lambda current_source, body: "group:726109087",
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: "刺探情报",
    )

    assert error is None
    assert tool_args == {
        "action": "hire_worker",
        "worker_name": "钢镚",
        "target_group": "group:726109087",
        "daily_report_target": "current_user_dm",
        "manual_report_target": "current_user_dm",
        "objective": "刺探情报",
    }


def test_match_qq_intel_control_request_can_resume_known_worker_without_explicit_prefix():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="让钢镚恢复任务，继续监听。",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=lambda body: "钢镚",
        looks_like_worker_context=lambda body: True,
        known_worker_names=["钢镚"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert error is None
    assert tool_args == {
        "action": "resume_worker",
        "worker_name": "钢镚",
    }


def test_extract_qq_worker_name_does_not_consume_timing_prefix():
    assert extract_qq_worker_name("让张三现在汇报") == "张三"


def test_match_qq_intel_control_request_can_report_known_worker_with_timing_prefix():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="让张三现在汇报",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=extract_qq_worker_name,
        looks_like_worker_context=lambda body: True,
        known_worker_names=["张三"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert error is None
    assert tool_args == {
        "action": "run_report_now",
        "worker_name": "张三",
        "manual_report_target": "current_user_dm",
    }


def test_extract_qq_worker_name_supports_explicit_worker_status_query():
    assert extract_qq_worker_name("员工钢镚还在吗") == "钢镚"


def test_match_qq_intel_control_request_can_query_explicit_worker_with_hai_zaima():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="员工钢镚还在吗",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=extract_qq_worker_name,
        looks_like_worker_context=lambda body: True,
        known_worker_names=["钢镚"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert error is None
    assert tool_args == {
        "action": "get_worker",
        "worker_name": "钢镚",
    }


def test_match_qq_intel_control_request_does_not_treat_bare_status_phrase_as_worker_name():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="那个情报员还在吗",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=extract_qq_worker_name,
        looks_like_worker_context=lambda body: True,
        known_worker_names=["钢镚"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert tool_args is None
    assert error is None


def test_match_qq_intel_control_request_does_not_treat_bot_alias_as_known_worker_without_explicit_prefix():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="让马哥现在汇报",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=extract_qq_worker_name,
        looks_like_worker_context=lambda body: True,
        known_worker_names=["马哥"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert tool_args is None
    assert error is None


def test_match_qq_intel_control_request_allows_explicit_worker_marker_for_bot_alias():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="让情报员马哥现在汇报",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=extract_qq_worker_name,
        looks_like_worker_context=lambda body: True,
        known_worker_names=["马哥"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert error is None
    assert tool_args == {
        "action": "run_report_now",
        "worker_name": "马哥",
        "manual_report_target": "current_user_dm",
    }


def test_match_qq_intel_control_request_does_not_treat_verbose_report_task_as_run_report_now():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="让钢镚现在汇报一下这个页面为什么回退了",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=extract_qq_worker_name,
        looks_like_worker_context=lambda body: True,
        known_worker_names=["钢镚"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert tool_args is None
    assert error is None


def test_match_qq_intel_control_request_does_not_treat_verbose_resume_task_as_worker_control():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="让钢镚继续监听线上部署日志，查清楚后向我汇报。",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=extract_qq_worker_name,
        looks_like_worker_context=lambda body: True,
        known_worker_names=["钢镚"],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: None,
    )

    assert tool_args is None
    assert error is None


def test_match_qq_intel_control_request_requires_target_for_hire():
    source = _make_source()

    tool_args, error = match_qq_intel_control_request(
        source=source,
        body="招一个情报员钢镚，去这个群刺探情报。",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        extract_worker_name=lambda body: "钢镚",
        looks_like_worker_context=lambda body: True,
        known_worker_names=[],
        target_extractor=lambda current_source, body: None,
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        hire_objective_extractor=lambda body, worker_name, target_group: "刺探情报",
    )

    assert tool_args is None
    assert error == "要安排情报员，请直接说清群号，或者在目标群里明确说“这个群”。"
