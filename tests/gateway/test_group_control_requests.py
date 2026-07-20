from gateway.config import Platform
from gateway.session import SessionSource

from gateway.group_control_requests import match_group_control_request


def _make_source(
    *,
    platform: Platform = Platform.QQ_NAPCAT,
    chat_type: str = "dm",
    user_id: str = "179033731",
    chat_id: str = "179033731",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_match_group_control_request_returns_collect_only_and_report_targets_for_admin():
    source = _make_source(chat_type="group", chat_id="726109087")

    tool_args, error = match_group_control_request(
        source=source,
        body="这个群只监听，不要走大模型，每天给我日报。",
        target="group:726109087",
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
    )

    assert error is None
    assert tool_args == {
        "target": "group:726109087",
        "action": "enable_collect_only",
        "daily_report_enabled": True,
        "daily_report_target": "current_user_dm",
        "manual_report_target": "current_user_dm",
    }


def test_match_group_control_request_enables_report_when_collect_only_specifies_delivery_target():
    source = _make_source(chat_type="group", chat_id="726109087")

    tool_args, error = match_group_control_request(
        source=source,
        body="这个群切到监听采集，日报发我私聊。",
        target="group:726109087",
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
    )

    assert error is None
    assert tool_args == {
        "target": "group:726109087",
        "action": "enable_collect_only",
        "daily_report_enabled": True,
        "daily_report_target": "current_user_dm",
        "manual_report_target": "current_user_dm",
    }


def test_match_group_control_request_returns_deliver_report_for_report_now():
    source = _make_source(platform=Platform.WEIXIN, chat_id="project@chatroom")

    tool_args, error = match_group_control_request(
        source=source,
        body="这个群立即汇报，在这个群发。",
        target="project@chatroom",
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_chat",
    )

    assert error is None
    assert tool_args == {
        "action": "deliver_report",
        "target": "project@chatroom",
        "delivery_target": "current_chat",
    }


def test_match_group_control_request_rejects_non_admin_for_matching_control_turn():
    source = _make_source(chat_type="group", chat_id="726109087", user_id="555")

    tool_args, error = match_group_control_request(
        source=source,
        body="这个群只监听，不要走大模型。",
        target="group:726109087",
        admin_ids_configured=True,
        is_admin_user=False,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
    )

    assert tool_args is None
    assert error == "admin only"


def test_match_group_control_request_returns_missing_target_error_after_admin_check():
    source = _make_source(chat_type="dm")

    tool_args, error = match_group_control_request(
        source=source,
        body="把这个群切成只监听采集。",
        target=None,
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
    )

    assert tool_args is None
    assert error == "missing target"


def test_match_group_control_request_does_not_treat_daily_report_query_as_enable_mutation():
    source = _make_source(chat_type="group", chat_id="726109087")

    tool_args, error = match_group_control_request(
        source=source,
        body="这个群日报发哪？",
        target="group:726109087",
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
    )

    assert tool_args is None
    assert error is None


def test_match_group_control_request_does_not_enable_daily_report_from_query_tail():
    source = _make_source(chat_type="group", chat_id="726109087")

    tool_args, error = match_group_control_request(
        source=source,
        body="这个群恢复聊天，日报发哪？",
        target="group:726109087",
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
    )

    assert error is None
    assert tool_args == {
        "target": "group:726109087",
        "action": "resume_chat",
    }


def test_match_group_control_request_does_not_mutate_on_chat_capability_question():
    source = _make_source(chat_type="group", chat_id="726109087")

    tool_args, error = match_group_control_request(
        source=source,
        body="这个群可以聊天吗？",
        target="group:726109087",
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
    )

    assert tool_args is None
    assert error is None
