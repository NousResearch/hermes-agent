"""Feishu low-budget routing and deterministic diagnostic shortcuts."""

from gateway.run import (
    _deep_command_message,
    _platform_budget_key_for_message,
    _platform_task_mode_for_message,
)
from gateway.known_ops_tasks import match_known_ops_task
from hermes_cli.commands import is_gateway_known_command, resolve_command


def test_feishu_plain_message_keeps_low_token_budget_key():
    assert _platform_budget_key_for_message("feishu", "继续上一轮") == "feishu"


def test_feishu_deep_prefix_selects_deep_budget_key():
    assert _platform_budget_key_for_message("feishu", "深诊断") == "feishu_deep"
    assert _platform_budget_key_for_message("feishu", "深诊断：继续查路由配置") == "feishu_deep"
    assert _platform_budget_key_for_message("feishu", "/deep 继续查路由配置") == "feishu_deep"


def test_deep_prefix_does_not_affect_other_platforms():
    assert _platform_budget_key_for_message("telegram", "/deep 继续查") == "telegram"


def test_feishu_github_install_request_uses_install_task_mode():
    message = "请帮我安装这个 : https://github.com/BigPizzaV3/CodexPlusPlus"
    budget_key = _platform_budget_key_for_message("feishu", message)

    assert budget_key == "feishu"
    assert _platform_task_mode_for_message("feishu", budget_key, message) == "install"


def test_feishu_deep_install_request_does_not_use_install_task_mode():
    message = "深诊断：请帮我安装这个 : https://github.com/BigPizzaV3/CodexPlusPlus"
    budget_key = _platform_budget_key_for_message("feishu", message)

    assert budget_key == "feishu_deep"
    assert _platform_task_mode_for_message("feishu", budget_key, message) == ""


def test_non_feishu_install_request_does_not_use_install_task_mode():
    message = "install https://github.com/BigPizzaV3/CodexPlusPlus"

    assert _platform_task_mode_for_message("telegram", "telegram", message) == ""


def test_deep_slash_command_is_registered_for_gateway():
    cmd = resolve_command("deep")

    assert cmd is not None
    assert cmd.name == "deep"
    assert is_gateway_known_command("deep")


def test_deep_slash_command_rewrites_to_text_trigger():
    assert _deep_command_message("") == "深诊断：继续上一轮问题"
    assert _deep_command_message("继续查路由") == "深诊断：继续查路由"
    assert _platform_budget_key_for_message("feishu", _deep_command_message("")) == "feishu_deep"


def test_today_token_usage_request_uses_fast_report_intent():
    task = match_known_ops_task(
        "feishu",
        "请查一下 今天截止到现在，输入输出Token的整体消耗情况，消耗Token最多的前三项任务是哪几个"
    )
    assert task is not None
    assert task.name == "token_usage_report"
    assert match_known_ops_task("feishu", "今日 token 用量统计") is not None


def test_unrelated_token_text_does_not_use_fast_report_intent():
    assert match_known_ops_task("feishu", "解释一下 token 是什么") is None
    assert match_known_ops_task("feishu", "今天帮我检查 OpenClaw 状态") is None
