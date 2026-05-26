from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command
from hermes_cli.weiqi_modes import (
    OPUS_STATUS,
    classify_weiqi_auto_mode,
    is_auto_request,
    is_status_request,
    list_weiqi_modes,
    resolve_weiqi_mode,
)


def test_chinese_mode_commands_resolve_to_mode_command():
    for command in ["默认", "论文", "研究", "mba", "顾问", "播客", "创意", "整理", "中文润色", "润色", "省钱", "状态", "自动"]:
        resolved = resolve_command(command)
        assert resolved is not None
        assert resolved.name == "mode"
        assert command in GATEWAY_KNOWN_COMMANDS


def test_weiqi_mode_presets():
    expected = {
        "默认": ("default", "openai-codex", "gpt-5.5", "medium"),
        "论文": ("research", "openai-codex", "gpt-5.5", "high"),
        "研究": ("research", "openai-codex", "gpt-5.5", "high"),
        "MBA": ("business", "openai-codex", "gpt-5.5", "high"),
        "顾问": ("advisor", "copilot-acp", "opus", "high"),
        "播客": ("creative", "copilot-acp", "opus", "high"),
        "创意": ("creative", "copilot-acp", "opus", "high"),
        "整理": ("notes", "openai-codex", "gpt-5.5", "high"),
        "中文润色": ("polish", "openrouter", "moonshotai/kimi-k2.6", "medium"),
        "省钱": ("cheap", "openrouter", "qwen/qwen3.6-plus", "low"),
    }
    for command, values in expected.items():
        preset = resolve_weiqi_mode(command, "")
        assert preset is not None
        assert (preset.key, preset.provider, preset.model, preset.reasoning) == values


def test_mode_arg_resolution_and_status():
    assert resolve_weiqi_mode("mode", "论文").key == "research"
    assert resolve_weiqi_mode("模式", "省钱").key == "cheap"
    assert is_status_request("状态", "")
    assert is_status_request("mode", "状态")
    assert is_auto_request("自动", "")
    assert is_auto_request("mode", "smart")
    assert OPUS_STATUS == "OPUS_ENABLED_CLAUDE_CODE_MAX_CHAT_ONLY"
    assert len(list_weiqi_modes()) == 8


def test_weiqi_auto_routing_classifier():
    examples = {
        "帮我润色这段中文，让它更自然高级": "polish",
        "帮我想一期播客脚本和内容方向": "creative",
        "从顾问角度挑战一下我的长期规划盲点": "advisor",
        "MBA 案例里这个商业模式怎么分析": "business",
        "我的论文开题和研究假设怎么写": "research",
        "把这段会议纪要整理成行动项": "notes",
        "先省钱给我一个快速草稿": "cheap",
        "今天晚上吃什么": "default",
    }
    for text, expected in examples.items():
        assert classify_weiqi_auto_mode(text).key == expected
