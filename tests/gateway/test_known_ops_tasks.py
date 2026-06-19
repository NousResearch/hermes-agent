"""Known ops task registry behavior."""

from gateway.known_ops_tasks import (
    known_ops_task_metadata,
    match_known_ops_task,
    render_known_ops_task,
)


def test_known_ops_task_metadata_exposes_promotion_contract():
    metadata = known_ops_task_metadata("feishu")

    token_task = next(item for item in metadata if item["name"] == "token_usage_report")
    assert "verification" in token_task
    assert "promotion_hint" in token_task
    assert token_task["platforms"] == ["feishu"]

    loop_task = next(item for item in metadata if item["name"] == "agent_loop_diagnostic_report")
    assert "diagnostic-loop" in " ".join(loop_task["verification"])


def test_known_ops_task_is_platform_scoped():
    text = "今日 token 用量统计"

    assert match_known_ops_task("feishu", text) is not None
    assert match_known_ops_task("telegram", text) is None


def test_known_ops_task_render_uses_registered_handler(monkeypatch):
    captured = {}

    def fake_render_token_usage_report(**kwargs):
        captured.update(kwargs)
        return "fake token report"

    monkeypatch.setattr(
        "tools.local_repair_tool.render_token_usage_report",
        fake_render_token_usage_report,
    )

    result = render_known_ops_task("feishu", "查一下飞书今日 Token 消耗 Top 5")

    assert result is not None
    assert result.task.name == "token_usage_report"
    assert result.text == "fake token report"
    assert captured["scope"] == "feishu"
    assert captured["top_n"] == 5
    assert captured["label"] == "今日"


def test_known_ops_task_render_parses_yesterday(monkeypatch):
    captured = {}

    def fake_render_token_usage_report(**kwargs):
        captured.update(kwargs)
        return "fake token report"

    monkeypatch.setattr(
        "tools.local_repair_tool.render_token_usage_report",
        fake_render_token_usage_report,
    )

    result = render_known_ops_task("feishu", "请查一下昨天一整天的token消耗情况")

    assert result is not None
    assert result.task.name == "token_usage_report"
    assert result.text == "fake token report"
    assert "target_date" in captured
    assert captured["label"] == "昨日"


def test_known_ops_task_render_parses_rolling_days(monkeypatch):
    captured = {}

    def fake_render_token_usage_report(**kwargs):
        captured.update(kwargs)
        return "fake token report"

    monkeypatch.setattr(
        "tools.local_repair_tool.render_token_usage_report",
        fake_render_token_usage_report,
    )

    result = render_known_ops_task("feishu", "统计最近3天 Token 消耗 Top 8")

    assert result is not None
    assert result.task.name == "token_usage_report"
    assert result.text == "fake token report"
    assert captured["days"] == 3
    assert captured["top_n"] == 8
    assert captured["label"] == "最近 3 天"


def test_known_ops_task_render_parses_explicit_date_range(monkeypatch):
    captured = {}

    def fake_render_token_usage_report(**kwargs):
        captured.update(kwargs)
        return "fake token report"

    monkeypatch.setattr(
        "tools.local_repair_tool.render_token_usage_report",
        fake_render_token_usage_report,
    )

    result = render_known_ops_task("feishu", "统计 2026-06-10 到 2026-06-17 的 Token 用量")

    assert result is not None
    assert result.task.name == "token_usage_report"
    assert result.text == "fake token report"
    assert captured["range_start"] == "2026-06-10"
    assert captured["range_end"] == "2026-06-18"
    assert captured["label"] == "2026-06-10 至 2026-06-17"


def test_agent_loop_diagnostic_request_uses_bounded_report():
    result = render_known_ops_task(
        "feishu",
        "请分析为什么 Hermes 查 OpenClaw 故障一直卡住，无法答复，陷入死循环",
    )

    assert result is not None
    assert result.task.name == "agent_loop_diagnostic_report"
    assert "确定性报告" in result.text
    assert "避免再次进入通用 Agent 探索循环" in result.text
