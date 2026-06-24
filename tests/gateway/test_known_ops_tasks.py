"""Known ops task registry behavior."""

from gateway.known_ops_tasks import (
    known_ops_task_metadata,
    match_known_ops_task,
    render_known_ops_task,
)


def test_known_ops_task_metadata_exposes_promotion_contract():
    metadata = known_ops_task_metadata("feishu")

    cron_task = next(item for item in metadata if item["name"] == "cron_schedule_status")
    assert "verification" in cron_task
    assert "cron/jobs.json" in cron_task["promotion_hint"]

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


def test_cron_schedule_question_uses_fast_path(monkeypatch):
    monkeypatch.setattr(
        "cron.jobs.load_jobs",
        lambda: [
            {
                "id": "61fc0eed3cbe",
                "name": "模型限频监控",
                "schedule": {"kind": "interval", "minutes": 10, "display": "every 10m"},
                "schedule_display": "every 10m",
                "enabled": True,
                "state": "scheduled",
                "next_run_at": "2026-06-23T23:03:57+08:00",
                "last_run_at": "2026-06-23T22:53:57+08:00",
                "last_status": "ok",
            }
        ],
    )

    result = render_known_ops_task(
        "feishu",
        "请问 定时自动运行的 模型限频监控 目前的间隔是多长时间?",
    )

    assert result is not None
    assert result.task.name == "cron_schedule_status"
    assert "模型限频监控: every 10m" in result.text
    assert "下次运行: 2026-06-23T23:03:57+08:00" in result.text
    assert "最近运行: 2026-06-23T22:53:57+08:00 ok" in result.text


def test_cron_schedule_question_is_platform_scoped(monkeypatch):
    monkeypatch.setattr(
        "cron.jobs.load_jobs",
        lambda: [{"name": "模型限频监控", "schedule_display": "every 10m"}],
    )

    assert (
        render_known_ops_task(
            "telegram",
            "请问 定时自动运行的 模型限频监控 目前的间隔是多长时间?",
        )
        is None
    )


def test_unrelated_cron_text_does_not_use_fast_path(monkeypatch):
    monkeypatch.setattr(
        "cron.jobs.load_jobs",
        lambda: [{"name": "模型限频监控", "schedule_display": "every 10m"}],
    )

    assert render_known_ops_task("feishu", "请解释一下什么是 cron") is None
    assert render_known_ops_task("feishu", "今天帮我检查 OpenClaw 状态") is None


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
