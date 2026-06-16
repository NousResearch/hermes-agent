"""Known ops task registry behavior."""

from gateway.known_ops_tasks import (
    known_ops_task_metadata,
    match_known_ops_task,
    render_known_ops_task,
)


def test_known_ops_task_metadata_exposes_promotion_contract():
    metadata = known_ops_task_metadata("feishu")

    token_task = next(item for item in metadata if item["name"] == "today_token_usage_report")
    assert "verification" in token_task
    assert "promotion_hint" in token_task
    assert token_task["platforms"] == ["feishu"]


def test_known_ops_task_is_platform_scoped():
    text = "今日 token 用量统计"

    assert match_known_ops_task("feishu", text) is not None
    assert match_known_ops_task("telegram", text) is None


def test_known_ops_task_render_uses_registered_handler(monkeypatch):
    captured = {}

    def fake_render_today_token_usage_report(*, scope, top_n):
        captured["scope"] = scope
        captured["top_n"] = top_n
        return "fake token report"

    monkeypatch.setattr(
        "tools.local_repair_tool.render_today_token_usage_report",
        fake_render_today_token_usage_report,
    )

    result = render_known_ops_task("feishu", "查一下飞书今日 Token 消耗 Top 5")

    assert result is not None
    assert result.task.name == "today_token_usage_report"
    assert result.text == "fake token report"
    assert captured == {"scope": "feishu", "top_n": 5}
