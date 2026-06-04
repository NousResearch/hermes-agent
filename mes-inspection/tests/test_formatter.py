"""格式化器单元测试。"""

import pytest

from alerter.formatter import (
    determine_alert_level, format_feishu_card, format_text_message, format_memory_entry
)


class TestDetermineAlertLevel:
    def test_critical(self):
        assert determine_alert_level(2) == "P0"

    def test_warning(self):
        assert determine_alert_level(1) == "P1"

    def test_normal(self):
        assert determine_alert_level(0) == "P3"


class TestFormatFeishuCard:
    def test_basic_card(self):
        checks = [
            {"name": "进程存活", "status_code": 0, "value": 1, "message": "正常"},
            {"name": "5xx错误率", "status_code": 2, "value": 8.5, "threshold": 5.0, "message": "错误率过高"},
        ]
        card = format_feishu_card("nginx", 2, checks, "Nginx 异常")
        assert "header" in card
        assert "🚨" in card["header"]["title"]["content"]
        assert card["header"]["template"] == "red"

    def test_normal_card(self):
        checks = [{"name": "进程存活", "status_code": 0, "value": 1, "message": "正常"}]
        card = format_feishu_card("nginx", 0, checks, "一切正常", level="P3")
        assert "📋" in card["header"]["title"]["content"]


class TestFormatTextMessage:
    def test_basic_message(self):
        checks = [{"name": "test", "status_code": 1, "value": 85, "message": "告警"}]
        msg = format_text_message("jvm", 1, checks, "JVM 告警", level="P1")
        assert "⚠️" in msg
        assert "jvm" in msg.lower() or "JVM" in msg

    def test_with_ai_diagnosis(self):
        checks = []
        msg = format_text_message("nginx", 2, checks, "异常", ai_diagnosis="Nginx 进程挂了")
        assert "AI 诊断" in msg
        assert "Nginx 进程挂了" in msg


class TestFormatMemoryEntry:
    def test_basic_entry(self):
        checks = [{"name": "进程存活", "status_code": 2, "message": "进程不存在"}]
        entry = format_memory_entry("nginx", 2, "Nginx 进程挂了", checks, root_cause="OOM", fix_action="restart")
        assert "nginx" in entry
        assert "OOM" in entry
        assert "restart" in entry
