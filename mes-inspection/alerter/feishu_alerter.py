"""飞书告警推送器。"""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from alerter.formatter import (
    format_feishu_card,
    format_text_message,
    determine_alert_level,
)


class FeishuAlerter:
    """飞书告警推送。

    支持两种方式：
    1. Hermes send_message 工具（推荐，需要 gateway 运行）
    2. Webhook URL（备用，无需 gateway）
    """

    def __init__(self, config: Dict[str, Any]):
        alerter_config = config.get("alerter", {})
        self.webhook_url = alerter_config.get("feishu_webhook_url", "")
        self.home_channel = os.getenv(
            alerter_config.get("feishu_home_channel_env", "FEISHU_HOME_CHANNEL"), ""
        )
        self.silence_window = alerter_config.get("silence_window_seconds", 900)
        self.batch_interval = alerter_config.get("batch_interval_seconds", 1800)
        self.max_message_length = alerter_config.get("max_message_length", 8000)
        self._silence_cache: Dict[str, float] = {}

    def should_alert(self, component: str, level: str) -> bool:
        """判断是否应该发送告警（防风暴）。"""
        if level in ("P0", "P1"):
            cache_key = f"{component}:{level}"
            last_alert = self._silence_cache.get(cache_key, 0)
            if time.time() - last_alert < self.silence_window:
                return False
            self._silence_cache[cache_key] = time.time()
        return True

    def send_alert(
        self,
        component: str,
        status_code: int,
        checks: List[Dict[str, Any]],
        summary: str,
        level: Optional[str] = None,
        ai_diagnosis: Optional[str] = None,
    ) -> bool:
        """发送告警到飞书。

        优先使用 Webhook，如果未配置则尝试 Hermes send_message。
        返回是否发送成功。
        """
        if level is None:
            level = determine_alert_level(status_code)

        # P3 不推送
        if level == "P3":
            return True

        if not self.should_alert(component, level):
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 尝试 Webhook
        if self.webhook_url:
            return self._send_via_webhook(
                component, status_code, checks, summary, level, timestamp, ai_diagnosis
            )

        # 尝试 Hermes send_message
        if self.home_channel:
            return self._send_via_hermes(
                component, status_code, checks, summary, level, timestamp, ai_diagnosis
            )

        # 无推送渠道，输出到 stderr
        text = format_text_message(
            component, status_code, checks, summary, level, timestamp, ai_diagnosis
        )
        import sys
        print(f"[ALERTE] 无飞书推送渠道，输出到 stderr:\n{text}", file=sys.stderr)
        return False

    def _send_via_webhook(
        self, component, status_code, checks, summary, level, timestamp, ai_diagnosis
    ) -> bool:
        """通过 Webhook 发送。"""
        card = format_feishu_card(
            component, status_code, checks, summary, level, timestamp, ai_diagnosis
        )
        payload = {"msg_type": "interactive", "card": card}
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result.get("code", -1) == 0 or result.get("StatusCode", -1) == 0
        except Exception as e:
            print(f"[ALERTE] Webhook 发送失败: {e}", file=__import__("sys").stderr)
            return False

    def _send_via_hermes(
        self, component, status_code, checks, summary, level, timestamp, ai_diagnosis
    ) -> bool:
        """通过 Hermes send_message 工具发送（写入文件供 cron 任务读取）。"""
        text = format_text_message(
            component, status_code, checks, summary, level, timestamp, ai_diagnosis
        )
        # 输出到 stdout，由 Hermes cron 任务的 agent 读取并发送
        print(f"[FEISHU_TARGET:{self.home_channel}]")
        print(text)
        return True
