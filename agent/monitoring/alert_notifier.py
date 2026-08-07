"""Alert notifier: subscribe to monitoring emitter, push alerts to Feishu.

Gateway 已有 monitoring 事件 + emitter (fire-and-forget 队列 + 后台 dispatch)。
本模块订阅 emitter 的 batch, 过滤告警事件 (MCP 熔断器状态转换 / 健康降级),
推送到飞书 incoming webhook bot。

设计约束:
  - fail-silent: 任何告警失败 (网络/webhook 错) 只 debug 日志, 绝不影响 gateway
  - 速率限制: 同主题 5 分钟最多 1 条, 防 half-open 反复刷屏
  - webhook URL 未配时: 静默 no-op, 不报错不刷日志
  - 同步 urllib POST 放后台线程: emitter 的 _dispatch 是同步 fan-out,
    不能阻塞 dispatch 主循环, 所以 _on_event 只做轻量过滤 + 提交线程池
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ALERT_RATE_LIMIT_SEC = 300.0  # 同主题 5 分钟去重
_WEBHOOK_TIMEOUT_SEC = 5.0
_PARKED_RETRY_INTERVAL_HINT = "默认间隔"  # mcp_tool 的 _PARKED_RETRY_INTERVAL (300s), 此处只做告警文案, 不强耦合导入

# 进程级状态 (单 emitter 单 notifier, 无需锁: _on_event 在 emitter dispatch 线程串行调用)
_last_sent: Dict[str, float] = {}  # dedup_key -> monotonic deadline
_last_sent_ts: float = 0.0  # 用 time.monotonic 去重
_executor: Optional[ThreadPoolExecutor] = None
_webhook_url: str = ""


def _now() -> float:
    import time
    return time.monotonic()


def _post_feishu(webhook_url: str, title: str, body: str) -> None:
    """同步 POST 飞书 incoming webhook。fail-silent。"""
    try:
        payload = json.dumps(
            {"msg_type": "text", "content": {"text": f"{title}\n{body}"}},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            webhook_url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=_WEBHOOK_TIMEOUT_SEC).read()
    except Exception as exc:
        logger.debug("feishu alert post failed: %s", exc, exc_info=True)


def _dedup_send(dedup_key: str, title: str, body: str) -> None:
    """带速率限制的发送: 同 dedup_key 5 分钟内最多 1 条。"""
    global _last_sent_ts
    now = _now()
    if now - _last_sent.get(dedup_key, 0.0) < _ALERT_RATE_LIMIT_SEC:
        return
    _last_sent[dedup_key] = now
    # 提交线程池, 不阻塞 emitter dispatch
    if _executor is not None and _webhook_url:
        _executor.submit(_post_feishu, _webhook_url, title, body)


def _format_breaker_alert(ev: Dict[str, Any]) -> Optional[tuple]:
    """从 GatewayDiagnosticEvent dict 提取熔断器转换告警。返回 (dedup_key, title, body) 或 None。"""
    if ev.get("event") != "gateway_diagnostic":
        return None
    name = ev.get("name", "")
    # MCP 熔断器转换事件 name 约定: mcp_breaker_transition (由块B的_bump/_reset发出)
    if "mcp_breaker" not in name and "breaker" not in name:
        return None
    old_state = ev.get("old_state") or "?"
    new_state = ev.get("new_state") or "?"
    subsystem = ev.get("subsystem", "mcp")
    severity = ev.get("severity", "warning")
    if new_state != "open":
        # 只在进入 open 时告警 (恢复 closed 可选, 默认不刷屏)
        return None
    server = ev.get("error_code") or ev.get("platform") or subsystem
    icon = "🔴" if severity == "error" else "🟠"
    dedup = f"mcp_breaker_open:{server}"
    title = f"{icon} [hermes] MCP 熔断器 open: {server}"
    body = f"{server} {old_state}→{new_state} (subsystem={subsystem})\n连续失败达阈值, 工具调用将短路"
    return dedup, title, body


def _format_liveness_alert(ev: Dict[str, Any]) -> Optional[tuple]:
    """MCP 会话存活状态机告警 (mcp_liveness_transition)。

    保守规则: 只在某 server 进入 ``parked`` (彻底停摆) 时告警一次;
    ``connected → degraded`` (太频繁) 与 ``parked → connected`` (好消息)
    默认不告警 — 两者仍进 OTLP trace, 只是不打扰飞书。
    与 _format_breaker_alert 的 ``new_state != "open"`` 闸同构:
    在这里 ``new_state != "parked"`` 即 return None。

    去重键按 server, 配合 _dedup_send 的 5min 窗口, 一个 flapping
    server (parked → self-probe → parked) 在窗口内只告一次。
    """
    if ev.get("event") != "gateway_diagnostic":
        return None
    if ev.get("name") != "mcp_liveness_transition":
        return None
    new_state = ev.get("new_state") or ""
    if new_state != "parked":
        return None  # 只告彻底停摆; degraded/connected 不刷屏
    old_state = ev.get("old_state") or "?"
    server = ev.get("error_code") or ev.get("platform") or "mcp"
    reason = ev.get("source_logger") or "unknown"
    severity = ev.get("severity", "warning")
    icon = "🔴" if severity == "error" else "🟠"
    dedup = f"mcp_liveness_parked:{server}"
    title = f"{icon} [hermes] MCP server parked: {server}"
    body = (
        f"{server} {old_state}→{new_state} (subsystem=mcp)\n"
        f"reason={reason}\n"
        f"会话停摆, 将每 {_PARKED_RETRY_INTERVAL_HINT} 自探测一次至恢复"
    )
    return dedup, title, body


def _format_health_alert(ev: Dict[str, Any]) -> Optional[tuple]:
    """gateway 健康降级告警 (fatal platform / gateway degraded)。"""
    if ev.get("event") != "gateway_health":
        return None
    new_state = ev.get("new_state") or ""
    if new_state not in ("degraded", "fatal", "error", "failed"):
        return None
    old_state = ev.get("old_state") or "?"
    fatal = ev.get("fatal_platform_count", 0)
    if fatal == 0 and new_state != "fatal":
        return None  # 非致命降级默认不告警, 避免噪音
    dedup = f"gateway_health:{new_state}"
    title = f"🔴 [hermes] gateway 状态降级: {new_state}"
    body = f"gateway {old_state}→{new_state}\nfatal_platforms={fatal}"
    return dedup, title, body


def _on_event(batch: list) -> None:
    """emitter batch 回调。轻量过滤 + 提交线程池。"""
    if not _webhook_url:
        return
    for ev in batch:
        if not isinstance(ev, dict):
            continue
        for formatter in (
            _format_breaker_alert,
            _format_liveness_alert,
            _format_health_alert,
        ):
            result = formatter(ev)
            if result:
                dedup, title, body = result
                _dedup_send(dedup, title, body)
                break


def start_alert_notifier(config: Dict[str, Any]) -> None:
    """gateway 启动时调用: 解析 webhook URL, 订阅 emitter。
    config 来自 config.yaml 的 monitoring.alert 段。"""
    global _executor, _webhook_url
    # webhook URL: config.yaml monitoring.alert.feishu_webhook_url 或 env FEISHU_WEBHOOK_URL
    mon = config.get("monitoring") or {}
    alert = (mon.get("alert") or {}) if isinstance(mon, dict) else {}
    _webhook_url = (
        alert.get("feishu_webhook_url")
        or os.environ.get("FEISHU_WEBHOOK_URL")
        or ""
    )
    if not _webhook_url:
        logger.info("alert notifier: no feishu_webhook_url configured, alerts disabled")
        return
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hermes-alert")
    # 订阅 emitter (本地 import 避免循环依赖)
    from agent.monitoring.emitter import get_emitter
    get_emitter().subscribe(_on_event)
    logger.info("alert notifier: subscribed to monitoring emitter, webhook configured")


def stop_alert_notifier() -> None:
    """gateway 关闭时调用。"""
    global _executor
    if _executor is not None:
        try:
            from agent.monitoring.emitter import get_emitter
            get_emitter().unsubscribe(_on_event)
        except Exception:
            pass
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
