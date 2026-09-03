"""
Approval Queue — Approval queue
=============================

核心职责：
1. Accept sudo requests, generate req_id, enqueue
2. 管理 TTL 超时（默认 5 分钟）
3. Accept approval responses, match req_id, return result
4. 线程安全（requests 可能并发提交）
"""

import json
import logging
import os
import secrets
import threading
import time
from typing import Optional

logger = logging.getLogger("vipd.approval_queue")


class ApprovalEntry:
    """A pending privilege request"""

    __slots__ = (
        "req_id", "command", "reason", "origin",
        "created_at", "expires_at", "result",
        "event", "connector",
    )

    def __init__(self, req_id: str, command: str, reason: str, origin: dict,
                 ttl: int = 300):
        now = time.time()
        self.req_id = req_id
        self.command = command
        self.reason = reason
        self.origin = origin
        self.created_at = now
        self.expires_at = now + ttl
        self.result: Optional[dict] = None
        self.event = threading.Event()
        self.connector: Optional[str] = None

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def resolved(self) -> bool:
        return self.result is not None

    def to_dict(self) -> dict:
        return {
            "req_id": self.req_id,
            "command": self.command,
            "reason": self.reason,
            "origin": self.origin,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "expired": self.expired,
            "resolved": self.resolved,
            "connector": self.connector,
        }


class ApprovalQueue:
    """Thread-safe approval queue"""

    def __init__(self, ttl: int = 300):
        self._ttl = ttl
        self._lock = threading.Lock()
        # req_id → ApprovalEntry
        self._pending: dict[str, ApprovalEntry] = {}
        # 持久化路径（可选）
        self._persist_path: Optional[str] = None

    # ── 请求生命周期 ──

    def submit(self, command: str, reason: str, origin: dict) -> ApprovalEntry:
        """
        Submit a privilege request。

        生成唯一的 req_id，创建 ApprovalEntry，入队列。
        返回 entry，调用者通过 entry.event.wait() 阻塞等待结果。
        """
        req_id = self._generate_req_id()
        entry = ApprovalEntry(req_id, command, reason, origin, self._ttl)

        with self._lock:
            self._pending[req_id] = entry

        logger.info("submit  req_id=%s command=%s", req_id, command)
        self._persist()
        return entry

    def resolve(self, req_id: str, action: str, connector: str,
                verified_by: str = "") -> bool:
        """
        批准或拒绝一条请求。

        Args:
            req_id: 请求 ID
            action: "approve" 或 "deny"
            connector: 连接器名称
            verified_by: 验证者标识

        Returns:
            True 如果找到并处理了该请求
        """
        with self._lock:
            entry = self._pending.get(req_id)
            if entry is None:
                logger.warning("resolve  req_id=%s not found", req_id)
                return False
            if entry.resolved:
                logger.warning("resolve  req_id=%s already resolved", req_id)
                return False

            entry.connector = connector
            entry.result = {
                "action": action,
                "connector": connector,
                "verified_by": verified_by,
                "resolved_at": time.time(),
            }
            del self._pending[req_id]

        entry.event.set()
        logger.info("resolve  req_id=%s action=%s connector=%s",
                     req_id, action, connector)
        self._persist()
        return True

    def set_result(self, req_id: str, result: dict) -> bool:
        """
        设置命令执行结果（Executor 调用）。

        result 格式: {stdout, stderr, exit_code, executed_at, duration_ms}
        """
        with self._lock:
            entry = self._pending.get(req_id)
            if entry is None:
                return False
        # result 是提交到队列时 Event 的消费者需要的数据
        # 这里使用一个外部存储来放执行结果
        entry.result.update({"exec_result": result})
        return True

    def get(self, req_id: str) -> Optional[ApprovalEntry]:
        """获取指定请求（未完成的）"""
        # 仍在 pending 中的请求
        with self._lock:
            entry = self._pending.get(req_id)
            if entry:
                return entry
        return None

    def list_pending(self) -> list[dict]:
        """列出所有待审批请求"""
        with self._lock:
            entries = [e.to_dict() for e in self._pending.values()
                      if not e.resolved and not e.expired]
        return sorted(entries, key=lambda e: e["created_at"])

    # ── 后台维护 ──

    def reap_expired(self) -> list[str]:
        """
        收割已过期的请求（标记为超时拒绝）。

        Returns: 收割的 req_id 列表
        """
        now = time.time()
        reaped = []
        with self._lock:
            expired_ids = [
                rid for rid, e in self._pending.items()
                if not e.resolved and now > e.expires_at
            ]
            for rid in expired_ids:
                entry = self._pending.pop(rid)
                entry.result = {
                    "action": "timeout",
                    "connector": "system",
                    "verified_by": "ttl",
                    "resolved_at": now,
                }
                entry.event.set()
                reaped.append(rid)

        if reaped:
            logger.info("reap  count=%d ids=%s", len(reaped), reaped)
            self._persist()
        return reaped

    def clear(self):
        """清空所有待审批请求（daemon 关闭时）"""
        with self._lock:
            for entry in self._pending.values():
                if not entry.resolved:
                    entry.result = {
                        "action": "deny",
                        "connector": "system",
                        "verified_by": "shutdown",
                        "resolved_at": time.time(),
                    }
                    entry.event.set()
            self._pending.clear()
        self._persist()
        logger.info("clear  all pending requests cancelled")

    # ── 持久化 ──

    PERSIST_FILE = "/var/run/hermes-vip/pending.json"

    def _persist(self):
        """将当前待审批请求持久化到磁盘（防 daemon 重启丢失）"""
        path = self._persist_path or self.PERSIST_FILE
        try:
            with self._lock:
                data = {
                    rid: {
                        "req_id": e.req_id,
                        "command": e.command,
                        "reason": e.reason,
                        "origin": e.origin,
                        "created_at": e.created_at,
                        "expires_at": e.expires_at,
                    }
                    for rid, e in self._pending.items()
                    if not e.resolved
                }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
        except PermissionError:
            pass  # 非 root 运行时跳过
        except Exception as exc:
            logger.debug("persist  error: %s", exc)

    def _load_persisted(self) -> dict:
        """从磁盘加载持久化的待审批请求"""
        path = self._persist_path or self.PERSIST_FILE
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def recover(self):
        """
        Daemon 启动时恢复上次未完成的请求。

        对已过期的直接标记超时，未过期的重新入队列等待。
        """
        data = self._load_persisted()
        if not data:
            return

        now = time.time()
        recovered = 0
        expired = 0
        with self._lock:
            for rid, d in data.items():
                if rid in self._pending:
                    continue
                entry = ApprovalEntry(
                    rid, d["command"], d.get("reason", ""),
                    d.get("origin", {}), self._ttl
                )
                entry.created_at = d["created_at"]
                entry.expires_at = d["expires_at"]

                if now > entry.expires_at:
                    entry.result = {
                        "action": "timeout",
                        "connector": "system",
                        "verified_by": "recovery_ttl",
                        "resolved_at": now,
                    }
                    entry.event.set()
                    expired += 1
                else:
                    self._pending[rid] = entry
                    recovered += 1

        logger.info("recover  recovered=%d expired=%d", recovered, expired)

    # ── 工具方法 ──

    @staticmethod
    def _generate_req_id() -> str:
        """生成唯一的请求 ID: 8 位 hex"""
        unique = secrets.token_hex(4)
        # 加上纳秒级时间戳确保同一毫秒内也不重复
        ns = time.time_ns() % 1000000
        return f"{unique}{ns:06x}"  # 12 位 hex
