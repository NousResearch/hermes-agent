"""
Audit — 不可变Audit log
=====================

append-only 日志，记录 VIP daemon 每个操作。
文件不可原地修改，只能追加。

格式:
  [timestamp] EVENT  key=value  key=value ...
"""

import logging
import os
import time

logger = logging.getLogger("vipd.audit")

AUDIT_LOG = "/var/log/hermes-vip/audit.log"


class AuditLogger:
    """Audit log写入器"""

    def __init__(self, path: str = AUDIT_LOG):
        self._path = path
        self._fd = None

    def open(self):
        """打开Audit log文件（append mode）"""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            # a+b: append-only, 二进制模式避免编码问题
            self._fd = open(self._path, "a+b")
            logger.info("audit log: %s", self._path)
        except PermissionError:
            logger.warning("无法写入Audit log %s（permission denied）", self._path)
            self._fd = None

    def close(self):
        if self._fd:
            self._fd.close()
            self._fd = None

    def log(self, event: str, **fields):
        """写入一条审计记录"""
        if not self._fd:
            return

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"[{now}]", event.upper()]
        for k, v in fields.items():
            if v is None:
                v = ""
            # 清理值中的换行和不可见字符
            sv = str(v).replace("\n", "\\n").replace("\r", "\\r")[:200]
            parts.append(f"{k}={sv}")

        line = "  ".join(parts) + "\n"
        try:
            self._fd.write(line.encode("utf-8"))
            self._fd.flush()
        except OSError as exc:
            logger.error("Audit log写入失败: %s", exc)

    # ── 常用事件快捷方法 ──

    def request(self, req_id: str, command: str, origin_channel: str):
        self.log("request", req_id=req_id, command=command[:80],
                 channel=origin_channel)

    def approve(self, req_id: str, connector: str, verified_by: str):
        self.log("approve", req_id=req_id, connector=connector,
                 verified_by=verified_by)

    def deny(self, req_id: str, connector: str, verified_by: str = ""):
        self.log("deny", req_id=req_id, connector=connector,
                 verified_by=verified_by or "")

    def timeout(self, req_id: str):
        self.log("timeout", req_id=req_id)

    def execute(self, req_id: str, exit_code: int, duration_ms: int,
                command: str):
        self.log("execute", req_id=req_id, exit_code=str(exit_code),
                 duration_ms=str(duration_ms), command=command[:60])

    def start(self):
        self.log("daemon_start", pid=str(os.getpid()))

    def stop(self):
        self.log("daemon_stop", pid=str(os.getpid()))

    def error(self, req_id: str, error: str):
        self.log("error", req_id=req_id or "-", error=error[:100])

    def kill_file(self, action: str):
        """kill 文件状态变更"""
        self.log("kill_file", action=action)


# 全局单例
audit = AuditLogger()
