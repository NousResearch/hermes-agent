"""hermes-ssh 审计日志。

独立于 Hermes 主日志：JSON Lines → ~/.hermes/plugin-data/hermes-ssh/audit.log
RotatingFileHandler 滚动（默认 20MB×3），永不记录命令输出内容与密码。
每条记录至少含：ts / action / target / caller_session / command / result 摘要。
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_AUDIT_LOGGER_NAME = "hermes_ssh.audit"
_configured: bool = False
# 进程级调用方类型+profile: configure 时捕获一次(进程启动后不变)。
# 例: "gateway/xiaot" | "cli/xiaot" | "cron:<任务名>/xiaot"


def _compute_origin() -> str:
    """进程类型/profile 名。这些 env 是进程启动时设置的, 运行期不变。"""
    hermes_home = Path(os.path.expanduser(
        os.environ.get("HERMES_HOME", "~/.hermes")))
    profile = hermes_home.name if hermes_home.name != ".hermes" else "default"
    if os.environ.get("_HERMES_GATEWAY"):
        kind = "gateway"
    elif os.environ.get("HERMES_CRON_SESSION"):
        kind = f"cron:{os.environ['HERMES_CRON_SESSION']}"
    elif os.environ.get("TDAI_MEMORY_SYSTEM_USER_KEY"):
        kind = "external-mcp"          # 被外部记忆网关等进程调起
    else:
        kind = "cli"
    return f"{kind}/{profile}"


def _log_dir() -> Path:
    hermes_home = os.environ.get("HERMES_HOME", "~/.hermes")
    return Path(os.path.expanduser(f"{hermes_home}/plugin-data/hermes-ssh"))


def configure(max_mb: float = 20, backups: int = 3) -> str:
    """幂等配置 audit logger；返回日志文件路径。"""
    global _configured
    log_dir = _log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        _configured = False
        return ""

    handler = logging.handlers.RotatingFileHandler(
        log_dir / "audit.log",
        maxBytes=int(max_mb * 1024 * 1024),
        backupCount=backups,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    lg = logging.getLogger(_AUDIT_LOGGER_NAME)
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    # 防重复挂载（热重载场景）
    seen = set()
    for h in list(lg.handlers):
        if isinstance(h, logging.handlers.RotatingFileHandler):
            if id(h) not in seen:
                seen.add(id(h))
        else:
            lg.removeHandler(h)
    lg.propagate = False
    _configured = True
    global _origin
    _origin = _compute_origin()
    return str(log_dir / "audit.log")


def emit(action: str, *, target: str = "", caller_session: str = "",
         command: Optional[str] = None, exit_code: int = -999,
         duration_ms: int = 0, status: str = "ok",
         error_type: Optional[str] = None,
         extra: Optional[Dict[str, Any]] = None) -> None:
    """写一条审计记录。任何异常都吞掉 —— 审计绝不影响主流程。

    调用方溯源三件套:
      origin         进程类型/profile: gateway|cli|cron:<名> / <profile>
                     (configure 时快照)
      caller_session agent 会话 ID (dispatch kwargs)
      task_id        任务/子agent 标识 (dispatch kwargs)
    """
    if not _configured:
        return
    rec: Dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "action": action,
        "target": target,
        "origin": _origin,
        "caller_session": caller_session or "",
        "status": status,
        "duration_ms": round(duration_ms, 1),
    }
    tid = ""
    if isinstance(extra, dict):
        tid = str(extra.get("task_id") or "")
    if tid:
        rec["task_id"] = tid
    if command is not None:
        rec["command"] = command[:2000]
    if exit_code != -999:
        rec["exit_code"] = exit_code
    if error_type:
        rec["error_type"] = str(error_type)[:80]
    if extra:
        # 显式白名单字段，防意外泄密
        for k in ("auth", "stdout_len", "stderr_len", "error_type",
                  "env_var", "note"):
            if k in extra:
                rec[k] = extra[k]
    try:
        logging.getLogger(_AUDIT_LOGGER_NAME).info(
            json.dumps(rec, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass
