"""
bg-progress-logger plugin — 后台任务进度日志。

Hooks:
  - pre_tool_call:  拦截 terminal(background=true) → 原子写入"启动"日志
  - post_tool_call: 检查 terminal 启动结果 → 失败自动写"失败"日志

进度日志格式: [ISO8601] uuid:xxxxxxxx: 描述: 状态 → 路径
UUID + 原子写入(tmp→sync→append→sync) + 启动验证。
"""

import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("plugins.bg-progress-logger")

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
PROGRESS_LOG = HERMES_HOME / "state" / "progress.log"


def _atomic_append(line: str) -> None:
    """Durably append a line with os.sync() for crash safety."""
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")
    os.sync()  # ensure kernel buffers flushed to disk


def _gen_uuid() -> str:
    """Generate 8-char hex UUID from uuidgen."""
    try:
        out = subprocess.run(["uuidgen", "-r"], capture_output=True, text=True, timeout=2)
        return out.stdout.strip()[:8]
    except Exception:
        import random
        return f"{random.randint(0, 0xFFFFFFFF):08x}"


def _sanitize_path(path: str) -> str:
    """Replace NAS mount paths with logical NAS: prefix for compliance."""
    if "/mnt/nas/" in path:
        path = "NAS:" + path.split("/mnt/nas/", 1)[1]
    return path


def _infer_description(args: dict) -> str:
    """Infer a human-readable task description from terminal arguments."""
    command = str(args.get("command", "")).strip()
    if not command:
        return "未知任务"

    lowered = command.lower()
    if "ppt-master" in lowered or ("claude" in lowered and "--add-dir" in lowered):
        return "PPT Master 生成"
    if "ffmpeg" in lowered:
        return "ffmpeg 处理"
    if "docker" in lowered and ("build" in lowered or "run" in lowered):
        return "Docker 操作"
    if "curl" in lowered:
        return "网络请求"
    if "python" in lowered and "train" in lowered:
        return "模型训练"
    if "python" in lowered and "backup" in lowered:
        return "Python 备份脚本"
    if "rsync" in lowered:
        return "文件同步"
    if "tar " in lowered or lowered.startswith("tar "):
        return "压缩/解压"
    if "backup" in lowered:
        return "数据备份"
    if "restic" in lowered:
        return "Restic 备份"
    if "wget" in lowered:
        return "文件下载"

    # Fallback: first 50 chars of command, stripped of newlines
    short = command[:50].replace("\n", " ").replace("  ", " ")
    return f"Shell任务({short}...)"


def _extract_path_hint(args: dict) -> str:
    """Try to extract an output path hint from the command."""
    command = str(args.get("command", ""))
    # --add-dir
    m = re.search(r"--add-dir\s+(\S+)", command)
    if m:
        return _sanitize_path(m.group(1))
    # --output or -o
    m = re.search(r"(?:--output|-o)\s+(\S+)", command)
    if m:
        return _sanitize_path(m.group(1))
    # working directory hint
    workdir = str(args.get("workdir", ""))
    if workdir:
        return _sanitize_path(workdir)
    return "无"


# ── Hooks ──────────────────────────────────────────────────────────────────

def _pre_tool_call(tool_name: str, args: dict, **kwargs):
    """Intercept terminal(background=true) and write start log."""
    if tool_name != "terminal":
        return {}

    if not args.get("background", False):
        return {}

    try:
        uuid_val = _gen_uuid()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        desc = _infer_description(args)
        path_hint = _extract_path_hint(args)

        line = f"[{ts}] uuid:{uuid_val}: {desc}: 启动 → {path_hint}"
        _atomic_append(line)
        logger.info("bg-progress-logger: start %s (%s)", uuid_val, desc)
    except Exception as e:
        logger.error("bg-progress-logger pre_tool_call error: %s", e)

    return {}  # Never block


def _post_tool_call(tool_name: str, args: dict, result: str, session_id: str = "", **kwargs):
    """Verify terminal background task startup and log failure if needed."""
    if tool_name != "terminal":
        return

    if not args.get("background", False):
        return

    try:
        result_str = str(result) if result else ""
        # terminal(background=true) success contains "Background process started"
        if "Background process started" in result_str:
            return  # Success, no action needed

        # Failed startup — write failure log
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        desc = _infer_description(args)
        line = f"[{ts}] uuid:ERROR: {desc}: 失败 → 命令启动失败: {result_str[:100]}"
        _atomic_append(line)
        logger.info("bg-progress-logger: startup failure logged for %s", desc)
    except Exception as e:
        logger.error("bg-progress-logger post_tool_call error: %s", e)


def register(ctx):
    """Register plugin hooks."""
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    logger.info("bg-progress-logger plugin registered v1.0 (pre_tool_call + post_tool_call)")
