"""media-miss-reshoot: 媒体漏发自动重投插件 v1.3

双 hook 机械执法：
- post_tool_call: 检测 terminal(background=true) → 写启动日志 + PID map
  + 检测 process(wait/poll) → PID死亡 → 写完成日志
  + 检测 send-media-safe.sh 调用 → 写 pending
- transform_llm_output: 读 pending → assistant_response 不含 MEDIA: → 直发文件

v1.1 (2026-07-03): 修复两处致命 bug——post_llm_call→transform_llm_output + 签名不匹配
v1.2 (2026-07-03): 修复第三处致命 bug——register() 从未调用 ctx.register_hook()
v1.3 (2026-07-03): bg-progress-logger 完成配对——PID追踪 + process hook + cron兜底
"""

import asyncio
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

PENDING_DIR = "/tmp/media-miss-reshoot"


def _log(msg: str):
    """写入诊断日志"""
    try:
        log_path = os.path.join(PENDING_DIR, "plugin.log")
        os.makedirs(PENDING_DIR, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def _post_tool_call(
    tool_name: str = "",
    args: dict = None,
    result: str = "",
    session_id: str = "",
    **kwargs,
):
    """
    检测 send-media-safe.sh 调用，从 stdout 提取 media_path。
    
    触发条件：Agent 在 terminal 中执行了 send-media-safe.sh。
    动作：解析 result → 提取 media_path → 写 pending。
    """
    if not session_id:
        return

    _log(f"ENTER: tool={tool_name} session={session_id}")

    # ── bg-progress-logger: process() completion detection ──
    # When Agent polls/waits for a background process and it exits,
    # write a "完成" entry immediately (before the send-media-safe logic).
    if tool_name == "process":
        _process_completion_check(args, result)
        return  # process tool doesn't need send-media-safe logic

    if tool_name != "terminal":
        _log(f"SKIP: not terminal (is {tool_name})")
        return

    if args is None:
        args = {}

    # ── bg-progress-logger: track ALL terminal(background=true) calls ──
    if args.get("background", False):
        _log_bg_start(tool_name, args, result)

    command = str(args.get("command", ""))
    if "send-media-safe.sh" not in command:
        return

    # 排除 cat/read/echo 等读取操作——只有真正执行脚本时才处理
    if not re.search(r'(?:^|[\s;/|&])send-media-safe\.sh\b', command):
        _log(f"SKIP: cmd contains script name but not executing: {command[:100]}")
        return

    _log(f"PROCESSING: result type={type(result).__name__}")

    # 尝试从 result 提取 media_path
    media_path = None
    if result:
        try:
            # result 可能是 JSON 字符串（新 send-media-safe.sh 输出 JSON）
            data = json.loads(result)
            if isinstance(data, dict):
                media_path = data.get("media_path")
        except (json.JSONDecodeError, TypeError):
            pass

    # 备用：从 command 中直接提取路径
    if not media_path:
        m = re.search(r"send-media-safe\.sh\s+(/\S+)", command)
        if m:
            media_path = m.group(1)
            _log(f"Extracted path from command: {media_path}")

    if not media_path:
        _log("SKIP: no media_path found")
        return

    pending_file = os.path.join(PENDING_DIR, session_id)
    try:
        os.makedirs(PENDING_DIR, exist_ok=True)
        with open(pending_file, "w") as f:
            f.write(f"{media_path}|{time.time()}|0\n")
        _log(f"PENDING: {session_id} -> {media_path}")
    except Exception as e:
        _log(f"PENDING write FAILED: {e}")


# ── bg-progress-logger helpers ──────────────────────────────────────────────

PROGRESS_LOG = os.path.join(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    "state", "progress.log"
)


def _gen_uuid():
    import subprocess
    try:
        out = subprocess.run(["uuidgen", "-r"], capture_output=True, text=True, timeout=2)
        return out.stdout.strip()[:8]
    except Exception:
        import random
        return f"{random.randint(0, 0xFFFFFFFF):08x}"


def _process_completion_check(args: dict, result: str):
    """检测 process(action='wait'/'poll') 返回，发现进程结束 → 写完成日志"""
    try:
        if args is None:
            args = {}
        result_str = str(result) if result else ""

        # Check if result indicates process exit
        has_exit = "exit_code" in result_str or "exited" in result_str

        if not has_exit:
            return

        # Scan pid map for dead PIDs → write completion for each
        if os.path.exists(PROGRESS_PID_MAP):
            with open(PROGRESS_PID_MAP) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 4:
                        uuid_val, pid_val, ts_val, desc_val = parts[0], parts[1], parts[2], parts[3]
                        try:
                            os.kill(int(pid_val), 0)
                        except (OSError, ValueError):
                            _bg_task_completed(uuid_val, pid_val, desc_val)
                            _log(f"AUTO-DONE: {uuid_val} pid={pid_val} via process() hook")
    except Exception as e:
        _log(f"PROCESS-CHECK ERROR: {e}")


PROGRESS_PID_MAP = os.path.join(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    "state", "progress_pid.map"
)


def _get_uuid_for_pid(pid: str) -> str:
    """从 pid map 中查找 uuid"""
    if not os.path.exists(PROGRESS_PID_MAP):
        return ""
    try:
        with open(PROGRESS_PID_MAP) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 2 and parts[1] == str(pid):
                    return parts[0]
    except Exception:
        pass
    return ""


def _bg_task_completed(uuid_val: str, pid: str = "", desc: str = ""):
    """Write a completion log line and clean up the pid map entry."""
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] uuid:{uuid_val}: {desc}: 完成"
        os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)
        with open(PROGRESS_LOG, "a") as f:
            f.write(line + "\n")
        os.sync()

        # Clean up pid map
        if os.path.exists(PROGRESS_PID_MAP) and uuid_val:
            try:
                with open(PROGRESS_PID_MAP) as f:
                    lines = f.readlines()
                with open(PROGRESS_PID_MAP, "w") as f:
                    for line in lines:
                        if not line.startswith(uuid_val + "|"):
                            f.write(line)
            except Exception:
                pass

        _log(f"BG-DONE: {uuid_val} {desc}")
    except Exception as e:
        _log(f"BG-DONE ERROR: {e}")


def _log_bg_start(tool_name: str, args: dict, result: str):
    """Write a start log line for a terminal(background=true) call."""
    try:
        from datetime import datetime
        command = str(args.get("command", ""))
        background = args.get("background", False)
        if not background:
            return

        uuid_val = _gen_uuid()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract PID from result
        pid = ""
        if result:
            m = re.search(r"pid.*?(\d{4,})", str(result))
            if m:
                pid = m.group(1)

        # Infer description
        lowered = command.lower()
        if "ppt-master" in lowered or ("claude" in lowered and "--add-dir" in lowered):
            desc = "PPT Master 生成"
        elif "ffmpeg" in lowered:
            desc = "ffmpeg 处理"
        elif "docker" in lowered:
            desc = "Docker 操作"
        elif "curl" in lowered:
            desc = "网络请求"
        elif "rsync" in lowered:
            desc = "文件同步"
        elif "backup" in lowered or "restic" in lowered:
            desc = "数据备份"
        else:
            short = command[:50].replace("\n", " ").replace("  ", " ")
            desc = f"Shell任务({short}...)"

        # Path hint
        import re as re2
        path_hint = "无"
        m = re2.search(r"--add-dir\s+(\S+)", command)
        if m:
            p = m.group(1)
            if "/mnt/nas/" in p:
                p = "NAS:" + p.split("/mnt/nas/", 1)[1]
            path_hint = p

        line = f"[{ts}] uuid:{uuid_val}: {desc}: 启动 → {path_hint}"
        os.makedirs(os.path.dirname(PROGRESS_LOG), exist_ok=True)
        with open(PROGRESS_LOG, "a") as f:
            f.write(line + "\n")
        os.sync()

        # Save PID mapping for completion tracking
        if pid:
            os.makedirs(os.path.dirname(PROGRESS_PID_MAP), exist_ok=True)
            with open(PROGRESS_PID_MAP, "a") as f:
                f.write(f"{uuid_val}|{pid}|{ts}|{desc}\n")
            os.sync()

        _log(f"BG-START: {uuid_val} pid={pid} {desc}")
    except Exception as e:
        _log(f"BG-START ERROR: {e}")


def _transform_llm_output(*args, **kwargs):
    """
    检测漏写 MEDIA: 标签，补发文件。
    
    触发条件：存在 pending 文件 且 response_text 不含 MEDIA: 标签。
    动作：send_weixin_direct(media_files=...) 直发文件 → 清 pending。
    """
    _log(f"TRANSFORM_CALLED: args={args[:2] if args else 'empty'} kwargs_keys={list(kwargs.keys())}")
    response_text = kwargs.get("response_text", "") or (args[0] if args else "")
    session_id = kwargs.get("session_id", "") or (args[1] if len(args) > 1 else "")
    platform = kwargs.get("platform", "")
    
    if not session_id:
        _log("TRANSFORM: no session_id, returning None")
        return None

    pending_file = os.path.join(PENDING_DIR, session_id)
    if not os.path.exists(pending_file):
        return None

    _log(f"CHECK: session={session_id} has pending, response_len={len(response_text or '')}")

    # 1. 读 pending
    try:
        with open(pending_file) as f:
            line = f.read().strip()
        if not line:
            os.remove(pending_file)
            _log("SKIP: empty pending file")
            return None
        parts = line.split("|")
        media_path = parts[0]
    except Exception as e:
        _log(f"PENDING read FAILED: {e}")
        return None

    # 2. 检查 MEDIA: 标签 — Agent 正常写了就跳过
    if "MEDIA:/" in (response_text or ""):
        _log(f"SKIP: MEDIA tag found in response")
        os.remove(pending_file)
        return None

    # 3. 漏写 — 获取 chat_id 并补发
    _log(f"RESHOOT NEEDED: {media_path}")
    try:
        from gateway.session_context import get_session_env
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
    except Exception as e:
        _log(f"chat_id FAILED: {e}")
        return None

    if not chat_id:
        _log("SKIP: no chat_id")
        return None

    # 4. 重发文件
    try:
        from gateway.platforms.weixin import send_weixin_direct

        _log(f"RESHOOT: sending {media_path} to {chat_id}")
        result = asyncio.run(asyncio.wait_for(
            send_weixin_direct(
                extra={},
                token=None,
                chat_id=chat_id,
                message="",
                media_files=[(media_path, False)],
            ),
            timeout=10.0,
        ))

        if result.get("success"):
            _log(f"RESHOT OK: {media_path}")
            os.remove(pending_file)
        else:
            _log(f"RESHOT FAIL: {result}")
    except Exception as e:
        _log(f"RESHOT ERROR: {e}")

    return None


def register(ctx):
    """插件注册——v1.3 修复：实际调用 ctx.register_hook()"""
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    _log("REGISTERED media-miss-reshoot v1.3 (post_tool_call + transform_llm_output)")
    logger.info("media-miss-reshoot: registered v1.3")
