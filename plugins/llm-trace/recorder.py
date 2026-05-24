"""LLM Trace Recorder — 本地文件埋点。

负责把每一次 OpenAI 兼容 chat.completions 调用（请求 / 响应 /
streaming chunk / 错误）写入 ``$HERMES_HOME/llm-traces/`` 下，按
``session/<id>/turn-<n>`` 组织。

线程安全：使用模块级单例 + ``threading.Lock``。本模块只做 IO，不
关心调用方是主对话还是辅助调用——上游通过 ``call_kind`` 字段区分。

启用方式：
    HERMES_LLM_TRACE=1                # 默认目录 $HERMES_HOME/llm-traces
    HERMES_LLM_TRACE_DIR=/abs/path    # 自定义目录
    HERMES_LLM_TRACE=0                # 显式关闭（即使插件启用也跳过）
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------- 序列化辅助 ----------------------------------------------------

def _safe_json_default(obj: Any) -> Any:
    """兜底 JSON 序列化，避免 pydantic / dataclass / set 等对象抛错。"""
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    return repr(obj)


def _to_jsonable(obj: Any) -> Any:
    """把任意对象转成可被 ``json.dumps`` 的结构。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _to_jsonable(fn())
            except Exception:
                pass
    return _safe_json_default(obj)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2,
                  default=_safe_json_default)
    tmp.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_to_jsonable(payload), ensure_ascii=False,
                      default=_safe_json_default)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------- 上下文 ------------------------------------------------------

@dataclass
class _CallCtx:
    """单次 chat.completions 调用的上下文（贯穿请求 / chunk / 响应 / 错误）。"""

    call_id: str
    call_dir: Path
    request_path: Path
    chunks_path: Path
    response_path: Path
    started_at: float
    streaming: bool
    call_kind: str           # "main" | "aux:<task>"
    chunk_count: int = 0
    finalized: bool = False  # 防止 finalize_stream 被重复调用


# Hermes 会把 ``client.chat.completions.create`` 放到独立的
# ``threading.Thread`` 里执行（见 ``agent/chat_completion_helpers.py``），
# 而 Python ContextVar 默认不跨线程传播。所以这里改用进程级状态 +
# 锁来让 hook（主线程）和 patched create（worker 线程）共享上下文。

_state_lock = threading.Lock()

# 由 ``on_session_start`` hook 设置；辅助调用 fallback 时归档到这里。
_global_session_id: Optional[str] = None

# 由 ``pre_api_request`` hook 设置一次，由下一次 patched create 消费一次。
# 命中时调用被分类为 ``main`` 并落到对应 turn_dir；否则走辅助分类。
_pending_main_claim: Optional[dict] = None


def set_global_session(session_id: Optional[str]) -> None:
    global _global_session_id
    with _state_lock:
        if session_id:
            _global_session_id = session_id


def get_global_session() -> Optional[str]:
    with _state_lock:
        return _global_session_id


def mark_pending_main_call(
    session_id: Optional[str],
    turn_dir: Path,
    task_id: Optional[str],
) -> None:
    """``pre_api_request`` 调用：登记下一次 ``create`` 是主对话调用。"""
    global _pending_main_claim
    with _state_lock:
        _pending_main_claim = {
            "session_id": session_id,
            "turn_dir": turn_dir,
            "task_id": task_id,
        }


def claim_pending_main_call() -> Optional[dict]:
    """patched ``create`` 调用：原子消费 pending 主调用标记。"""
    global _pending_main_claim
    with _state_lock:
        claim = _pending_main_claim
        _pending_main_claim = None
        return claim


def clear_pending_main_call() -> None:
    """``post_api_request`` 兜底清理（防止 hook 设了之后 create 没触发）。"""
    global _pending_main_claim
    with _state_lock:
        _pending_main_claim = None


# ---------- Recorder ---------------------------------------------------

class LlmTraceRecorder:
    """单例式埋点记录器。"""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # session_id -> (last_turn_dir, turn_count)
        self._sessions: dict[str, dict[str, Any]] = {}

    # ---- 目录管理 ----

    def _session_root(self, session_id: Optional[str]) -> Path:
        sid = session_id or "unknown-session"
        return self.base_dir / "sessions" / _sanitize(sid)

    def begin_turn(
        self,
        session_id: Optional[str],
        user_message: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Path:
        """开新一轮 turn 目录，返回 turn_dir。

        调用方（hook）需要把返回值连同 session_id/task_id 一起塞到
        ``mark_pending_main_call``，让 patched create 能定位到这里。
        """
        with self._lock:
            sroot = self._session_root(session_id)
            sroot.mkdir(parents=True, exist_ok=True)
            sid = session_id or "unknown-session"
            entry = self._sessions.setdefault(sid, {"turn_count": 0})
            entry["turn_count"] += 1
            n = entry["turn_count"]
            ts = time.strftime("%Y%m%d-%H%M%S")
            turn_dir = sroot / f"turn-{n:04d}-{ts}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            entry["turn_dir"] = turn_dir

            meta = {
                "session_id": session_id,
                "task_id": task_id,
                "turn_index": n,
                "started_at": time.time(),
                "user_message_preview": (user_message or "")[:500],
            }
            _dump_json(turn_dir / "_turn.json", meta)
        return turn_dir

    def aux_dir(self, session_id: Optional[str]) -> Path:
        """辅助调用 fallback 目录：``sessions/<sid>/aux/``。"""
        sroot = self._session_root(session_id or get_global_session())
        aux = sroot / "aux"
        aux.mkdir(parents=True, exist_ok=True)
        return aux

    # ---- 单次调用生命周期 ----

    def start_chat_completion_call(
        self,
        api_kwargs: dict,
        streaming: bool,
        parent_dir: Path,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        call_kind: str = "main",
    ) -> _CallCtx:
        call_id = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        kind_tag = call_kind.replace(":", "_").replace("/", "_")
        call_dir = parent_dir / f"call-{call_id}-{kind_tag}"
        call_dir.mkdir(parents=True, exist_ok=True)

        request_path = call_dir / "request.json"
        chunks_path = call_dir / "chunks.ndjson"
        response_path = call_dir / "response.json"

        request_payload = {
            "call_id": call_id,
            "call_kind": call_kind,
            "session_id": session_id,
            "task_id": task_id,
            "provider": provider,
            "model": model or api_kwargs.get("model"),
            "base_url": base_url,
            "streaming": streaming,
            "started_at": time.time(),
            "kwargs": _redact_kwargs(api_kwargs),
        }
        _dump_json(request_path, request_payload)

        return _CallCtx(
            call_id=call_id,
            call_dir=call_dir,
            request_path=request_path,
            chunks_path=chunks_path,
            response_path=response_path,
            started_at=time.time(),
            streaming=streaming,
            call_kind=call_kind,
        )

    def append_stream_chunk(self, ctx: _CallCtx, idx: int, chunk: Any) -> None:
        if ctx is None:
            return
        try:
            payload = {
                "idx": idx,
                "ts": time.time(),
                "chunk": _to_jsonable(chunk),
            }
            _append_jsonl(ctx.chunks_path, payload)
            ctx.chunk_count = idx + 1
        except Exception:
            pass

    def write_non_stream_response(self, ctx: _CallCtx, response: Any) -> None:
        if ctx is None or ctx.finalized:
            return
        ctx.finalized = True
        try:
            payload = {
                "call_id": ctx.call_id,
                "ended_at": time.time(),
                "elapsed_ms": int((time.time() - ctx.started_at) * 1000),
                "streaming": False,
                "response": _to_jsonable(response),
            }
            _dump_json(ctx.response_path, payload)
        except Exception:
            pass

    def finalize_stream(self, ctx: _CallCtx) -> None:
        """流式调用结束（正常 EOS）。汇总 chunk 信息到 response.json。"""
        if ctx is None or ctx.finalized:
            return
        ctx.finalized = True
        try:
            payload = {
                "call_id": ctx.call_id,
                "ended_at": time.time(),
                "elapsed_ms": int((time.time() - ctx.started_at) * 1000),
                "streaming": True,
                "chunk_count": ctx.chunk_count,
                "note": "完整 chunk 序列见 chunks.ndjson",
            }
            _dump_json(ctx.response_path, payload)
        except Exception:
            pass

    def write_error(self, ctx: _CallCtx, exc: BaseException) -> None:
        if ctx is None:
            return
        ctx.finalized = True
        try:
            payload = {
                "call_id": ctx.call_id,
                "ended_at": time.time(),
                "elapsed_ms": int((time.time() - ctx.started_at) * 1000),
                "streaming": ctx.streaming,
                "chunk_count": ctx.chunk_count,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            _dump_json(ctx.response_path, payload)
        except Exception:
            pass


# ---------- 辅助 -------------------------------------------------------

_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."


def _sanitize(s: str) -> str:
    out = "".join(c if c in _SAFE else "_" for c in str(s))
    return (out[:120] or "x").strip("_") or "x"


_REDACT_KEYS = {"api_key", "authorization", "x-api-key"}


def _redact_kwargs(kwargs: dict) -> dict:
    """脱敏：移除 kwargs 内可能的密钥字段（OpenAI SDK 一般不放在 kwargs，但保险起见）。"""
    out: dict[str, Any] = {}
    for k, v in (kwargs or {}).items():
        if k.lower() in _REDACT_KEYS:
            out[k] = "***REDACTED***"
        else:
            out[k] = v
    return out


# ---------- 单例 -------------------------------------------------------

_recorder: Optional[LlmTraceRecorder] = None
_recorder_lock = threading.Lock()


def init_recorder(base_dir: Path) -> LlmTraceRecorder:
    """初始化（或重用）单例。"""
    global _recorder
    with _recorder_lock:
        if _recorder is None:
            _recorder = LlmTraceRecorder(base_dir)
        return _recorder


def get_recorder() -> Optional[LlmTraceRecorder]:
    return _recorder


def disable_recorder() -> None:
    global _recorder
    with _recorder_lock:
        _recorder = None
