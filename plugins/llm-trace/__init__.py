"""llm-trace 插件入口。

零核心入侵，覆盖以下 LLM 调用：

1. **主对话**：通过 ``pre_api_request`` hook 划分 turn 边界。
2. **辅助调用**（compression / vision / title_generation 等）：通过
   ``openai.resources.chat.completions.Completions.create`` 的运行时
   包装捕获——同时通过栈帧探测 ``agent/auxiliary_client.py::call_llm``
   的 ``task`` 参数，记录到 ``call_kind`` 字段。
3. **流式**：包装 ``Stream`` / ``AsyncStream`` 的 ``__iter__`` /
   ``__aiter__``，逐 chunk 落入 ``chunks.ndjson``。

激活：
    1. 添加到 ``~/.hermes/config.yaml``::

           plugins:
             enabled:
               - llm-trace

    2. （可选）``HERMES_LLM_TRACE_DIR=<abs_path>`` 自定义输出目录；
       默认 ``$HERMES_HOME/llm-traces/``。

    3. （可选）``HERMES_LLM_TRACE=0`` 显式关闭：插件保持加载但不记录。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from . import recorder as _rec_mod
from .recorder import (
    LlmTraceRecorder,
    claim_pending_main_call,
    clear_pending_main_call,
    get_global_session,
    get_recorder,
    init_recorder,
    mark_pending_main_call,
    set_global_session,
)

logger = logging.getLogger(__name__)

# ---- 全局开关 ------------------------------------------------------------

def _is_disabled_via_env() -> bool:
    val = os.environ.get("HERMES_LLM_TRACE", "").strip().lower()
    return val in {"0", "false", "off", "no"}


def _resolve_base_dir() -> Path:
    explicit = os.environ.get("HERMES_LLM_TRACE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "llm-traces"
    except Exception:
        return Path.home() / ".hermes" / "llm-traces"


# ---- monkey-patch 标记 --------------------------------------------------

_PATCH_MARKER = "_hermes_llm_trace_patched"


# ---- 辅助调用 task 检测 -------------------------------------------------

def _detect_aux_task() -> Optional[str]:
    """从调用栈里找 ``agent/auxiliary_client.py::call_llm``，读取 ``task``。"""
    try:
        frame = sys._getframe(1)
    except Exception:
        return None
    depth = 0
    while frame is not None and depth < 30:
        co = frame.f_code
        fname = co.co_filename
        if fname.endswith("auxiliary_client.py") and co.co_name in (
            "call_llm",
            "async_call_llm",
            "_call_llm_impl",
            "_call_llm_with_failover",
        ):
            task = frame.f_locals.get("task")
            if isinstance(task, str):
                return task
        frame = frame.f_back
        depth += 1
    return None


def _classify_aux_call() -> str:
    """patch 在没有 main claim 命中时调用，给辅助调用打标签。"""
    aux_task = _detect_aux_task()
    if aux_task:
        return f"aux:{aux_task}"
    return "aux:unknown"


def _client_base_url(self_ref: Any) -> str:
    try:
        client = getattr(self_ref, "_client", None)
        if client is None:
            return ""
        return str(getattr(client, "base_url", "") or "")
    except Exception:
        return ""


# ---- streaming 包装 -----------------------------------------------------

class _TracingStreamProxy:
    """同步 Stream 代理。

    透明转发所有非 ``__iter__`` 调用（包括 ``response``、``__enter__`` 等
    上游用到的属性），仅在迭代时插入 chunk 落盘。
    """

    __slots__ = ("_s", "_rec", "_ctx", "_idx", "_consumed")

    def __init__(self, stream, rec, ctx) -> None:
        object.__setattr__(self, "_s", stream)
        object.__setattr__(self, "_rec", rec)
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_idx", 0)
        object.__setattr__(self, "_consumed", False)

    def __iter__(self):
        try:
            for chunk in self._s:
                self._rec.append_stream_chunk(self._ctx, self._idx, chunk)
                object.__setattr__(self, "_idx", self._idx + 1)
                yield chunk
        except BaseException as e:
            try:
                self._rec.write_error(self._ctx, e)
            except Exception:
                pass
            raise
        else:
            object.__setattr__(self, "_consumed", True)
            try:
                self._rec.finalize_stream(self._ctx)
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_s"), name)

    def __enter__(self):
        s = object.__getattribute__(self, "_s")
        if hasattr(s, "__enter__"):
            s.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        s = object.__getattribute__(self, "_s")
        if exc is not None and not self._consumed:
            try:
                self._rec.write_error(self._ctx, exc)
            except Exception:
                pass
        if hasattr(s, "__exit__"):
            return s.__exit__(exc_type, exc, tb)
        return None

    def close(self):
        s = object.__getattribute__(self, "_s")
        if hasattr(s, "close"):
            return s.close()


class _TracingAsyncStreamProxy:
    """异步 Stream 代理（同 ``_TracingStreamProxy`` 但走 ``__aiter__``）。"""

    __slots__ = ("_s", "_rec", "_ctx", "_idx", "_consumed")

    def __init__(self, stream, rec, ctx) -> None:
        object.__setattr__(self, "_s", stream)
        object.__setattr__(self, "_rec", rec)
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_idx", 0)
        object.__setattr__(self, "_consumed", False)

    async def __aiter__(self):
        try:
            async for chunk in self._s:
                self._rec.append_stream_chunk(self._ctx, self._idx, chunk)
                object.__setattr__(self, "_idx", self._idx + 1)
                yield chunk
        except BaseException as e:
            try:
                self._rec.write_error(self._ctx, e)
            except Exception:
                pass
            raise
        else:
            object.__setattr__(self, "_consumed", True)
            try:
                self._rec.finalize_stream(self._ctx)
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_s"), name)

    async def __aenter__(self):
        s = object.__getattribute__(self, "_s")
        if hasattr(s, "__aenter__"):
            await s.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        s = object.__getattribute__(self, "_s")
        if exc is not None and not self._consumed:
            try:
                self._rec.write_error(self._ctx, exc)
            except Exception:
                pass
        if hasattr(s, "__aexit__"):
            return await s.__aexit__(exc_type, exc, tb)
        return None

    async def aclose(self):
        s = object.__getattribute__(self, "_s")
        if hasattr(s, "aclose"):
            return await s.aclose()


# ---- monkey-patch openai SDK -------------------------------------------

def _install_openai_patch() -> bool:
    """对 openai SDK 的 ``Completions.create`` / ``AsyncCompletions.create`` 打补丁。"""
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except Exception as e:
        logger.warning("llm-trace: 未安装 openai SDK，跳过 patch (%s)", e)
        return False

    if getattr(Completions, _PATCH_MARKER, False):
        return True

    orig_sync = Completions.create
    orig_async = AsyncCompletions.create

    def _resolve_target(rec: LlmTraceRecorder):
        """返回 (parent_dir, session_id, task_id, call_kind)。"""
        claim = claim_pending_main_call()
        if claim is not None:
            return (
                claim["turn_dir"],
                claim.get("session_id"),
                claim.get("task_id"),
                "main",
            )
        sid = get_global_session()
        return (rec.aux_dir(sid), sid, None, _classify_aux_call())

    def _sync_create(self, *args, **kwargs):
        rec = get_recorder()
        if rec is None or _is_disabled_via_env():
            return orig_sync(self, *args, **kwargs)

        is_streaming = bool(kwargs.get("stream"))
        parent_dir, sid, tid, call_kind = _resolve_target(rec)
        ctx = None
        try:
            ctx = rec.start_chat_completion_call(
                api_kwargs=kwargs,
                streaming=is_streaming,
                parent_dir=parent_dir,
                session_id=sid,
                task_id=tid,
                model=kwargs.get("model"),
                base_url=_client_base_url(self),
                call_kind=call_kind,
            )
        except Exception as e:
            logger.debug("llm-trace start_call failed: %s", e)
            return orig_sync(self, *args, **kwargs)

        try:
            result = orig_sync(self, *args, **kwargs)
        except BaseException as e:
            try:
                rec.write_error(ctx, e)
            except Exception:
                pass
            raise

        if not is_streaming:
            try:
                rec.write_non_stream_response(ctx, result)
            except Exception:
                pass
            return result

        return _TracingStreamProxy(result, rec, ctx)

    async def _async_create(self, *args, **kwargs):
        rec = get_recorder()
        if rec is None or _is_disabled_via_env():
            return await orig_async(self, *args, **kwargs)

        is_streaming = bool(kwargs.get("stream"))
        parent_dir, sid, tid, call_kind = _resolve_target(rec)
        ctx = None
        try:
            ctx = rec.start_chat_completion_call(
                api_kwargs=kwargs,
                streaming=is_streaming,
                parent_dir=parent_dir,
                session_id=sid,
                task_id=tid,
                model=kwargs.get("model"),
                base_url=_client_base_url(self),
                call_kind=call_kind,
            )
        except Exception as e:
            logger.debug("llm-trace start_call failed: %s", e)
            return await orig_async(self, *args, **kwargs)

        try:
            result = await orig_async(self, *args, **kwargs)
        except BaseException as e:
            try:
                rec.write_error(ctx, e)
            except Exception:
                pass
            raise

        if not is_streaming:
            try:
                rec.write_non_stream_response(ctx, result)
            except Exception:
                pass
            return result

        return _TracingAsyncStreamProxy(result, rec, ctx)

    Completions.create = _sync_create
    AsyncCompletions.create = _async_create
    setattr(Completions, _PATCH_MARKER, True)
    setattr(AsyncCompletions, _PATCH_MARKER, True)
    logger.info("llm-trace: openai SDK patched (sync + async)")
    return True


# ---- hook 实现 ----------------------------------------------------------

_current_turn_dir_by_session: dict = {}


def _on_session_start(**kwargs) -> None:
    """会话开始：把 session_id 设到全局，让辅助调用也能正确归档。"""
    sid = kwargs.get("session_id")
    if sid:
        set_global_session(sid)


def _on_pre_api_request(**kwargs) -> None:
    """主对话路径：

    - ``api_call_count == 1`` 时是新一轮 turn（用户刚发消息），创建 turn 目录。
    - 后续 ``api_call_count > 1`` 是同一 turn 内的工具回调，复用同一 turn_dir。
    - 把 (session_id, turn_dir, task_id) 写入 pending claim，让 worker 线程
      上的 patched ``create`` 能识别"下一个 create 是主调用"。
    """
    rec = get_recorder()
    if rec is None or _is_disabled_via_env():
        return
    sid = kwargs.get("session_id") or get_global_session()
    if sid:
        set_global_session(sid)
    api_call_count = kwargs.get("api_call_count", 0) or 0
    task_id = kwargs.get("task_id")

    try:
        if api_call_count <= 1 or sid not in _current_turn_dir_by_session:
            turn_dir = rec.begin_turn(
                session_id=sid or "",
                user_message=kwargs.get("user_message"),
                task_id=task_id,
            )
            _current_turn_dir_by_session[sid or ""] = turn_dir
        else:
            turn_dir = _current_turn_dir_by_session[sid]
    except Exception as e:
        logger.debug("llm-trace begin_turn failed: %s", e)
        return

    mark_pending_main_call(session_id=sid, turn_dir=turn_dir, task_id=task_id)


def _on_post_api_request(**kwargs) -> None:
    """主对话收尾：清掉未消费的 pending claim，避免污染下一次。"""
    clear_pending_main_call()


# ---- 注册入口 -----------------------------------------------------------

def register(ctx) -> None:
    if _is_disabled_via_env():
        logger.info("llm-trace: HERMES_LLM_TRACE 关闭，仅加载不启用记录")
        return

    base_dir = _resolve_base_dir()
    init_recorder(base_dir)
    logger.info("llm-trace: 启用，输出目录 %s", base_dir)

    _install_openai_patch()

    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
