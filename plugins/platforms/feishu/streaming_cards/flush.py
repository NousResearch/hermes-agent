"""通用节流调度器 — FlushController."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

_logger = logging.getLogger("hermes_lark_streaming")


CARDKIT_MS = 0.100  # CardKit 流式 API 的刷新间隔
LONG_GAP_MS = 2.000  # 超过此间隔 → 认为是长时间空闲
BATCH_AFTER_GAP_MS = 0.300  # 长时间空闲后等待这个时间再 flush


class FlushController:
    """带互斥锁 + 延迟刷新的通用节流调度器.

    不包含飞书业务逻辑，只负责决定何时执行回调.
    """

    def __init__(self, throttle_ms: float = CARDKIT_MS, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._throttle_ms = throttle_ms
        self._flush_in_progress = False
        self._needs_reflush = False
        self._pending_timer: asyncio.TimerHandle | None = None
        self._last_update_time = 0.0
        self._completed = False
        self._card_message_ready = False
        self._flush_resolvers: list[asyncio.Future[None]] = []
        self._loop = loop if loop is not None else asyncio.get_running_loop()

    @property
    def throttle_ms(self) -> float:
        return self._throttle_ms

    def schedule_update(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        """请求一次节流后的卡片刷新.

        do_flush: async callable，执行实际 API 调用.
        """
        if self._completed or not self._card_message_ready:
            return
        now = time.monotonic()
        elapsed = now - self._last_update_time

        if elapsed >= self._throttle_ms:
            # 超出节流窗口
            if elapsed > LONG_GAP_MS:
                # 长时间空闲 → 延迟一小批让内容更完整
                if self._pending_timer is None:
                    self._schedule(delay=BATCH_AFTER_GAP_MS, do_flush=do_flush)
            else:
                # 立即 flush
                self._do_flush_task(do_flush)
        else:
            # 仍在节流窗口内 → 延迟到窗口边界
            if self._pending_timer is None:
                delay = self._throttle_ms - elapsed
                self._schedule(delay=delay, do_flush=do_flush)

    async def flush_now(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        """立即执行一次 flush，等待完成."""
        if self._completed or not self._card_message_ready:
            return
        self._cancel_timer()
        await self._do_flush(do_flush)

    async def wait_for_flush(self) -> None:
        """等待进行中的 flush 完成.

        若已 mark_completed（complete 收尾路径会先调 mark_completed 再进
        wait_for_flush），直接返回——resolvers 已被清空，再等会永久挂起
        （2026-08-01 实测：卡片停在 streaming 态 + 省略号跳动，无异常日志）。
        """
        if self._completed:
            return
        if not self._flush_in_progress:
            return
        future: asyncio.Future[None] = self._loop.create_future()
        self._flush_resolvers.append(future)
        await future

    def mark_completed(self) -> None:
        """标记完成，不再接受新更新."""
        self._completed = True
        self._cancel_timer()
        for r in self._flush_resolvers:
            if not r.done():
                r.set_result(None)
        self._flush_resolvers.clear()

    def reset_for_reactivate(self) -> None:
        """重激活：撤销 mark_completed，重新接受更新.

        用于跨回合合并（background 回合复用已 COMPLETED 的卡片）：终态卡片重新接纳
        后续 segment 增量，靠 batch_update 追加（streaming_mode 已关，打字机失效但内容会更新）。
        """
        self._completed = False
        self._flush_in_progress = False
        self._needs_reflush = False
        self._cancel_timer()
        self._last_update_time = time.monotonic()

    def set_throttle(self, ms: float) -> None:
        self._throttle_ms = ms

    def set_card_message_ready(self, ready: bool) -> None:
        """设置卡片消息已就绪，初始化时间戳."""
        self._card_message_ready = ready
        if ready:
            self._last_update_time = time.monotonic()

    def _schedule(self, delay: float, do_flush: Callable[[], Awaitable[None]]) -> None:
        self._cancel_timer()
        self._pending_timer = self._loop.call_later(
            delay,
            self._do_flush_task,
            do_flush,
        )

    def _do_flush_task(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        self._pending_timer = None
        self._loop.call_soon(asyncio.create_task, self._do_flush(do_flush))

    async def _do_flush(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        if self._completed or self._flush_in_progress:
            self._needs_reflush = True
            return

        self._flush_in_progress = True
        self._needs_reflush = False
        try:
            await do_flush()
        except Exception:
            _logger.debug("flush error suppressed", exc_info=True)
        finally:
            self._flush_in_progress = False
            self._last_update_time = time.monotonic()
            # 唤醒等待者
            resolvers = self._flush_resolvers
            self._flush_resolvers = []
            for r in resolvers:
                if not r.done():
                    r.set_result(None)

        # 如果 flush 期间又有新数据 → 立即重刷
        if self._needs_reflush and not self._completed:
            self._needs_reflush = False
            self._loop.call_soon(asyncio.create_task, self._do_flush(do_flush))

    def _cancel_timer(self) -> None:
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None
