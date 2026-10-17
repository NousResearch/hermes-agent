"""Telegram initialization retries and failed-attempt retirement."""

import asyncio
import contextlib
import sys
from pathlib import Path


def _initialization_await_chain(task) -> str:
    """Describe suspended Python awaits without task reprs, source text or frame locals."""
    current = task.get_coro() if isinstance(task, asyncio.Task) else None
    chain = []
    for _ in range(32):
        frame = getattr(current, "cr_frame", None) or getattr(current, "gi_frame", None)
        if frame is not None:
            code = frame.f_code
            chain.append(f"{Path(code.co_filename).name}:{frame.f_lineno} ({code.co_name})")
        current = getattr(current, "cr_await", None) or getattr(current, "gi_yieldfrom", None)
        if current is None:
            break
    return " -> ".join(chain) or "<no suspended Python frame>"


async def initialize_app_with_retries(adapter, builder) -> None:
    """``app.initialize()`` with a bounded retry ladder; rebuilds ``adapter._app``/``adapter._bot`` from
    ``builder`` after each failed attempt; OSError when the per-attempt or total watchdog expires."""
    from plugins.platforms.telegram import adapter as telegram

    _max_connect = 8
    _init_timeout = telegram.env_float("HERMES_TELEGRAM_INIT_TIMEOUT", 30.0)  # per attempt
    # Total watchdog: bounds the whole connect loop even if the retry loop silently stalls.
    _total_deadline = asyncio.get_running_loop().time() + _init_timeout * _max_connect + 120.0
    _timed_out = f"Telegram initialization timed out after {_max_connect} attempts ({_init_timeout:.0f}s each)"
    for _attempt in range(_max_connect):
        rebuild_app = False
        try:
            if asyncio.get_running_loop().time() >= _total_deadline:
                raise OSError(
                    f"{_timed_out} — total connect watchdog deadline ({_init_timeout * _max_connect + 120.0:.0f}s) exceeded. "
                    f"Check network connectivity to api.telegram.org or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT / "
                    f"HERMES_TELEGRAM_INIT_TIMEOUT to a lower value.")
            telegram.logger.warning("[%s] Connecting to Telegram (attempt %d/%d)…", adapter.name, _attempt + 1, _max_connect)
            # On timeout the (possibly shielded) initialize() task is abandoned; release the half-built
            # app's httpx client so it isn't leaked across the ladder.
            init_task = asyncio.ensure_future(adapter._app.initialize())
            await telegram._await_with_thread_deadline(
                init_task, timeout=_init_timeout, on_abandon=lambda app=adapter._app: telegram._shutdown_abandoned_app(app),
                label="telegram-init")
            break
        except TimeoutError:
            # Stay on this loop tick: cancellation or cleanup can erase the suspended frames.
            telegram.logger.warning(
                "[%s] Telegram initialization await chain: %s", adapter.name, _initialization_await_chain(init_task))
            rebuild_app = True
            if _attempt >= _max_connect - 1:
                raise OSError(
                    f"{_timed_out}. Check network connectivity to api.telegram.org "
                    f"or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT to a lower value.")
            wait = min(2 ** _attempt, 15)
            telegram.logger.warning(
                "[%s] Connect attempt %d/%d timed out after %.0fs — retrying in %ds", adapter.name, _attempt + 1,
                _max_connect, _init_timeout, wait)
            await asyncio.sleep(wait)
        except Exception as init_err:
            # OSError always retries; anything else only when it looks like a network error.
            if (not isinstance(init_err, OSError) and not adapter._looks_like_network_error(init_err)) or _attempt >= _max_connect - 1:
                raise
            rebuild_app = True
            wait = min(2 ** _attempt, 15)
            telegram.logger.warning(
                "[%s] Connect attempt %d/%d failed: %s — retrying in %ds", adapter.name, _attempt + 1, _max_connect, init_err, wait)
            await asyncio.sleep(wait)
        except BaseException:
            # CancelledError etc.: log for the operator, then reraise. LAST so the Exception handlers win.
            telegram.logger.warning(
                "[%s] Connect attempt %d/%d interrupted by %s — propagating", adapter.name, _attempt + 1, _max_connect,
                "CancelledError" if isinstance(sys.exc_info()[1], asyncio.CancelledError) else type(sys.exc_info()[1]).__name__)
            raise
        finally:
            # PTB retains explicitly supplied requests in the builder. An abandoned initialize and
            # its delayed cleanup must never share connection state with the next attempt.
            if rebuild_app and _attempt < _max_connect - 1:
                old_app = adapter._app
                request, get_updates_request = await adapter._build_ptb_requests()
                builder.request(request).get_updates_request(get_updates_request)
                adapter._app = builder.build()
                adapter._bot = adapter._app.bot
                # Same order as connect(): plugin handlers first (the wired-set is keyed per app, so
                # the rebuilt app gets them again), then core and the observer in lockstep.
                adapter._wire_plugin_handlers(adapter._app)
                adapter._register_handlers(adapter._app)
                with contextlib.suppress(Exception):
                    await telegram._await_with_thread_deadline(
                        telegram._shutdown_abandoned_app(old_app), timeout=telegram._DRAIN_TIMEOUT,
                        label="telegram-init-cleanup", dump_on_blocked_loop=False)
