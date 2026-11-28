"""CardKit request identity and blocking-worker settlement.

OpenAPI requires strictly increasing sequence values, not gap-free counters.
UUID identifies one logical operation: transport ambiguity must replay the SAME
request, never allocate another UUID or guess whether the sequence was consumed.
https://open.feishu.cn/document/cardkit-v1/card/update
"""

import asyncio
import json
import logging
from contextvars import ContextVar

_cancellation_requested: ContextVar[asyncio.Event | None] = ContextVar(
    "feishu_settlement_cancellation", default=None
)


def cancellation_requested():
    signal = _cancellation_requested.get()
    return signal is not None and signal.is_set()


logger = logging.getLogger(__name__)


def _ambiguous(exc):
    from requests.exceptions import RequestException

    return isinstance(exc, (TimeoutError, json.JSONDecodeError, RequestException))


async def await_settled(awaitable, *, cancelled_error_factory=None):
    """Delay cancellation until an uncancellable SDK worker actually exits."""
    signal = _cancellation_requested.get()
    token = None
    if signal is None:
        signal = asyncio.Event()
        token = _cancellation_requested.set(signal)
    worker = asyncio.ensure_future(awaitable)
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            signal.set()
        except Exception:
            logging.getLogger(__name__).debug("Card worker failed; retrieving its result", exc_info=True)
            break
    result = None
    try:
        result = worker.result()  # retrieve failures even when the caller was cancelled
        return result
    finally:
        if token is not None:
            _cancellation_requested.reset(token)
        if cancelled:
            if cancelled_error_factory is not None and result is not None:
                raise cancelled_error_factory(result)
            raise asyncio.CancelledError()


async def mutate(adapter, card_id, function, request, error_message):
    sequence = request.request_body.sequence

    async def settle():
        for attempt in range(3):
            try:
                response = await adapter._run_blocking(function, request)
            except Exception as exc:
                if not _ambiguous(exc):
                    raise
                # Reserve this sequence even if all retries fail. A later *different*
                # operation must be newer, whether this operation committed or not.
                adapter._reserve_cardkit_sequence(card_id, sequence)
                if attempt == 2 or cancellation_requested():
                    raise
                logger.warning(
                    "[Feishu] Ambiguous CardKit mutation; replaying identical UUID/sequence for %s",
                    card_id,
                )
                continue
            if not adapter._response_succeeded(response):
                raise RuntimeError(
                    adapter._response_error_result(response, default_message=error_message).error
                )
            adapter._commit_cardkit_sequence(card_id, sequence)
            return

    # Protect the complete logical operation (including retries and commit).
    await await_settled(settle())
