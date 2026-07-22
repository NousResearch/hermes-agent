"""Regression tests for ``CDPSupervisor._attach_initial_page`` (#69331).

The real supervisor talks CDP over a WebSocket; these tests drive
``_attach_initial_page`` directly against a fake ``_cdp`` whose responses
(and response delays) we control. That lets us exercise the
reuse-with-timeout-and-fallback path without a live Chrome.

What we're guarding against: when the reused remote page never replies to
``Page.enable`` on the flattened session, the supervisor used to hang
silently and every ``browser_navigate`` timed out at 120 s. The fix races
``Page.enable`` against ``_CDP_HANDSHAKE_TIMEOUT_S`` and falls back to a
fresh ``about:blank`` target.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple

import pytest

from tools import browser_supervisor as _bs_mod
from tools.browser_supervisor import CDPSupervisor


def _make_supervisor() -> CDPSupervisor:
    """Build a CDPSupervisor without starting its loop/thread.

    ``_attach_initial_page`` only touches ``self._cdp`` (which the tests
    replace) and ``self._page_session_id`` / ``self._install_dialog_bridge``,
    so a bare constructor instance is enough.
    """
    return CDPSupervisor(task_id="pytest-69331", cdp_url="ws://fake/cdp")


@pytest.fixture
def fast_handshake_timeout(monkeypatch):
    """Shrink the handshake timeout to keep the hang-fallback tests fast.

    The production default is 5 s; tests that exercise the fallback path
    wait for that timeout to fire, so we patch the module global down to a
    fraction of a second. ``_attach_and_enable`` reads the global at call
    time, so monkeypatching before ``_attach_initial_page`` runs is enough.
    """
    monkeypatch.setattr(_bs_mod, "_CDP_HANDSHAKE_TIMEOUT_S", 0.15)
    return 0.15


class FakeCDP:
    """Records calls and dispatches canned responses, optionally delayed.

    ``handlers`` maps method name -> either a dict (returned as the CDP
    ``{"result": ...}`` frame) or a callable ``(params, session_id) ->
    dict | awaitable dict``. A callable lets a test hang (e.g. sleep longer
    than the handshake timeout) to simulate an unresponsive page.

    The fake mimics ``CDPSupervisor._cdp``'s contract: it returns the
    ``{"result": ...}`` payload and honors the ``timeout`` kwarg by raising
    ``asyncio.TimeoutError`` when the handler doesn't resolve in time —
    exactly what the real ``asyncio.wait_for`` would do.
    """

    def __init__(
        self,
        handlers: Dict[str, Any],
        delays: Optional[Dict[str, float]] = None,
    ) -> None:
        self._handlers = handlers
        self._delays = delays or {}
        self.calls: List[Tuple[str, Optional[Dict[str, Any]], Optional[str]]] = []

    async def __call__(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        self.calls.append((method, params, session_id))

        async def resolve() -> Dict[str, Any]:
            # Optional artificial delay *before* the handler runs, for tests
            # that want a slow-but-successful response.
            delay = self._delays.get(method)
            if delay is not None:
                await asyncio.sleep(delay)
            handler = self._handlers.get(method)
            if handler is None:
                raise RuntimeError(f"FakeCDP has no handler for {method!r}")
            if callable(handler):
                result = handler(params, session_id)
                if inspect.isawaitable(result):
                    result = await result
            else:
                result = handler
            # The real ``_cdp`` returns the full CDP frame
            # (``{"id": .., "result": ..}``) and production code indexes
            # ``["result"]`` off it. Mirror that contract so handlers only
            # have to specify the result body.
            if isinstance(result, dict) and "result" in result:
                return result
            return {"result": result}

        # Honor the per-call timeout exactly like the real ``_cdp``'s
        # ``asyncio.wait_for``: a handler that sleeps past ``timeout``
        # (e.g. a hanging Page.enable) raises asyncio.TimeoutError instead
        # of resolving late. This is what triggers the fallback path.
        return await asyncio.wait_for(resolve(), timeout=timeout)

    def calls_for(self, method: str) -> List[Tuple[str, Optional[Dict[str, Any]], Optional[str]]]:
        return [c for c in self.calls if c[0] == method]


def _page_target(target_id: str) -> Dict[str, Any]:
    return {"type": "page", "targetId": target_id, "url": "https://example.invalid/"}


def _browser_target(target_id: str) -> Dict[str, Any]:
    return {"type": "browser", "targetId": target_id}


# A baseline set of "everything succeeds instantly" handlers, parametrized by
# the targetId we expect to attach to. Tests override individual entries.
def _ok_handlers(attach_target_id: str, session_id: str = "sess-1") -> Dict[str, Any]:
    return {
        "Target.getTargets": {"targetInfos": []},
        "Target.createTarget": {"targetId": "fresh-blank"},
        "Target.attachToTarget": {"sessionId": session_id},
        "Page.enable": {},
        "Runtime.enable": {},
        "Target.setAutoAttach": {},
        "Page.addScriptToEvaluateOnNewDocument": {},
        "Fetch.enable": {},
        "Runtime.evaluate": {},
    }


async def _attach(sv: CDPSupervisor, fake: FakeCDP) -> None:
    """Run _attach_initial_page with _cdp + _install_dialog_bridge stubbed.

    ``_install_dialog_bridge`` issues its own CDP calls via ``_cdp``; we keep
    it real so the fallback/reuse paths exercise the same call sequence the
    production code does, with the fake answering every method.
    """
    sv._cdp = fake  # type: ignore[method-assign]
    await sv._attach_initial_page()


# ── 1. No existing pages → create blank ────────────────────────────────────


@pytest.mark.asyncio
async def test_attach_with_no_existing_pages_creates_blank() -> None:
    sv = _make_supervisor()
    fake = FakeCDP(_ok_handlers(attach_target_id="fresh-blank"))
    await _attach(sv, fake)

    # No page in getTargets -> createTarget(about:blank) -> attach to it.
    creates = fake.calls_for("Target.createTarget")
    assert len(creates) == 1
    assert creates[0][1] == {"url": "about:blank"}
    attaches = fake.calls_for("Target.attachToTarget")
    assert len(attaches) == 1
    assert attaches[0][1] == {"targetId": "fresh-blank", "flatten": True}
    # Page.enable was issued on the freshly-attached session.
    enables = fake.calls_for("Page.enable")
    assert len(enables) == 1
    assert enables[0][2] == "sess-1"
    assert sv._page_session_id == "sess-1"
    # We never tried to close anything in this path.
    assert fake.calls_for("Target.closeTarget") == []


# ── 2. Responsive existing page → reuse it ────────────────────────────────


@pytest.mark.asyncio
async def test_attach_with_responsive_existing_page_reuses_it() -> None:
    sv = _make_supervisor()
    handlers = _ok_handlers(attach_target_id="existing-1")
    handlers["Target.getTargets"] = {
        "targetInfos": [_page_target("existing-1"), _browser_target("browser-1")]
    }
    handlers["Target.attachToTarget"] = {"sessionId": "reuse-sess"}
    fake = FakeCDP(handlers)
    await _attach(sv, fake)

    # Reuse path: no createTarget at all.
    assert fake.calls_for("Target.createTarget") == []
    attaches = fake.calls_for("Target.attachToTarget")
    assert len(attaches) == 1
    assert attaches[0][1] == {"targetId": "existing-1", "flatten": True}
    assert sv._page_session_id == "reuse-sess"
    # The first page-type target is the one picked, not the browser target.
    assert fake.calls_for("Target.closeTarget") == []


# ── 3. Unresponsive existing page → create blank ──────────────────────────


@pytest.mark.asyncio
async def test_attach_with_unresponsive_existing_page_creates_blank(
    fast_handshake_timeout,
) -> None:
    sv = _make_supervisor()
    handlers = _ok_handlers(attach_target_id="existing-hang")
    handlers["Target.getTargets"] = {"targetInfos": [_page_target("existing-hang")]}

    # The first attach (to the hanging page) gets its own session; the fallback
    # attach (to the fresh blank) gets a different one.
    attach_sessions = iter(["hang-sess", "fresh-sess"])

    def attach_handler(params, session_id):
        return {"sessionId": next(attach_sessions)}

    handlers["Target.attachToTarget"] = attach_handler

    # Page.enable on the hanging session sleeps well past the handshake
    # timeout. The real _cdp would raise asyncio.TimeoutError; the fake does
    # the same because we pass timeout=_CDP_HANDSHAKE_TIMEOUT_S through.
    hang_delay = fast_handshake_timeout + 2.0

    async def page_enable_handler(params, session_id):
        await asyncio.sleep(hang_delay)
        return {}

    # Only the *first* Page.enable should hang; the fallback page's enable
    # must succeed so attach completes. Dispatch on the session id we handed
    # out above.
    fresh_seen = {"v": False}

    async def page_enable_dispatch(params, session_id):
        if session_id == "fresh-sess":
            fresh_seen["v"] = True
            return {}
        await asyncio.sleep(hang_delay)
        return {}

    handlers["Page.enable"] = page_enable_dispatch
    # Force the fake to apply the handshake timeout to Page.enable.
    fake = FakeCDP(handlers)

    # Patch _cdp to forward the Page.enable timeout. FakeCDP already honors
    # the ``timeout`` kwarg, and _attach_and_enable passes
    # timeout=_CDP_HANDSHAKE_TIMEOUT_S — so the first Page.enable raises
    # asyncio.TimeoutError and we fall back.
    await _attach(sv, fake)

    # We fell back: a fresh about:blank was created.
    creates = fake.calls_for("Target.createTarget")
    assert len(creates) == 1
    assert creates[0][1] == {"url": "about:blank"}
    # Final session is the fresh one.
    assert sv._page_session_id == "fresh-sess"
    assert fresh_seen["v"] is True


# ── 4. Unresponsive page is closed before the blank is created ────────────


@pytest.mark.asyncio
async def test_attach_closes_unresponsive_page_before_creating_blank(
    fast_handshake_timeout,
) -> None:
    sv = _make_supervisor()
    handlers = _ok_handlers(attach_target_id="existing-hang-2")
    handlers["Target.getTargets"] = {"targetInfos": [_page_target("existing-hang-2")]}
    attach_sessions = iter(["hang-sess-2", "fresh-sess-2"])

    def attach_handler(params, session_id):
        return {"sessionId": next(attach_sessions)}

    handlers["Target.attachToTarget"] = attach_handler

    hang_delay = fast_handshake_timeout + 2.0

    async def page_enable_dispatch(params, session_id):
        if session_id == "fresh-sess-2":
            return {}
        await asyncio.sleep(hang_delay)
        return {}

    handlers["Page.enable"] = page_enable_dispatch
    fake = FakeCDP(handlers)
    await _attach(sv, fake)

    method_order = [c[0] for c in fake.calls]

    # The hanging target must be closed *before* the replacement is created.
    close_idx = method_order.index("Target.closeTarget")
    create_idx = method_order.index("Target.createTarget")
    assert close_idx < create_idx, (
        f"closeTarget must precede createTarget; order was: {method_order}"
    )

    # And the closed target is the unresponsive page, not a browser target.
    closes = fake.calls_for("Target.closeTarget")
    assert len(closes) == 1
    assert closes[0][1] == {"targetId": "existing-hang-2"}


# ── 5. closeTarget failure doesn't block the fallback ─────────────────────


@pytest.mark.asyncio
async def test_attach_falls_back_even_if_close_target_raises(
    fast_handshake_timeout,
) -> None:
    """If closeTarget itself errors, we still create + attach the blank."""
    sv = _make_supervisor()
    handlers = _ok_handlers(attach_target_id="existing-hang-3")
    handlers["Target.getTargets"] = {"targetInfos": [_page_target("existing-hang-3")]}
    attach_sessions = iter(["hang-sess-3", "fresh-sess-3"])

    def attach_handler(params, session_id):
        return {"sessionId": next(attach_sessions)}

    handlers["Target.attachToTarget"] = attach_handler
    handlers["Target.closeTarget"] = _raise_runtime_error

    hang_delay = fast_handshake_timeout + 2.0

    async def page_enable_dispatch(params, session_id):
        if session_id == "fresh-sess-3":
            return {}
        await asyncio.sleep(hang_delay)
        return {}

    handlers["Page.enable"] = page_enable_dispatch
    fake = FakeCDP(handlers)
    await _attach(sv, fake)

    assert sv._page_session_id == "fresh-sess-3"
    assert len(fake.calls_for("Target.createTarget")) == 1


async def _raise_runtime_error(params, session_id):  # noqa: ANN201
    raise RuntimeError("closeTarget failed on purpose")
