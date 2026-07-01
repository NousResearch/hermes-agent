"""Regression lock: Telegram restart/reconnect message-loss parity.

Discord had a drain-window message-loss bug (PR #157) because it is a *push*
transport with no server-side redelivery. Telegram is a *pull* transport
(getUpdates long-poll) with a server-side update queue, so downtime is
backfilled for free — BUT only because of three separate properties that a
future refactor could each silently break. This file locks those properties as
behavior contracts so the loss bug can never appear on Telegram.

Spike writeup: ~/.hermes/plans/2026-07-01_telegram-parity-SPIKE.md
SPEC:          ~/.hermes/plans/2026-07-01_telegram-parity-regression-SPEC.md

INV-1  reconnect (is_reconnect=True) preserves the server-side queue
       (drop_pending_updates=False); cold boot (is_reconnect=False) drops it.
INV-2  every non-bootstrap start_polling recovery ladder preserves the queue
       (drop_pending_updates=False) unconditionally.
INV-3  disconnect() drains the handler queue before shutdown:
       updater.stop() -> app.stop() (update_queue.join) -> app.shutdown().
"""

import ast
import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    # Only shim when the REAL python-telegram-bot isn't already importable. A
    # bare ``hasattr(mod, "__file__")`` is insufficient — a MagicMock satisfies
    # any attribute — so a prior test's mock in sys.modules would be misread as
    # "real telegram present" and skip our setup. Distinguish a genuine module
    # (a real __file__ string, not a Mock) from a MagicMock explicitly.
    existing = sys.modules.get("telegram")
    if existing is not None and not isinstance(existing, MagicMock):
        file_attr = getattr(existing, "__file__", None)
        if isinstance(file_attr, str):
            return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    telegram_mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.discover_fallback_ips", _noop
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.HTTPXRequest",
        lambda **kwargs: MagicMock(),
    )


async def _cancel_heartbeat(adapter):
    """Cancel the lifetime heartbeat task connect() starts in polling mode."""
    task = getattr(adapter, "_polling_heartbeat_task", None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    adapter._polling_heartbeat_task = None


def _build_connect_harness(monkeypatch, captured):
    """Wire the real connect() builder chain against mocks, capturing the
    kwargs the real code passes to start_polling and delete_webhook."""
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    async def fake_start_polling(**kwargs):
        captured["start_polling"] = kwargs

    async def fake_delete_webhook(**kwargs):
        captured.setdefault("delete_webhook", kwargs)

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(
        set_my_commands=AsyncMock(),
        delete_webhook=AsyncMock(side_effect=fake_delete_webhook),
    )
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    return app


# ── INV-1: reconnect preserves the queue; cold boot drops it ──────────────


@pytest.mark.asyncio
async def test_reconnect_preserves_pending_updates(monkeypatch):
    """AC-1: connect(is_reconnect=True) must NOT drop the server-side queue —
    otherwise every message sent during an outage is silently lost."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    captured = {}
    _build_connect_harness(monkeypatch, captured)

    ok = await adapter.connect(is_reconnect=True)

    assert ok is True
    assert "start_polling" in captured, "connect() never reached start_polling"
    assert captured["start_polling"]["drop_pending_updates"] is False, (
        "reconnect MUST preserve pending updates (drop_pending_updates=False); "
        "a True here silently drops every message queued during the outage"
    )
    await _cancel_heartbeat(adapter)


@pytest.mark.asyncio
async def test_cold_boot_drops_pending_updates(monkeypatch):
    """AC-2: connect(is_reconnect=False) drops the stale queue on a cold first
    boot (intentional, #46621). This is the negative control that proves the
    test can actually distinguish the two paths."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    captured = {}
    _build_connect_harness(monkeypatch, captured)

    ok = await adapter.connect(is_reconnect=False)

    assert ok is True
    assert captured["start_polling"]["drop_pending_updates"] is True, (
        "cold first boot should drop the stale Bot API queue (is_reconnect=False "
        "=> drop_pending_updates=True); if this flips, the test can no longer "
        "tell reconnect from cold boot"
    )
    await _cancel_heartbeat(adapter)


# ── INV-2: every non-bootstrap recovery poll preserves the queue ──────────


def _start_polling_calls_in_adapter():
    """Parse adapter.py and return, for each `start_polling(` call, a tuple of
    ``(lineno, enclosing_function_name, drop_pending_updates_expr)`` where the
    expr is the unparsed source of the drop_pending_updates kwarg, or the
    sentinel ``"<MISSING>"`` when the kwarg is absent.

    The enclosing-function name is the STRUCTURAL discriminator (Required
    Change 1/4): we distinguish the initial-connect bootstrap (inside
    ``connect``) from the recovery ladders (network-error / 409-conflict
    handlers, in other functions) by the AST parent function, never by line
    number or lexical proximity — so an edit above a call site cannot silently
    reclassify it. Missing kwarg is surfaced as ``"<MISSING>"`` rather than
    ``None`` so the oracle can decide its verdict explicitly (a bare
    ``start_polling()`` relies on PTB's ``drop_pending_updates=None`` default,
    which is falsy/preserves today but is an implicit contract we refuse to
    depend on)."""
    src = Path(inspect.getfile(tg_adapter)).read_text()
    tree = ast.parse(src)

    # Map each Call node to its nearest enclosing FunctionDef/AsyncFunctionDef.
    enclosing = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(fn):
                if isinstance(child, ast.Call):
                    # innermost wins: nested functions are walked later, so only
                    # overwrite when this fn is a descendant (smaller span).
                    prev = enclosing.get(id(child))
                    if prev is None or (
                        fn.lineno >= prev.lineno
                        and getattr(fn, "end_lineno", fn.lineno)
                        <= getattr(prev, "end_lineno", prev.lineno)
                    ):
                        enclosing[id(child)] = fn

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "start_polling"):
            continue
        dpu = "<MISSING>"
        for kw in node.keywords:
            if kw.arg == "drop_pending_updates":
                dpu = ast.unparse(kw.value)
        fn = enclosing.get(id(node))
        fn_name = fn.name if fn is not None else "<module>"
        results.append((node.lineno, fn_name, dpu))
    return results


def test_all_start_polling_sites_preserve_queue_or_gate_on_reconnect():
    """AC-3 / INV-2: every start_polling call site must either preserve the
    queue unconditionally (drop_pending_updates=False) or gate on the cold-boot
    signal (drop_pending_updates=not is_reconnect). Explicit verdict on a
    MISSING kwarg: it FAILS — we require the argument be spelled out so no
    refactor can lean on PTB's implicit default (Required Change 1)."""
    calls = _start_polling_calls_in_adapter()
    assert calls, "expected at least one start_polling call site in adapter.py"

    allowed = {"False", "not is_reconnect"}
    offenders = [
        (lineno, fn, expr) for lineno, fn, expr in calls if expr not in allowed
    ]
    assert not offenders, (
        "every start_polling must preserve the Telegram backfill queue: use "
        "drop_pending_updates=False (recovery ladders) or =not is_reconnect "
        "(initial connect); a MISSING kwarg is rejected on purpose. Offending "
        "call sites (lineno, enclosing_fn, drop_pending_updates): "
        f"{offenders}"
    )


def test_recovery_ladders_preserve_queue_unconditionally():
    """AC-3 sharper / INV-2: the recovery ladders (every start_polling NOT in
    the initial-connect ``connect`` function) must be UNCONDITIONAL False — a
    recovery poll only fires after the bot was already up, so its queue is
    always genuine backlog, never a cold-boot backlog to discard. Only calls
    structurally inside ``connect`` may gate on is_reconnect.

    Discrimination is by enclosing function name (AST), not line number, so it
    is not a change-detector: moving code around cannot flip a ladder's class
    unless it actually changes which function it lives in."""
    calls = _start_polling_calls_in_adapter()

    recovery = [
        (ln, fn, e) for ln, fn, e in calls if fn != "connect"
    ]
    bootstrap = [
        (ln, fn, e) for ln, fn, e in calls if fn == "connect"
    ]

    assert recovery, (
        "expected recovery-ladder start_polling calls outside connect(); found "
        f"none (all calls: {calls})"
    )
    bad_recovery = [(ln, fn, e) for ln, fn, e in recovery if e != "False"]
    assert not bad_recovery, (
        "recovery ladders must preserve the queue unconditionally "
        "(drop_pending_updates=False); a recovery poll that gates on "
        "is_reconnect could drop backlog after an outage. Offenders "
        f"(lineno, enclosing_fn, drop_pending_updates): {bad_recovery}"
    )
    # The bootstrap inside connect() is the only place a cold-boot gate is legal.
    bad_bootstrap = [
        (ln, fn, e) for ln, fn, e in bootstrap
        if e not in {"not is_reconnect", "False"}
    ]
    assert not bad_bootstrap, (
        "the connect() bootstrap must gate on cold-boot (not is_reconnect) or "
        f"preserve (False); offenders: {bad_bootstrap}"
    )


# ── INV-3: graceful disconnect drains before shutdown ─────────────────────


@pytest.mark.asyncio
async def test_disconnect_drains_before_shutdown(monkeypatch):
    """AC-4 / INV-3 (ordering half): disconnect() must stop the updater, then
    stop the app, then shutdown, in that order. This test proves the ORDERING
    CONTRACT only — because app.stop is a mock here, it does NOT exercise the
    real update_queue.join() drain. That the drain actually happens inside
    Application.stop() is a separate PTB-provenance assertion
    (test_ptb_application_stop_drains_queue below), split out per Required
    Change 3 so this test doesn't quietly claim join coverage it lacks.

    Reordering (e.g. shutdown before stop) would drop queued-but-unprocessed
    updates on a graceful restart, so the order itself is a real invariant."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    order = []
    updater = SimpleNamespace(
        stop=AsyncMock(side_effect=lambda: order.append("updater.stop")),
        running=True,
    )
    app = SimpleNamespace(
        updater=updater,
        stop=AsyncMock(side_effect=lambda: order.append("app.stop")),
        shutdown=AsyncMock(side_effect=lambda: order.append("app.shutdown")),
        running=True,
    )
    adapter._app = app
    adapter._bot = SimpleNamespace()
    # Neutralize side helpers disconnect() calls so we isolate the drain order.
    adapter._polling_heartbeat_task = None
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    monkeypatch.setattr(adapter, "_cancel_pending_delivery_tasks", AsyncMock())
    monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
    monkeypatch.setattr(adapter, "_mark_disconnected", MagicMock())

    await adapter.disconnect()

    assert order == ["updater.stop", "app.stop", "app.shutdown"], (
        "disconnect must drain in order updater.stop -> app.stop "
        f"(update_queue.join) -> app.shutdown; got {order}. app.stop is the "
        "step that blocks until every fetched update is handler-processed; "
        "calling shutdown before stop drops in-flight updates."
    )
    updater.stop.assert_awaited_once()
    app.stop.assert_awaited_once()
    app.shutdown.assert_awaited_once()


def test_ptb_application_stop_drains_queue():
    """AC-4 / INV-3 (effect half, Required Change 3): the ordering test above
    rests on the assumption that Application.stop() actually blocks on
    update_queue.join() — i.e. drains every fetched update through the handler
    before returning. Assert that against the INSTALLED PTB source, not a mock,
    so a PTB upgrade that changes drain semantics re-triggers this review
    rather than silently inverting the loss-safety conclusion.

    This is the pinned-provenance guard: if PTB ever stops draining in stop(),
    the ordering contract in disconnect() no longer guarantees no-loss and this
    test fails loudly.

    Note: this module installs a MagicMock for ``telegram`` (via
    ``_ensure_telegram_mock``) so the adapter imports without a real bot. That
    mock would defeat ``inspect.getsource``, so here we load the REAL installed
    PTB package straight from disk by file path, bypassing sys.modules. If PTB
    is genuinely not installed (no dist), skip — CI installs it and gates there.
    """
    import site
    import sysconfig

    # sys.modules['telegram'] is a MagicMock here, so importlib.util.find_spec
    # can't resolve the subpackage. Locate the real PTB source on disk by
    # scanning the interpreter's site-packages roots directly — independent of
    # the poisoned parent module.
    candidate_roots = []
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidate_roots.append(purelib)
    try:
        candidate_roots.extend(site.getsitepackages())
    except Exception:
        pass
    up = getattr(site, "getusersitepackages", None)
    if callable(up):
        try:
            candidate_roots.append(up())
        except Exception:
            pass
    # de-dup, preserve order
    seen = set()
    roots = [r for r in candidate_roots if r and not (r in seen or seen.add(r))]

    app_path = None
    for root in roots:
        cand = Path(root) / "telegram" / "ext" / "_application.py"
        if cand.is_file():
            app_path = cand
            break
    if app_path is None:
        pytest.skip(
            "real python-telegram-bot not found on disk in site-packages "
            f"(searched {roots})"
        )

    source = app_path.read_text()
    # Find the module-level ``Application`` class and its ``stop`` method. Iterate
    # only top-level statements (not ast.walk) so we can't match a nested/inner
    # class named Application, and stop at the first module-level match.
    tree = ast.parse(source)
    stop_fn = None
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Application":
            for item in cls.body:
                if isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)) and (
                    item.name == "stop"
                ):
                    stop_fn = item
                    break
            break
    assert stop_fn is not None, (
        f"could not find module-level Application.stop in the installed PTB "
        f"source at {app_path}"
    )
    stop_src = ast.get_source_segment(source, stop_fn) or ""
    assert "update_queue.join" in stop_src, (
        f"PTB Application.stop() at {app_path} no longer calls "
        "update_queue.join() — the Telegram graceful-restart no-loss guarantee "
        "rests on stop() draining the fetched-update queue before returning. "
        "Re-verify the drain path and the loss-parity conclusion before "
        "accepting this PTB version."
    )
