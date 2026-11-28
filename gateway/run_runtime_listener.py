"""Runtime API listener startup, bounded Webapp address-wait retries, and private-listener fallback."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import errno
import logging
import time

# Log-record parity with gateway/run_runtime.py.
logger = logging.getLogger("gateway.run_runtime")

# A specific ``webapp.host`` (a LAN or tailnet address) may not be assigned to an interface yet when this
# gateway starts: Tailscale, Headscale or WireGuard hand the address to the interface a moment after their
# daemon reports ready, and a DHCP lease lands the same way. The bind then fails with EADDRNOTAVAIL, and
# the gateway keeps running with the Webapp off (see below) — nothing crashes, so no Restart= policy
# retries it and the address stays dark until someone restarts the gateway by hand. Wait, bounded, for the
# address to appear, then bind again. Only EADDRNOTAVAIL is waited out, so a busy port or a refused bind
# still reports at once.
WEBAPP_ADDRESS_WAIT_SECONDS = 20.0
_WEBAPP_ADDRESS_POLL_SECONDS = 0.5


async def _wait_for_address_retry(runner, delay: float) -> None:
    """Sleep up to *delay* seconds for the address to appear, waking at once on shutdown."""
    shutdown_event = getattr(runner, "_shutdown_event", None)
    if isinstance(shutdown_event, asyncio.Event):
        if shutdown_event.is_set():
            return
        with suppress(TimeoutError):
            await asyncio.wait_for(shutdown_event.wait(), timeout=delay)
        return
    await asyncio.sleep(delay)


async def _close_listener_if_stopping(run_api, runner, handle) -> bool:
    """Close a freshly started listener if shutdown began while it was binding; True when closed."""
    from gateway.run_common import startup_should_abort
    if not startup_should_abort(runner):
        return False
    stop_api = getattr(run_api, "stop_gateway_api", None)
    if callable(stop_api):
        await stop_api(handle)
    return True


async def _start_gateway_api_waiting_for_address(run_api, runner, *, host, port, webapp, wait_seconds):
    """Start the gateway's listener, waiting out a specific ``webapp.host`` an interface has not been given yet.

    A failed start leaves nothing behind (``start_gateway_api`` closes its listener and clears its
    runner), so retrying is safe. Only EADDRNOTAVAIL on an enabled Webapp with a specific
    non-loopback, non-wildcard address is retried: loopback/wildcard binds, a private listener,
    a busy port, a refused bind and the loopback companion's own failure (``ListenerAddressError``)
    are reported at once, as before. Returns ``None`` when shutdown or restart starts before the
    listener is published.
    """
    from gateway.run_common import startup_should_abort
    from gateway.webapp_listener import loopback_companion

    wait_for_address = wait_seconds > 0 and webapp is not None and bool(loopback_companion(host))
    deadline = time.monotonic() + wait_seconds
    warned = False

    async def _start():
        # Without the Webapp the listener is main's: the same call, a private ephemeral port.
        return await (run_api.start_gateway_api(runner, host=host, port=port, webapp=webapp)
                      if webapp is not None else run_api.start_gateway_api(runner))

    while True:
        if startup_should_abort(runner):
            return None
        try:
            handle = await _start()
        except run_api.ListenerAddressError:
            raise  # the loopback socket beside a specific address: its address does not appear later
        except OSError as exc:
            if startup_should_abort(runner):
                return None
            remaining = deadline - time.monotonic()
            if not wait_for_address or exc.errno != errno.EADDRNOTAVAIL or remaining <= 0:
                raise
            if not warned:
                logger.warning("Webapp: %s is not assigned to any interface yet; retrying for up to %.0fs",
                               host, remaining)
                warned = True
            await _wait_for_address_retry(runner, min(_WEBAPP_ADDRESS_POLL_SECONDS, remaining))
        else:
            if await _close_listener_if_stopping(run_api, runner, handle):
                return None
            return handle


def _webapp_unavailable_reason(exc: BaseException, *, host: str, port: int, run_api) -> str:
    """Format the operator-facing reason when Webapp startup falls back to the private listener."""
    from gateway.webapp_listener import BrowserCredentialError, loopback_companion

    if isinstance(exc, SystemExit):
        return str(exc.code).strip()  # the gate's own message names the fix
    if isinstance(exc, BrowserCredentialError):
        return str(exc)  # names the file and why it cannot be stored
    if isinstance(exc, OSError) and exc.filename:  # the browser credential's file, not the port
        return f"could not use {exc.filename} ({exc.strerror or exc})"
    cause = (exc.strerror or exc) if isinstance(exc, OSError) else exc
    # Only the specific address itself; a loopback socket that cannot bind (``::1`` with IPv6
    # loopback off) keeps the OS's words.
    if (loopback_companion(host) and isinstance(exc, OSError) and exc.errno == errno.EADDRNOTAVAIL
            and not isinstance(exc, run_api.ListenerAddressError)):
        cause = "the address is not assigned to any network interface on this machine"
    # A specific address's loopback socket names its own address (run_api.ListenerAddressError).
    address = getattr(exc, "address", None) or f"{host}:{port}"
    return f"could not listen on {address} ({cause})"


async def start_gateway_runtime_api(runner, *, address_wait: float = WEBAPP_ADDRESS_WAIT_SECONDS):
    from gateway import run_api, run_runtime
    from gateway.run_common import startup_should_abort
    from gateway.webapp_listener import (
        STARTING, bind_origin, browser_session_token, dial_origin, listener_plan,
        register_control_verbs,
    )
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    # The bootstrap registered these before its control socket answered; a runner started without it
    # (an embedded gateway) gets them here, before the listener's slow start.
    register_control_verbs(runner, home)
    host, port, webapp, unusable = listener_plan(home)
    # A webapp: block that asks for the Webapp but cannot be used is reported with its reason.
    runner.webapp_status = ({"state": "unavailable", "url": None, "reason": unusable} if unusable
                            else {"state": "off", "url": None, "reason": None} if webapp is None
                            else dict(STARTING))
    try:
        # Without the Webapp the listener is main's: the same call, a private ephemeral port.
        handle = await _start_gateway_api_waiting_for_address(
            run_api, runner, host=host, port=port, webapp=webapp, wait_seconds=address_wait)
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- Webapp listener bind/credential/auth-gate failures intentionally fall back to the private listener and record the reason in webapp_status
        if startup_should_abort(runner):
            runner.session_api = None
            return
        if webapp is None:
            raise
        # The Webapp's listener could not start (port taken, address unavailable, its browser credential
        # not storable in the home, or the dashboard's auth gate exiting on a wildcard bind with no
        # sign-in provider): native clients still need this gateway, so it keeps main's private
        # listener and reports why the Webapp is not served.
        reason = _webapp_unavailable_reason(exc, host=host, port=port, run_api=run_api)
        logger.error("Webapp unavailable: %s", reason)
        runner.webapp_status = {"state": "unavailable", "url": None, "reason": reason}
        # Open tabs keep sending their credential to whatever answers on the Webapp's port, and that
        # is not this gateway now. Replace it, so one captured there never opens this gateway; the
        # tabs sign in again once the Webapp is back.
        try:
            browser_session_token(home, rotate=True)
        except OSError as write_error:
            logger.warning("Could not replace the Webapp's browser credential: %s", write_error)
        fallback = await run_api.start_gateway_api(runner)
        if await _close_listener_if_stopping(run_api, runner, fallback):
            runner.session_api = None
            return
        runner.session_api = fallback
    else:
        if handle is None:
            runner.session_api = None
            return
        runner.session_api = handle
        if webapp is not None:
            runner.webapp_status = {"state": "serving", "url": dial_origin(host, port), "reason": None}
            if own := bind_origin(host, port):  # a specific address: the one a browser opens
                runner.webapp_status["address"] = own
    runner.session_runtime_descriptor['api_origin'] = runner.session_api.api_origin
    for authority in run_runtime._authorities(runner):
        await run_runtime._recover_profile_bot_mailbox(runner, authority)
