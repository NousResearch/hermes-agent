"""The one Nous sign-in: the plain device-code login every surface renders, plus its states and copy."""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, ContextManager, Iterator, Optional

from hermes_cli.auth_constants import httpx


UPGRADE_START = "Sign in with a Nous account to unlock more models and tools."
UPGRADE_ALREADY_SIGNED_IN = "Already signed in."
UPGRADE_DO_NOT_SHARE = "Do not share this code."
UPGRADE_TIMED_OUT = "That sign-in link has expired. Start again whenever you're ready."
UPGRADE_NOT_COMPLETED = "Sign-in didn't finish. Try again whenever you're ready."
# The account service was busy or unreachable mid sign-in (an ``anon_*`` code from
# ``anon_auth.classify_mint_exception``): nothing was saved, so the copy reassures before the way forward.
UPGRADE_SERVICE_BUSY = ("Signing in couldn't finish because the Nous service is busy. "
                        "Try again in {wait}. Your session is still here in the meantime.")
UPGRADE_SERVICE_UNREACHABLE = ("The Nous service couldn't be reached to finish signing you in. "
                               "Check your internet connection and try again. Your session is still here.")

UPGRADE_NO_DEFAULT_TERMINAL = "No default model is set yet; run `hermes model` to pick one."
UPGRADE_NO_DEFAULT_CHAT = "No default model is set yet; run /model to pick one."
UPGRADE_WAITING = "Waiting for sign-in..."
UPGRADE_WAITING_UP_TO = "Waiting for sign-in, up to {minutes}."
UPGRADE_CANCELLED = "\nSign-in cancelled."
LOGIN_COMMAND = "/login"
LOGIN_STARTING = "Starting sign-in..."
LOGIN_DM_ONLY = "Sign in from a direct message with Hermes."
LOGIN_BUSY_ELSEWHERE = "Another sign-in is already running on this Hermes. Try again in a few minutes."
LOGIN_NOT_ALLOWED = "Only an operator of this Hermes can sign it in."
# The card form (a surface with its own sign-in button) and the chat form (names /login).
FREE_TIER_RATE_LIMIT_CARD = (
    "You've used up the allowance for chatting without signing in. It refreshes in {reset}. "
    "Sign in for a bigger allowance, it's free.")
FREE_TIER_RATE_LIMIT_CHAT = FREE_TIER_RATE_LIMIT_CARD + " To sign in: /login."


def format_wait_line(expires_in: int) -> str:
    """The honest "up to N minutes" line; N = ceil(expires_in / 60), singular at 1."""
    n = max(1, math.ceil(max(0, int(expires_in)) / 60))
    return UPGRADE_WAITING_UP_TO.format(minutes=f"{n} minute" if n == 1 else f"{n} minutes")


# --- Sign-in states -----------------------------------------------------------------------------
#
# One sign-in composition (:func:`run_sign_in`) yields these; every surface is a renderer over them.
# Each state carries its own user copy, so no renderer ever maps a reason to a string: ``.copy`` is
# the in-chat form (never a raw exception, a URL or a ``hermes`` verb) and ``.copy_terminal`` the
# form a top-level terminal command prints.


@dataclass(frozen=True)
class SignInState:
    """One step of a sign-in. Every state carries its own user copy; renderers never map."""

    kind: ClassVar[str] = ""
    terminal: ClassVar[bool] = True     # False only for Code and Waiting
    ok: ClassVar[bool] = False          # exit-code 0 / "this ended well"
    precondition: ClassVar[bool] = False  # True only for AlreadySignedIn

    @property
    def copy(self) -> str:
        """The in-chat form."""
        return ""

    @property
    def copy_terminal(self) -> str:
        """The terminal form; differs for Completed and Failed."""
        return self.copy


@dataclass(frozen=True)
class Code(SignInState):
    """The device page link and the sign-in code, plus the clock the caller may show."""

    link: str
    code: str
    expires_in: int
    interval: int
    kind: ClassVar[str] = "code"
    terminal: ClassVar[bool] = False

    @property
    def copy(self) -> str:
        return UPGRADE_DO_NOT_SHARE

    @property
    def copy_with_wait(self) -> str:
        """The do-not-share line plus the honest "up to N minutes"; composed here, never in a renderer."""
        return f"{UPGRADE_DO_NOT_SHARE} {format_wait_line(self.expires_in)}"


@dataclass(frozen=True)
class Waiting(SignInState):
    kind: ClassVar[str] = "waiting"
    terminal: ClassVar[bool] = False

    @property
    def copy(self) -> str:
        return UPGRADE_WAITING


@dataclass(frozen=True)
class Completed(SignInState):
    """``handoff`` holds the guest handoff's result lines (what moved, what to reconnect), empty
    when the sign-in replaced no free tier. ``handoff_status`` is the latest handoff record it ran,
    in ``guest_handoff.wire_status`` form, for a surface that renders the result itself."""

    email: str = ""
    model: str = ""
    model_changed: bool = False
    handoff: tuple[str, ...] = ()
    handoff_status: Optional[dict[str, Any]] = None
    kind: ClassVar[str] = "completed"
    ok: ClassVar[bool] = True

    def _lines(self, no_default: str) -> str:
        lines = [f"Signed in as {self.email}." if self.email else "Signed in.", *self.handoff]
        if self.model_changed:
            lines.append(f"Default model is now {self.model}." if self.model else no_default)
        return "\n".join(lines)

    @property
    def copy(self) -> str:
        return self._lines(UPGRADE_NO_DEFAULT_CHAT)

    @property
    def copy_terminal(self) -> str:
        return self._lines(UPGRADE_NO_DEFAULT_TERMINAL)


@dataclass(frozen=True)
class Cancelled(SignInState):
    """Stopped from outside before anything was saved: the user's cancel, a newer /login, a shutdown."""

    kind: ClassVar[str] = "cancelled"

    @property
    def copy(self) -> str:
        return UPGRADE_CANCELLED.strip()


@dataclass(frozen=True)
class TimedOut(SignInState):
    #: The enriched device-auth guidance, for a surface with room for it. Never shown in a chat.
    detail: str = ""
    kind: ClassVar[str] = "timed_out"

    @property
    def copy(self) -> str:
        return UPGRADE_TIMED_OUT


@dataclass(frozen=True)
class Failed(SignInState):
    """``reason``: an ``anon_*`` code naming the service's verdict (busy, unreachable...), or "" for
    anything unnamed. ``retry_after``: the wait the service asked for, in seconds (0 when it named
    none). ``retryable``: whether a later attempt can succeed, so a renderer knows to offer "try
    again" and when."""

    reason: str = ""
    detail: str = ""
    retry_after: float = 0.0
    kind: ClassVar[str] = "failed"

    @property
    def retryable(self) -> bool:
        from hermes_cli import anon_auth as _core
        return bool(self.reason.startswith("anon_") and self.reason not in _core.ANON_TERMINAL_CODES)

    @property
    def copy(self) -> str:
        from hermes_cli import anon_auth as _core
        if self.reason in _core.ANON_UNREACHABLE_CODES:
            return UPGRADE_SERVICE_UNREACHABLE
        if self.reason in (_core.ANON_RATE_LIMITED, _core.ANON_GATE_PAUSED):
            return UPGRADE_SERVICE_BUSY.format(wait=_core.friendly_wait(self.retry_after or 60))
        if self.reason in _core.ANON_FAILURE_COPY:
            return _core.anon_failure_copy(self.reason, retry_after=self.retry_after)
        return UPGRADE_NOT_COMPLETED

    @property
    def copy_terminal(self) -> str:
        if self.reason and self.copy != UPGRADE_NOT_COMPLETED:
            return self.copy
        return f"Sign-in failed: {self.detail}" if self.detail else UPGRADE_NOT_COMPLETED


@dataclass(frozen=True)
class AlreadySignedIn(SignInState):
    kind: ClassVar[str] = "already_signed_in"
    ok: ClassVar[bool] = True
    precondition: ClassVar[bool] = True

    @property
    def copy(self) -> str:
        return UPGRADE_ALREADY_SIGNED_IN


def _failed_from_exception(exc: BaseException) -> Failed:
    """A ``Failed`` that keeps the service's own verdict: the ``anon_*`` code and wait hint an
    ``AuthError`` carries, or the wire's shape for a transport error. The raw detail never reaches a
    chat; ``copy_terminal`` may show it when nothing better is known."""
    from hermes_cli import anon_auth as _core
    err = _core.classify_mint_exception(exc)
    reason = str(err.code or "")
    if reason == _core.ANON_SERVER_ERROR and not isinstance(exc, _core.AuthError):
        # An unnamed local failure (a bad CA bundle, a lock timeout): keep the generic copy and its
        # terminal detail rather than blaming the service.
        return Failed(reason="", detail=str(exc))
    return Failed(reason=reason, detail=str(exc), retry_after=float(err.retry_after or 0.0))


def _default_persist_guard(is_cancelled: Callable[[], bool]) -> Callable[[], ContextManager[bool]]:
    """The persist guard used when a surface brings none: proceed unless the attempt was stopped."""
    @contextlib.contextmanager
    def _guard():
        yield not is_cancelled()
    return _guard


def _account_email(account_state: dict[str, Any], fallback: Optional[str]) -> str:
    """The signed-in account's email, read from its own access token."""
    from hermes_cli.auth_constants import _decode_jwt_claims
    email = _decode_jwt_claims(account_state.get("access_token")).get("email")
    return str(email or fallback or "").strip()


def _handoff_lines(results: list[dict[str, Any]]) -> tuple[str, ...]:
    from hermes_cli.guest_handoff import result_lines
    return tuple(line for view in results for line in result_lines(view))


def _handoff_status(results: list[dict[str, Any]], replaced: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The handoff a completed sign-in reports (``guest_handoff.wire_status`` form): the latest record its
    own run attempted, else the replaced free tier's saved record (another process held it), else
    None when it replaced no free tier. Read under the scope."""
    from hermes_cli import anon_auth as _core
    from hermes_cli.guest_handoff import record_status, wire_status
    if results:
        return wire_status(results[-1])
    anon_token = replaced.get("anon_token") if _core.is_guest_state(replaced) else None
    status = record_status(anon_token) if isinstance(anon_token, str) and anon_token else None
    return wire_status(status) if status else None


def run_sign_in(
    *,
    timeout_seconds: float = 15.0,
    cancelled: Optional[Callable[[], bool]] = None,
    replace_account: bool = False,
    persist_guard: Optional[Callable[[], ContextManager[bool]]] = None,
    scope: Optional[Callable[[], ContextManager[Any]]] = None,
    client_factory: Optional[Callable[[float, Any], ContextManager[httpx.Client]]] = None,
) -> Iterator[SignInState]:
    """Sign in with a Nous account: the plain device-code login. Yields :class:`SignInState`s.

    One composition behind every surface: device code, then a cancel-aware token poll on one
    deadline, then the save. The save is where the guest handoff hooks in (``guest_handoff``): a free
    tier the account replaces has its apps moved once the surface's lock is released, and the result
    rides :class:`Completed`. :func:`settle_after_upgrade` runs exactly once per completion. Works
    with or without a free tier on disk. It always ends by yielding exactly one state whose
    ``terminal`` is True: a persist or settle failure becomes ``Failed``, never an exception out of
    ``next()``.

    *cancelled* is polled before the wait, on a <= 1 s tick inside the token poll, and once more
    under *persist_guard* before the save; a cancel always wins and nothing is saved.
    *replace_account* lets a signed-in install sign in again (the desktop's "connect another
    account"); without it a signed-in install yields ``AlreadySignedIn`` before any network.
    *persist_guard* lets a surface make its own cancel check and the save atomic under its own lock.
    *scope* is entered only around the two non-network blocks (precondition, persist/settle), never
    across a ``yield``: ``run_in_executor`` does not carry contextvars, so the scope has to be
    entered inside the generator, on whichever thread is advancing it.
    *client_factory* is the HTTP client seam, ``client_factory(timeout_seconds, verify)``.
    """
    from hermes_cli import anon_auth as _core
    from hermes_cli.auth import PROVIDER_REGISTRY, _resolve_verify
    from hermes_cli.auth_device_flow import DeviceFlowCancelled, _request_device_code
    from hermes_cli.auth_error_copy import is_device_code_expired
    from hermes_cli.auth_nous import _nous_http_client, _nous_portal_base_url
    from hermes_cli.guest_handoff import drain

    is_cancelled = cancelled or (lambda: False)
    open_scope = scope or contextlib.nullcontext

    # Yielded outside the scope, because a scope must never be held across a ``yield``.
    state: Optional[dict[str, Any]] = None
    try:
        with open_scope():
            state = _core.current_nous_state()
            # Inside the scope: the operator's Portal override is a profile setting, and it wins over
            # a saved URL the same way it does for every other Nous call.
            portal = _nous_portal_base_url(state or {})
    except Exception as exc:  # health: allow BLE001 -- the generator's contract is one terminal state, never a raise
        yield Failed(reason="", detail=str(exc))
        return
    if state and not _core.is_guest_state(state) and not replace_account:
        yield AlreadySignedIn()
        return

    try:
        pconfig = PROVIDER_REGISTRY["nous"]
        client_id, scope_str = pconfig.client_id, pconfig.scope
        # A malformed CA bundle raises here, before the wire: inside the try, so it lands on Failed.
        verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=None)
        open_client = client_factory or _nous_http_client
        with open_client(timeout_seconds, verify) as client:
            device = _request_device_code(client, portal, client_id, scope_str)
            expires_in = int(device["expires_in"])
            interval = int(device.get("interval") or 5)
            deadline = time.monotonic() + max(1, expires_in)
            yield Code(link=str(device["verification_uri_complete"]), code=str(device["user_code"]),
                       expires_in=expires_in, interval=interval)
            if is_cancelled():
                yield Cancelled()
                return
            yield Waiting()
            token_data = _core._poll_for_token(
                client=client, portal_base_url=portal, client_id=client_id,
                device_code=str(device["device_code"]), expires_in=max(1, int(deadline - time.monotonic())),
                poll_interval=interval, cancelled=is_cancelled)
        account_state = _core._account_state_from_token(
            token_data, portal_base_url=portal, client_id=client_id, scope=scope_str,
            verify=verify, timeout_seconds=timeout_seconds)
    except DeviceFlowCancelled:
        yield Cancelled()
        return
    except TimeoutError as exc:
        # The code ran out while the token endpoint kept failing (``_poll_device_token_generic`` sets
        # the cause): the service was unreachable, not a sign-in left unapproved.
        outage = exc.__cause__ if isinstance(exc.__cause__, ConnectionError) else None
        yield _failed_from_exception(outage) if outage else TimedOut(detail=str(exc))
        return
    except Exception as exc:
        # The token endpoint's own ``expired_token`` answer is the same timeout as running out the clock.
        yield TimedOut(detail=str(exc)) if is_device_code_expired(exc) else _failed_from_exception(exc)
        return

    guard = persist_guard or _default_persist_guard(is_cancelled)
    try:
        with open_scope():
            with guard() as may_persist:
                if may_persist:
                    _core.persist_nous_credentials(account_state)
            if may_persist:
                # The handoff runs once the surface's lock is released, never inside it.
                handoff = drain()
                settled = _core.settle_after_upgrade(account_state)
                handoff_status = _handoff_status(handoff, state)
    except Exception as exc:
        # persist_nous_credentials takes the auth-store lock, writes auth.json, takes the shared
        # store's file lock and reseeds the credential pool: a lock timeout or a read-only home
        # must not escape next(gen).
        yield Failed(reason="", detail=str(exc))
        return
    if not may_persist:                  # yielded outside the scope, never across it
        yield Cancelled()
        return

    yield Completed(
        email=_account_email(account_state, next((v["account_email"] for v in handoff if v["account_email"]), None)),
        model=str(settled.get("model") or ""),
        model_changed=bool(settled.get("changed")),
        handoff=_handoff_lines(handoff),
        handoff_status=handoff_status)
