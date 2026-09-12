"""Shared sign-in states, copy, and connector-preserving account promotion flow."""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, ContextManager, Dict, Iterator, Optional

from hermes_cli.auth_constants import httpx


UPGRADE_START = "Sign in with a Nous account to unlock more models and tools."
UPGRADE_ALREADY_SIGNED_IN = "Already signed in."
UPGRADE_DO_NOT_SHARE = "Do not share this code."
UPGRADE_TIMED_OUT = "Sign-in timed out; run the command again."
UPGRADE_NOT_COMPLETED = "Sign-in did not complete; run the command again."
UPGRADE_UNAVAILABLE = "The free tier is not available right now; run `hermes auth add nous` to sign in."
UPGRADE_REASON_COPY = {
    "user_declined": "Sign-in was rejected in the browser.",
    "superseded": "A newer sign-in code replaced this one.",
    "account_retired": "This free-tier identity was already used or expired; a new one is set up on the next start.",
    "account_not_anonymous": "This free-tier identity was already used or expired; a new one is set up on the next start.",
    "account_busy": "The transfer could not run; run the command again.",
}
_RETIRED_REASONS = frozenset({"account_retired", "account_not_anonymous"})

UPGRADE_NO_DEFAULT_TERMINAL = "No default model is set yet; run `hermes model` to pick one."
UPGRADE_NO_DEFAULT_CHAT = "No default model is set yet; run /model to pick one."
UPGRADE_WAITING = "Waiting for sign-in..."
UPGRADE_WAITING_UP_TO = "Waiting for sign-in, up to {minutes}."
UPGRADE_CANCELLED = "\nSign-in cancelled."
UPGRADE_UNAVAILABLE_CHAT = "The free tier is not available right now. Try /login again in a moment."
LOGIN_COMMAND = "/login"
LOGIN_STARTING = "Starting sign-in..."
LOGIN_DM_ONLY = "Sign in from a direct message with Hermes."
LOGIN_BUSY_ELSEWHERE = "Another sign-in is already running on this Hermes. Try again in a few minutes."
LOGIN_NOT_ALLOWED = "Only an operator of this Hermes can sign it in."
FREE_TIER_RATE_LIMIT_CHAT = (
    "Nous free tier rate limit active \u2014 resets in {reset}. "
    "Sign in with a Nous account for higher limits: /login.")


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
    precondition: ClassVar[bool] = False  # True only for AlreadySignedIn and Unavailable

    @property
    def copy(self) -> str:
        """The in-chat form."""
        return ""

    @property
    def copy_terminal(self) -> str:
        """The terminal form; differs for Completed, Failed and Unavailable."""
        return self.copy


@dataclass(frozen=True)
class Code(SignInState):
    """The consent link and the sign-in code, plus the clock the caller may show."""

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
    email: str = ""
    model: str = ""
    model_changed: bool = False
    kind: ClassVar[str] = "completed"
    ok: ClassVar[bool] = True

    def _lines(self, no_default: str) -> str:
        lines = [f"Signed in as {self.email}." if self.email else "Signed in."]
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
class Declined(SignInState):
    kind: ClassVar[str] = "declined"

    @property
    def copy(self) -> str:
        return UPGRADE_REASON_COPY["user_declined"]


@dataclass(frozen=True)
class Superseded(SignInState):
    """Stopped from outside before it could persist: a newer code, a cancel, or a shutdown."""

    kind: ClassVar[str] = "superseded"

    @property
    def copy(self) -> str:
        return UPGRADE_REASON_COPY["superseded"]


@dataclass(frozen=True)
class TimedOut(SignInState):
    #: The enriched device-auth guidance, for a surface with room for it. Never shown in a chat.
    detail: str = ""
    kind: ClassVar[str] = "timed_out"

    @property
    def copy(self) -> str:
        return UPGRADE_TIMED_OUT


@dataclass(frozen=True)
class Retired(SignInState):
    """The identity this sign-in started from is gone; run_sign_in already cleared it."""

    kind: ClassVar[str] = "retired"

    @property
    def copy(self) -> str:
        return UPGRADE_REASON_COPY["account_retired"]


@dataclass(frozen=True)
class Failed(SignInState):
    reason: str = ""
    detail: str = ""
    kind: ClassVar[str] = "failed"

    @property
    def copy(self) -> str:
        return UPGRADE_REASON_COPY.get(self.reason, UPGRADE_NOT_COMPLETED)

    @property
    def copy_terminal(self) -> str:
        ruled = UPGRADE_REASON_COPY.get(self.reason)
        if ruled:
            return ruled
        return f"Sign-in failed: {self.detail}" if self.detail else UPGRADE_NOT_COMPLETED


@dataclass(frozen=True)
class AlreadySignedIn(SignInState):
    kind: ClassVar[str] = "already_signed_in"
    ok: ClassVar[bool] = True
    precondition: ClassVar[bool] = True

    @property
    def copy(self) -> str:
        return UPGRADE_ALREADY_SIGNED_IN


@dataclass(frozen=True)
class Unavailable(SignInState):
    detail: str = ""
    kind: ClassVar[str] = "unavailable"
    precondition: ClassVar[bool] = True

    @property
    def copy(self) -> str:
        return UPGRADE_UNAVAILABLE_CHAT

    @property
    def copy_terminal(self) -> str:
        return f"{UPGRADE_UNAVAILABLE} ({self.detail})" if self.detail else UPGRADE_UNAVAILABLE


def _outcome_state(outcome: Dict[str, Any], anon_token: str) -> SignInState:
    """The one reason -> state mapping in the tree, for a promotion that did not complete.

    A retiring outcome clears the dead identity here, pinned to the token this attempt started
    from, so a losing attempt can never remove a newer one.
    """
    from hermes_cli import anon_auth as _core

    status = str(outcome.get("status") or "unknown")
    reason = str(outcome.get("reason") or "")
    if status == "timeout":
        return TimedOut()
    if reason in _RETIRED_REASONS:
        _core.clear_dead_guest("retired", dead_token=anon_token or None)
        return Retired()
    if reason == "user_declined":
        return Declined()
    if reason == "superseded":
        return Superseded()
    return Failed(reason=reason)


def _default_persist_guard(is_cancelled: Callable[[], bool]) -> Callable[[], ContextManager[bool]]:
    """The persist guard used when a surface brings none: proceed unless the attempt was stopped."""
    @contextlib.contextmanager
    def _guard():
        yield not is_cancelled()
    return _guard


def run_sign_in(
    *,
    timeout_seconds: float = 15.0,
    cancelled: Optional[Callable[[], bool]] = None,
    cancel_wins_after_promotion: bool = True,
    persist_guard: Optional[Callable[[], ContextManager[bool]]] = None,
    scope: Optional[Callable[[], ContextManager[Any]]] = None,
    client_factory: Optional[Callable[[float, Any], ContextManager[httpx.Client]]] = None,
) -> Iterator[SignInState]:
    """Sign the free tier into a Nous account, keeping its connectors. Yields :class:`SignInState`s.

    One composition behind every surface: it reads the current identity itself, mints one when there
    is none, registers the connector transfer, holds ONE absolute deadline across both waits,
    persists only after a completed transfer AND a token grant, and runs
    :func:`settle_after_upgrade` exactly once per completion. It always ends by yielding exactly one
    state whose ``terminal`` is True -- a persist or settle failure becomes ``Failed``, never an
    exception out of ``next()``.

    *cancelled* is polled between round trips and inside :func:`wait_for_promotion`.
    *cancel_wins_after_promotion* rules what a cancel means once the account service has already
    transferred the connectors: True (the desktop) aborts and persists nothing -- the install
    re-mints a free tier on next use; False (a chat that was superseded) finishes, because a
    transfer the user approved in the browser is irreversible and discarding it would leave the
    connectors moved with no account to reach them.
    *persist_guard* lets a surface make its own cancel check and the save atomic under its own lock.
    *scope* is entered only around the two non-network blocks (preconditions/mint, persist/settle),
    never across a ``yield``: ``run_in_executor`` does not carry contextvars, so the scope has to be
    entered inside the generator, on whichever thread is advancing it.
    *client_factory* is the HTTP client seam, ``client_factory(timeout_seconds, verify)``.
    """
    from hermes_cli import anon_auth as _core
    from hermes_cli.auth import PROVIDER_REGISTRY, _resolve_verify
    from hermes_cli.auth_device_flow import _request_device_code
    from hermes_cli.auth_nous import _nous_http_client

    is_cancelled = cancelled or (lambda: False)
    # Once the server says "completed" the transfer has happened; a cancel only undoes it where the
    # surface says it does.
    post_promotion_cancelled = is_cancelled if cancel_wins_after_promotion else (lambda: False)
    open_scope = scope or contextlib.nullcontext

    # Preconditions run inside the scope; the state they produce is yielded outside it, because a
    # scope must never be held across a ``yield``. A sign-in never creates the identity it signs in
    # from: with none on disk there is nothing to promote and the answer is ``Unavailable`` (the boot
    # bootstrap is the only creator, NS-845 Q1.2).
    precondition_state: Optional[SignInState] = None
    state: Optional[Dict[str, Any]] = None
    try:
        with open_scope():
            state = _core.current_nous_state()
            if state and not _core.is_guest_state(state):
                precondition_state = AlreadySignedIn()
            elif not state or not _core.guest_enabled():
                precondition_state = Unavailable()
    except Exception as exc:
        # An unreadable auth store means the same thing here: there is no free tier to sign in from.
        # It becomes the one precondition state, so nothing escapes ``next()``. KeyboardInterrupt
        # and GeneratorExit are not Exceptions: they still propagate.
        precondition_state = Unavailable(detail=str(exc))
    if precondition_state is not None:
        yield precondition_state
        return

    anon_token = str(state.get("anon_token") or "")
    portal = (state.get("portal_base_url") or _core._portal_base_url()).rstrip("/")

    outcome: Dict[str, Any] = {}
    account_state: Optional[Dict[str, Any]] = None
    try:
        pconfig = PROVIDER_REGISTRY["nous"]
        client_id, scope_str = pconfig.client_id, pconfig.scope
        # A malformed CA bundle raises here, before the wire: inside the try, so it lands on Failed.
        verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=None)
        open_client = client_factory or _nous_http_client
        with open_client(timeout_seconds, verify) as client:
            device = _request_device_code(client, portal, client_id, scope_str)
            intent = _core.register_promotion_intent(
                client, portal, anon_token, user_code=str(device["user_code"]),
                device_code=str(device["device_code"]))
            # The browser leg is the consent page for THIS sign-in (claim_url), not the generic
            # device page: it shows both identities and the button. Relative paths are
            # portal-relative.
            link = str(intent.get("claim_url") or "")
            if link.startswith("/"):
                link = f"{portal}{link}"
            link = link or str(device["verification_uri_complete"])
            expires_in = min(int(device["expires_in"]), int(intent.get("expires_in") or device["expires_in"]))
            interval = int(intent.get("interval") or device.get("interval") or 5)
            deadline = time.monotonic() + max(1, expires_in)
            yield Code(link=link, code=str(intent["claim_code"]), expires_in=expires_in, interval=interval)
            if is_cancelled():   # nothing is approved anywhere yet: a cancel always wins here
                yield Superseded()
                return
            yield Waiting()

            remaining = max(1, int(deadline - time.monotonic()))
            outcome = _core.wait_for_promotion(
                client, portal, str(intent["claim_code"]),
                expires_in=remaining, interval=interval, cancelled=is_cancelled)
            status = str(outcome.get("status") or "unknown")
            if status != "completed":
                # A cancel is authoritative for every non-completed outcome, including the
                # {"status": "cancelled"} the hook returns.
                if is_cancelled():
                    yield Superseded()
                    return
                # Scoped: a retiring outcome clears the identity out of this profile's auth store.
                with open_scope():
                    ended = _outcome_state(outcome, anon_token)
                yield ended
                return
            if post_promotion_cancelled():
                yield Superseded()
                return

            remaining = max(1, int(deadline - time.monotonic()))
            token_data = _core._poll_for_token(
                client=client, portal_base_url=portal, client_id=client_id,
                device_code=str(device["device_code"]), expires_in=remaining, poll_interval=interval)
        account_state = _core._account_state_from_token(
            token_data, portal_base_url=portal, client_id=client_id, scope=scope_str,
            verify=verify, timeout_seconds=timeout_seconds)
    except _core.AnonCredentialDead:
        # Best effort: the credential is provably dead at the account service, so the outcome is
        # Retired whatever the local write does. A clear that fails (locked or read-only store)
        # self-heals on the next rejection, and must not cost this run its terminal state.
        with contextlib.suppress(Exception):
            with open_scope():
                _core.clear_dead_guest("retired", dead_token=anon_token or None)
        yield Retired()
        return
    except TimeoutError as exc:
        yield TimedOut(detail=str(exc))
        return
    except Exception as exc:
        yield Failed(reason="", detail=str(exc))
        return

    try:
        if post_promotion_cancelled():
            yield Superseded()
            return
        guard = persist_guard or _default_persist_guard(post_promotion_cancelled)
        with open_scope():
            with guard() as may_persist:
                if may_persist:
                    _core.persist_nous_credentials(account_state)
            if may_persist:
                settled = _core.settle_after_upgrade(account_state)
    except Exception as exc:
        # persist_nous_credentials takes the auth-store lock, writes auth.json, takes the shared
        # store's file lock and reseeds the credential pool: a lock timeout or a read-only home
        # must not escape next(gen).
        yield Failed(reason="", detail=str(exc))
        return
    if not may_persist:                  # yielded outside the scope, never across it
        yield Superseded()
        return

    yield Completed(
        email=str(outcome.get("account_email") or "").strip(),
        model=str(settled.get("model") or ""),
        model_changed=bool(settled.get("changed")))
