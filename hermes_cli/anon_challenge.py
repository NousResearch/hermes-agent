"""Nous free tier: the browser challenge the account service may put in front of a token exchange.

NAS can answer ``POST /api/anonymous/token`` with **428** ``challenge_required`` and a list of
challenges. Hermes implements exactly one primitive, ``browser-v1``: *open this portal URL, poll
until it clears, exchange again*. Everything the page does (bot detection, an in-browser proof of
work, an interactive fallback, its copy) belongs to the portal and changes without a Hermes release.

Who opens the URL depends on the surface:

* **desktop** (``HERMES_DESKTOP=1``): this process never opens anything. It publishes the challenge
  (``free_tier.challenge`` global event, and ``free_tier.status``'s ``challenge`` field for a client
  that connects later); the renderer hands it to the Electron main process, which loads it in a
  HIDDEN window and reveals that window only if the page asks for the human.
* **terminal**: print the URL, and open the system browser when there is a graphical one to open.

The wait runs OUTSIDE the auth-store and shared-store locks: the exchange that hit the 428 has
already unwound (the exception left the ``with`` blocks), so a slow page never stalls a sibling
profile or process. One challenge is worked at a time per process; a second thread that hits the
same 428 waits for the first and then simply exchanges again (NAS hands a retried mint the same
ticket, and a cleared credential a token).

Only a caller someone is waiting on waits for a challenge. A background reader (the keepalive tick,
a status paint) runs inside :func:`background_caller`: it still gets the challenge in front of a
desktop client (so the hidden window can clear it before anyone needs a token) but never blocks,
prints, or opens a browser.

An *optional* challenge (``required: false`` on a successful exchange: the service is measuring, not
enforcing) is only ever announced to a desktop client, never opened in a user's browser, and
nothing waits on it.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
import platform
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, TypeVar
from urllib.parse import urlparse

from hermes_cli.anon_auth import (
    ANON_CHALLENGE_REQUIRED, ANON_FAILURE_COPY, ANON_SIGNIN_REQUIRED, _anon_headers, _mint_memo_key)
from hermes_cli.auth_constants import AuthError, httpx

logger = logging.getLogger("hermes_cli.auth")

BROWSER_CAPABILITY = "browser-v1"
CHALLENGE_EVENT = "free_tier.challenge"

# How long one exchange attempt waits for its challenge. The page normally clears in seconds; this
# bounds the interactive case (a human solving a check) without parking a request for the ticket's
# whole ten minutes. A later attempt resumes the same ticket.
CHALLENGE_WAIT_SECONDS = 120.0
_POLL_MIN_SECONDS, _POLL_MAX_SECONDS = 1.0, 5.0
# ``expires_in`` is network input that ends up in timers on two runtimes: keep it finite and sane.
_EXPIRES_MIN_SECONDS, _EXPIRES_MAX_SECONDS, _EXPIRES_DEFAULT_SECONDS = 30.0, 900.0, 600.0
_MESSAGE_MAX_CHARS = 300
# After one attempt ran out of patience, the callers queued behind it fail fast for this long
# instead of each parking for another full wait (boot fires a burst of token reads).
_GAVE_UP_COOLDOWN_SECONDS = 30.0

CHALLENGE_COPY = "Nous needs to run a quick check before starting your free session."

T = TypeVar("T")


@dataclass(frozen=True)
class BrowserChallenge:
    url: str
    required: bool
    expires_in: float
    interval: float
    message: str
    attempt: int = 0

    def as_payload(self) -> dict[str, Any]:
        return {"type": "browser", "url": self.url, "required": self.required,
                "expires_in": int(self.expires_in), "message": self.message, "attempt": self.attempt}


class AnonChallengeRequired(AuthError):
    """The exchange needs a browser challenge first. Carries what :func:`run_with_challenge` needs
    to work it: the challenge, and the portal, credential and auth state (its ``tls`` block) to poll with."""

    def __init__(self, challenge: BrowserChallenge, *, portal_base_url: str, anon_token: str,
                 auth_state: dict[str, Any]) -> None:
        super().__init__(challenge.message, provider="nous", code=ANON_CHALLENGE_REQUIRED, retryable=True)
        self.challenge = challenge
        self.portal_base_url = portal_base_url
        self.anon_token = anon_token
        self.auth_state = auth_state


def signin_required_error(message: Any = None) -> AuthError:
    """The service will not serve this client without an account (or asked for something this
    version cannot do). Terminal for the process, like a closed gate."""
    return AuthError(server_message(message) or ANON_FAILURE_COPY[ANON_SIGNIN_REQUIRED], provider="nous",
                     code=ANON_SIGNIN_REQUIRED, retryable=False)


def server_message(value: Any) -> str:
    """User-facing copy the service sent, if it is plausibly that: a short single paragraph."""
    if not isinstance(value, str):
        return ""
    # Printable text only: this reaches a terminal, and a control character there is an escape
    # sequence the service (or whoever answered as it) gets to run.
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in value).split())
    return text if 0 < len(text) <= _MESSAGE_MAX_CHARS else ""


# --- What this client tells the service about itself -------------------------------------------------


def client_surface() -> str:
    """``gateway`` in the process that holds the messaging gateway's runtime lock (nobody is at its
    console), ``desktop`` for the backend the desktop app spawned, else ``cli``. ``HERMES_DESKTOP``
    alone is inherited by every terminal-pane shell, so desktop ownership needs the spawn credential."""
    status = sys.modules.get("gateway.status")
    if status is not None and status.owns_gateway_runtime_lock():
        return "gateway"
    from hermes_cli.process_identity import is_desktop_owned_backend
    return "desktop" if is_desktop_owned_backend() else "cli"


def client_info() -> dict[str, Any]:
    """The self-reported ``client`` block on a token exchange. It sorts honest clients (which
    surface, which challenge primitives) for the service's rules; it proves nothing, by design."""
    from hermes_cli.version_info import get_version_info
    return {"name": "hermes-agent", "version": get_version_info().base_version, "surface": client_surface(),
            "platform": sys.platform, "capabilities": [BROWSER_CAPABILITY]}


def user_agent() -> str:
    from hermes_cli.version_info import get_version_info
    return (f"hermes-agent/{get_version_info().base_version} "
            f"({client_surface()}; {sys.platform}; {platform.machine() or 'unknown'})")


# --- Parsing ------------------------------------------------------------------------------------------


def _same_origin(url: str, portal_base_url: str) -> bool:
    """A challenge URL is only ever opened on the portal the exchange itself went to: the response
    is network input, and "open this URL" must not become "open any URL"."""
    try:
        target, portal = urlparse(url), urlparse(portal_base_url)
    except ValueError:
        return False
    return (target.scheme in ("https", "http") and target.scheme == portal.scheme
            and bool(target.netloc) and target.netloc.lower() == portal.netloc.lower())


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value)


def parse_browser_challenge(payload: dict[str, Any], portal_base_url: str) -> Optional[BrowserChallenge]:
    """The first challenge in ``payload["challenges"]`` this client can run, or None."""
    entries = payload.get("challenges")
    if not isinstance(entries, list):
        return None
    message = server_message(payload.get("message")) or CHALLENGE_COPY
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "browser":
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.isprintable() or not _same_origin(url, portal_base_url):
            logger.info("Nous free tier: ignoring a challenge URL outside the portal origin")
            continue
        return BrowserChallenge(
            url=url, required=entry.get("required") is not False,
            expires_in=min(_EXPIRES_MAX_SECONDS, max(
                _EXPIRES_MIN_SECONDS, _number(entry.get("expires_in"), _EXPIRES_DEFAULT_SECONDS))),
            interval=min(_POLL_MAX_SECONDS, max(_POLL_MIN_SECONDS, _number(entry.get("interval"), 2.0))),
            message=message)
    return None


def challenge_error(payload: dict[str, Any], *, portal_base_url: str, anon_token: str,
                    auth_state: dict[str, Any]) -> AuthError:
    """The error for a 428 ``challenge_required``: a challenge to work, or (nothing offered that
    this version can run) the sign-in fallback."""
    challenge = parse_browser_challenge(payload, portal_base_url)
    if challenge is None:
        # The payload's ``message`` describes the challenge, so it is not reused for the fallback.
        return signin_required_error()
    return AnonChallengeRequired(challenge, portal_base_url=portal_base_url, anon_token=anon_token,
                                 auth_state=auth_state)


# --- Per-profile state: what a status read shows, and who is working it ---------------------------------


@dataclass
class _ProfileChallenge:
    """One profile's challenge state, behind ``_state_lock``. ``work_lock`` serialises waits PER
    PROFILE (a sibling profile's wait must not queue behind it)."""
    work_lock: threading.Lock = field(default_factory=threading.Lock)
    pending: Optional[BrowserChallenge] = None
    deadline: float = 0.0
    outcome: Optional[str] = None      # how the host's window ended, for ``pending``'s attempt
    gave_up_until: float = 0.0


_state_lock = threading.Lock()
_profiles: dict[str, _ProfileChallenge] = {}
_opened_urls: set[str] = set()      # challenge URLs whose browser tab was opened (one per ticket)

# Set by :func:`background_caller`: this caller has nobody waiting on it.
_background: contextvars.ContextVar[bool] = contextvars.ContextVar("anon_challenge_background", default=False)


@contextlib.contextmanager
def background_caller():
    """Mark the enclosed token reads as background work (a keepalive tick, a status paint): a
    challenge is announced to a desktop client but never waited on, printed, or opened."""
    token = _background.set(True)
    try:
        yield
    finally:
        _background.reset(token)


def _profile() -> _ProfileChallenge:
    """This profile's record. Callers hold ``_state_lock`` for anything but ``work_lock``."""
    with _state_lock:
        return _profiles.setdefault(_mint_memo_key(), _ProfileChallenge())


def _live(record: _ProfileChallenge) -> Optional[BrowserChallenge]:
    if record.pending and record.deadline <= time.monotonic():
        record.pending, record.outcome = None, None
    return record.pending


def pending_challenge() -> Optional[dict[str, Any]]:
    """The challenge this profile is waiting on, for ``free_tier.status`` (a client that connected
    after the event fired still learns it has a window to open). None once cleared or expired, or
    once the host said how its window ended (a window the user closed must not come back)."""
    record = _profile()
    with _state_lock:
        challenge = _live(record)
        if challenge is None or record.outcome not in (None, "timeout", "error"):
            return None
        return {**challenge.as_payload(), "expires_in": math.ceil(record.deadline - time.monotonic())}


def _clear_pending() -> None:
    record = _profile()
    with _state_lock:
        record.pending, record.outcome = None, None


def _record(challenge: BrowserChallenge, *, new_attempt: bool) -> BrowserChallenge:
    """Record *challenge* as the profile's pending one. A replay of the same ticket (a background
    read, a second 428) keeps the original deadline, attempt and host outcome; ``new_attempt`` (a
    foreground caller about to wait) bumps the attempt, which lets it reopen a window the user closed."""
    record = _profile()
    with _state_lock:
        previous = _live(record)
        deadline = time.monotonic() + challenge.expires_in
        if previous is None or previous.url != challenge.url:
            record.pending, record.deadline, record.outcome = replace(challenge, attempt=0), deadline, None
        else:
            attempt = previous.attempt + 1 if new_attempt else previous.attempt
            record.pending, record.deadline = replace(challenge, attempt=attempt), min(deadline, record.deadline)
            if new_attempt:
                record.outcome = None
        return record.pending


def record_host_outcome(url: str, attempt: int, outcome: str) -> bool:
    """A presentation hint: wakes a waiter to re-mint, never grants clearance locally. A late
    reply belongs to the attempt that opened the window, not to a newer retry."""
    record = _profile()
    with _state_lock:
        challenge = _live(record)
        if challenge is None or (challenge.url, challenge.attempt) != (url, attempt):
            return False
        record.outcome = outcome
        return True


def _host_outcome(challenge: BrowserChallenge) -> Optional[str]:
    record = _profile()
    with _state_lock:
        pending = record.pending
        matches = pending is not None and (pending.url, pending.attempt) == (challenge.url, challenge.attempt)
        return record.outcome if matches else None


def _give_up() -> None:
    record = _profile()
    with _state_lock:
        record.gave_up_until = time.monotonic() + _GAVE_UP_COOLDOWN_SECONDS


def _gave_up_recently() -> bool:
    record = _profile()
    with _state_lock:
        return time.monotonic() < record.gave_up_until


def reset_for_tests() -> None:
    with _state_lock:
        _profiles.clear()
        _opened_urls.clear()


# --- Presenting ---------------------------------------------------------------------------------------


def _announce(challenge: BrowserChallenge) -> bool:
    """Broadcast to connected desktop clients. False when this process is not a gateway at all
    (``HERMES_DESKTOP`` is an env var, and a CLI the desktop spawned inherits it): importing the
    gateway here just to find nobody listening would also write an event frame to stdout."""
    server = sys.modules.get("tui_gateway.server")
    if server is None:
        return False
    try:
        server._broadcast_global_event(CHALLENGE_EVENT, challenge.as_payload())
    except (OSError, ValueError) as exc:
        # The stdio channel re-raises host I/O errors (ENOSPC, a locale that can't encode the
        # server's copy); the caller falls back to the terminal instead of losing the link.
        logger.debug("%s not broadcast: %s", CHALLENGE_EVENT, exc)
        return False
    return True


def _present_in_terminal(challenge: BrowserChallenge) -> None:
    from hermes_cli.auth_device_flow import _can_open_graphical_browser, _is_remote_session
    opened = False
    # One tab per ticket, however many attempts resume it: a user who has not got to it yet is
    # not helped by a second copy. A gateway has no console of its own: it logs the link only.
    already_open = challenge.url in _opened_urls
    if (not already_open and client_surface() != "gateway"
            and not _is_remote_session() and _can_open_graphical_browser()):
        import webbrowser
        try:
            opened = bool(webbrowser.open(challenge.url))
        except (webbrowser.Error, OSError) as exc:
            logger.debug("could not open the challenge in a browser: %s", exc)
        if opened:
            _opened_urls.add(challenge.url)
    print(f"\n{challenge.message}", file=sys.stderr)
    print(f"  Open: {challenge.url}", file=sys.stderr)
    print("  (Opened in your browser.)\n" if opened or already_open
          else "  (Open that link in any browser; this continues on its own.)\n", file=sys.stderr)


def present(challenge: BrowserChallenge) -> None:
    """Get the URL in front of something that can load it.

    The desktop backend hands every challenge to its hidden window. The stdio TUI gateway (whose
    stderr the TUI keeps as a log, never shows) hands its client a required one a foreground caller
    is waiting on: a background read there must not pop a browser any more than in a terminal."""
    if client_surface() == "desktop" and _announce(challenge):
        return
    if not challenge.required or _background.get():
        return
    server = sys.modules.get("tui_gateway.server")
    if server is not None and server._stdio_is_rpc_channel and _announce(challenge):
        return
    _present_in_terminal(challenge)


# --- Working a challenge ----------------------------------------------------------------------------


def _poll_status(client: httpx.Client, portal_base_url: str, anon_token: str) -> str:
    """One status read. Anything that is not a clear ``pending`` / ``needs_interaction`` ends the
    wait: the exchange that follows is the authority on whether the credential is cleared. (A
    passed challenge reads ``none``: passing detaches the ticket.)"""
    try:
        response = client.post(f"{portal_base_url.rstrip('/')}/api/anonymous/challenge/status",
                               headers=_anon_headers(), json={"token": anon_token})
        if 400 <= response.status_code < 500 and response.status_code != 429:
            # The service will not answer this poll (credential reaped, surface off): waiting cannot
            # change that. End the wait; the exchange's own verdict says what happened.
            return "none"
        body = response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("challenge status poll failed: %s", exc)
        body = None
    status = body.get("status") if isinstance(body, dict) else None
    # Only a 200 that names a status is a verdict. A 429, a 5xx or a dropped connection is a
    # blip: keep waiting (the deadline bounds it) rather than abandon a check the user is on.
    return status if isinstance(status, str) else "pending"


_WAITING = ("pending", "needs_interaction")
_sleep = time.sleep     # seam for tests (same idiom as ``free_tier_bootstrap._sleep``)


def wait_for_challenge(exc: AnonChallengeRequired) -> bool:
    """Present *exc*'s challenge and poll until status or a host hint ends the wait.
    True = a result arrived; False = patience ran out. Re-mint is authoritative in either case.

    Called under the profile's work lock. A caller that queued behind another finds the ticket
    already settled on its first status read and returns without presenting anything."""
    from hermes_cli.auth import _resolve_verify
    from hermes_cli.auth_nous import _nous_http_client
    if _gave_up_recently():
        return False
    challenge = exc.challenge
    deadline = time.monotonic() + min(CHALLENGE_WAIT_SECONDS, challenge.expires_in)
    # Poll under the trust the credential was minted with: a guest's ``tls`` block is written from
    # the mint's own verify, so this is the context the 428 exchange just used, not a narrowing.
    verify = _resolve_verify(auth_state=exc.auth_state)
    told_interactive = False
    with _nous_http_client(10.0, verify) as client:
        if _poll_status(client, exc.portal_base_url, exc.anon_token) not in _WAITING:
            return True
        challenge = exc.challenge = _record(challenge, new_attempt=True)
        present(challenge)
        while time.monotonic() < deadline:
            if _host_outcome(challenge) is not None:
                return True
            _sleep(challenge.interval)
            status = _poll_status(client, exc.portal_base_url, exc.anon_token)
            if status not in _WAITING:
                return True
            if status == "needs_interaction" and not told_interactive:
                told_interactive = True
                if client_surface() != "desktop":
                    print("  Finish the quick check in your browser to continue.", file=sys.stderr)
    _give_up()
    return False


def _still_pending(cause: AnonChallengeRequired) -> AuthError:
    message = ("The quick check couldn't finish. Try again." if _host_outcome(cause.challenge)
               else ANON_FAILURE_COPY[ANON_CHALLENGE_REQUIRED])
    error = AuthError(message, provider="nous", code=ANON_CHALLENGE_REQUIRED, retryable=True)
    error.__cause__ = cause
    return error


def run_with_challenge(exchange: Callable[[], T]) -> T:
    """Run *exchange*; if the service asks for a browser challenge, work it (outside every lock:
    the exception has already unwound them) and run *exchange* once more.

    A background caller (:func:`background_caller`) never waits: the challenge is announced so a
    desktop client can start clearing it, and the retryable error goes straight back.

    A second ``challenge_required`` is not looped on: it surfaces as a retryable error whose copy
    says what the user can do, the callers queued behind it fail fast for a while, and the next
    attempt resumes the same ticket."""
    try:
        return exchange()
    except AnonChallengeRequired as exc:
        first = exc
    # A messaging gateway has nobody at its console to clear a check, and its token reads can run
    # on the event loop: it never waits either.
    if _background.get() or client_surface() == "gateway":
        present(_record(first.challenge, new_attempt=False))
        raise _still_pending(first)
    with _profile().work_lock:
        wait_for_challenge(first)
    # Even a timed-out status poll can lag a committed clearance. Mint is
    # authoritative and gets one final attempt before we report a pending check.
    try:
        return exchange()
    except AnonChallengeRequired as again:
        # Whatever ended the wait (status, a host hint, patience), a second 428 starts the fail-fast
        # cooldown; the attempt after it opens a fresh window (``new_attempt``).
        again.challenge = _record(again.challenge, new_attempt=False)
        _give_up()
        raise _still_pending(again)
    except AuthError as error:
        if not error.retryable:
            _clear_pending()
        raise


def note_optional_challenges(payload: dict[str, Any], portal_base_url: str) -> None:
    """A successful exchange may advertise a challenge nobody has to pass (the service is measuring
    before it enforces). Only a desktop client runs it, hidden; a terminal never opens a browser
    for something optional. Fire and forget."""
    challenge = parse_browser_challenge(payload, portal_base_url)
    if challenge is None:
        _clear_pending()
    elif not challenge.required and client_surface() == "desktop":
        _announce(_record(challenge, new_attempt=False))
