"""Per-turn access token: the signed-in user's OAuth token, resolved and scoped to one turn.

The dashboard login proves *who* is asking. A downstream resource server (an MCP server) needs a
token minted for *that user* to authorise the call on their behalf. This module carries that token
from the authentication layer to whichever tool actually makes the request, without ever placing
it in the model's context, the session, or a config file.

Three ideas live here, and the layering between them is the point — one module, one import:

``TurnToken``
    An immutable snapshot: the token, whose it is (``issuer``/``subject``), and when it dies. It
    is the value that crosses a thread boundary.
``remember`` / ``note_login``
    Which dashboard login drove a gateway session, and the token that login produced. A session
    maps to a ``(provider, issuer, subject)`` triple; the token is keyed by that triple, so two
    tabs of the *same* user share one entry while two different users never do.
``bind_turn_token`` / ``current_turn_token``
    A :class:`contextvars.ContextVar` binding for one turn. The turn runner binds it, and every
    tool worker started under that turn inherits it because the executor copies the context
    (``tools.thread_context.propagate_context_to_thread``). Contexts are copied, so concurrent
    turns in one process never see each other's token and a worker that outlives its turn keeps
    the snapshot it started with rather than reading a newer one.

A turn with no login (CLI, cron, a messaging platform) simply has no binding; consumers must
treat that as "cannot act as the user" and fail closed rather than falling back to a shared
credential.

Nothing here is persisted: the token lives in this process's memory only. A restart therefore
loses every entry, which :func:`backfill` repairs from the next request that presents a session
the provider still accepts — without it, a restart would demand a re-login the user does not owe.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# Refresh the turn token this many seconds before it expires: a downstream call that
# starts just under the wire would otherwise be rejected by the resource server mid-flight.
DEFAULT_EXPIRY_SKEW_SECONDS = 60

# Bounded so a long-lived gateway cannot accumulate entries forever. A dropped entry only costs
# one re-authentication (silent SSO), never a wrong credential.
MAX_STORE_ENTRIES = 256
MAX_SESSION_ENTRIES = 512


@dataclass(frozen=True)
class TurnToken:
    """One user's OAuth access token, bound for the duration of a turn.

    ``issuer``/``subject`` identify whose token this is (the OIDC ``iss``/``sub``), so a
    consumer can tell a re-login apart from the same user and can refuse a token minted for a
    different issuer. ``expires_at`` is unix seconds; ``0`` means "unknown", and consumers
    that require a known lifetime must treat that as expired.

    ``refresh_token`` is carried so a turn that starts with a dead access token can renew it
    itself (:func:`refresh_store_entry`) instead of refusing. It never leaves the process: the
    resource server only ever sees ``access_token``.

    Both tokens are excluded from ``repr``: this object travels through turn state, exception
    traces and debug logs, and a bearer in a log file is a leaked bearer (the provider cannot
    revoke a signed JWT).
    """

    access_token: str = field(repr=False)
    issuer: str = ""
    subject: str = ""
    expires_at: int = 0
    refresh_token: str = field(default="", repr=False)

    def is_fresh(self, *, skew_seconds: int = DEFAULT_EXPIRY_SKEW_SECONDS, now: Optional[float] = None) -> bool:
        """True when the token has a known expiry still comfortably in the future."""
        if not self.access_token or self.expires_at <= 0:
            return False
        return self.expires_at - skew_seconds > (time.time() if now is None else now)


# ---- store: identity triple -> turn token ---------------------------------
#
# Keyed by login, not by host or session: the dashboard collapses every login to the machine
# owner for *session ownership* (gateway.owner_identity), which is a deliberate choice so the
# Desktop and a browser tab resume the same conversations. That collapse is exactly what makes
# the *token* unusable if it is carried the same way, so the token is tracked separately.

_store_lock = threading.Lock()
#: (provider, issuer, subject) -> (TurnToken, updated_at)
_store: dict[tuple[str, str, str], tuple[TurnToken, float]] = {}


def identity_key(*, provider: str, issuer: str, subject: str) -> Optional[tuple[str, str, str]]:
    """Store key for a login. ``None`` when the triple cannot identify anyone.

    A missing ``subject`` is the dangerous case: it would collapse every anonymous login onto
    one key, so an entry can only be created for a non-empty subject.
    """
    subject = (subject or "").strip()
    if not subject:
        return None
    return ((provider or "").strip(), (issuer or "").strip(), subject)


def key_for_identity(identity: Any) -> Optional[tuple[str, str, str]]:
    """Store key for a WS/RPC ``{user_id, provider, issuer}`` identity dict.

    Unknown shapes answer ``None`` rather than a partial key, so a caller can never look up a
    credential with a guess.
    """
    if not isinstance(identity, dict):
        return None
    return identity_key(
        provider=str(identity.get("provider") or ""),
        issuer=str(identity.get("issuer") or ""),
        subject=str(identity.get("user_id") or ""))


def key_for_session(session: Any) -> Optional[tuple[str, str, str]]:
    """Store key for a dashboard-auth ``Session`` (or any object with the same fields)."""
    return identity_key(
        provider=str(getattr(session, "provider", "") or ""),
        issuer=str(getattr(session, "issuer", "") or ""),
        subject=str(getattr(session, "user_id", "") or ""))


def _as_int(value: Any) -> int:
    """Best-effort int for a provider-reported timestamp; ``0`` (unknown) on anything else."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _token_from_session(session: Any, key: tuple[str, str, str]) -> TurnToken:
    """The ``TurnToken`` a dashboard-auth session describes under ``key``.

    The one place a session's fields become a stored credential, so the login path and the
    renewal path cannot drift on which fields they copy. Issuer/subject come from ``key`` (the
    identity that was just resolved), not from the session, so a provider that echoes them
    slightly differently cannot land the token under another key.
    """
    return TurnToken(
        access_token=str(getattr(session, "access_token", "") or ""),
        issuer=key[1], subject=key[2],
        expires_at=_as_int(getattr(session, "expires_at", 0)),
        refresh_token=str(getattr(session, "refresh_token", "") or ""))


def remember(session: Any, *, now: Optional[float] = None) -> bool:
    """Record the access token carried by ``session``; True when an entry was written.

    Called for every login and every refresh. ``Session.access_token`` is the credential the
    dashboard itself re-verifies, and for this deployment's provider (Keycloak) it *is* the
    OAuth access token a resource server expects, so it is forwarded as-is. A session with no
    token *removes* any previous entry: a stale credential must not outlive the login that
    replaced it.

    Total by construction (no raising path): this runs inside the login/refresh handlers, and a
    malformed provider field must never turn a successful sign-in into a 500.
    """
    key = key_for_session(session)
    token = str(getattr(session, "access_token", "") or "")
    if key is None or not token:
        forget(key)
        return False
    with _store_lock:
        _store[key] = (_token_from_session(session, key),
                       time.time() if now is None else now)
        _prune_store_locked()
    return True


def backfill(session: Any, *, now: Optional[float] = None) -> bool:
    """Record ``session``'s token only when the store has no live entry for that login.

    The store is process memory, so a gateway restart empties it while the browser still
    holds a valid session cookie. Without refilling, the first turn after a restart would
    reach an MCP tool with nothing to forward and fail closed, forcing a re-login that the
    user does not owe — the cookie it just presented is proof enough. Callers are the
    request gate's admitted paths: a token it verified is a token this process may use.

    Never overwrites a live entry. A login or refresh that lands while a request is in flight
    owns the newer credential, and letting a still-valid older token replace a rotated one
    would be a downgrade. An entry that has aged out does not count as live, so this also
    recovers a store whose entries all expired.

    The caller's session often carries no refresh token — ``verify_session`` re-verifies a
    bearer and returns ``refresh_token=""`` by construction. Writing that as-is would erase a
    good refresh token from a stale-but-renewable entry and destroy the very thing
    :func:`refresh_store_entry` needs on the next idle return. So an incoming session without
    one keeps whatever the store already had.

    True means "the store now holds a usable token for this login". An expired session with no
    refresh token to inherit is recorded by nobody: pruning drops it immediately, so claiming
    success would misreport the state to every caller.
    """
    key = key_for_session(session)
    if key is None or not str(getattr(session, "access_token", "") or ""):
        return False
    if recall(key, now=now) is not None:
        return False
    entry = _token_from_session(session, key)
    with _store_lock:
        # A refresh token the incoming session does not carry (see the docstring) is taken
        # from the entry it replaces: the re-verified bearer has none, but the stale entry it
        # is about to overwrite may.
        previous = _store.get(key)
        if not entry.refresh_token and previous is not None:
            entry = dataclasses.replace(entry, refresh_token=previous[0].refresh_token)
        if not entry.is_fresh(now=now) and not entry.refresh_token:
            # Nothing usable to keep; pruning would drop it on this very call.
            return False
        _store[key] = (entry, time.time() if now is None else now)
        _prune_store_locked()
    return True


def recall(key: Optional[tuple[str, str, str]], *,
           skew_seconds: int = DEFAULT_EXPIRY_SKEW_SECONDS,
           now: Optional[float] = None) -> Optional[TurnToken]:
    """The live token for ``key``, or ``None`` when absent or too close to expiry.

    ``None`` is a verdict, not an error: the caller must not substitute anything for it.
    """
    if key is None:
        return None
    with _store_lock:
        entry = _store.get(key)
    if entry is None:
        return None
    token = entry[0]
    return token if token.is_fresh(skew_seconds=skew_seconds, now=now) else None


def forget(key: Optional[tuple[str, str, str]]) -> None:
    """Drop one entry (a refresh that came back without a token)."""
    if key is None:
        return
    with _store_lock:
        _store.pop(key, None)


def forget_token(access_token: str) -> bool:
    """Drop the entry holding ``access_token``; True when one was removed.

    The sign-out path has the token (from its cookie) but not the identity triple it is stored
    under, so it cannot address the entry by key. Removing it matters because an access token is
    a bearer: the provider cannot revoke a signed JWT, so this process's copy is the only thing
    that stops a turn outliving the sign-out from still acting as that user.
    """
    if not access_token:
        return False
    with _store_lock:
        keys = [k for k, (token, _) in _store.items() if token.access_token == access_token]
        for key in keys:
            del _store[key]
    return bool(keys)


def clear_store() -> None:
    """Test-only: drop every stored token."""
    with _store_lock:
        _store.clear()


def refresh_store_entry(key: Optional[tuple[str, str, str]], *,
                        refresher: Optional[Callable[[str, str], Any]] = None,
                        ) -> Optional[TurnToken]:
    """Renew ``key``'s access token with the login's refresh token; the new entry, or ``None``.

    This is what makes a long-idle browser work. A dashboard session refreshes only when a
    *request* arrives, and a user who leaves the tab alone for hours comes back to a dead access
    token with the very next turn — no request has happened yet, so nothing renewed it. Without
    this, that turn would be refused ("sign in again") even though its refresh token is still
    perfectly good.

    Called immediately before a turn binds its credential, so the renewal lands before any
    user-scoped MCP call. Synchronous and network-bound by nature (it drives the provider's
    refresh grant); the single-flight cache in ``hermes_cli.dashboard_auth.refresh_singleflight``
    means a burst of turns for one login still reaches the IdP once, which matters because a
    rotated refresh token replayed twice revokes the whole session.

    ``refresher`` is the injectable seam: ``(refresh_token, provider) -> Session | None``. The
    default drives the real provider registry, which only exists in the process that loaded the
    dashboard-auth plugins — the gateway, which is also where turns run.

    Never raises. Every failure — no refresh token, an expired one, an unreachable IdP, no
    registered provider — answers ``None`` and leaves the store untouched, so the caller falls
    through to the existing refusal. A failed renewal must never become a traceback on the turn
    path.
    """
    if key is None:
        return None
    with _store_lock:
        entry = _store.get(key)
    if entry is not None and entry[0].is_fresh():
        # Already usable: rotating again would burn an IdP round trip and, for providers with
        # reuse detection, spend a refresh token for nothing. Callers normally reach here only
        # when the entry is stale, but the guard keeps the function safe to call unconditionally.
        return entry[0]
    refresh_token = entry[0].refresh_token if entry is not None else ""
    if not refresh_token:
        # Nothing to renew with. A fresh entry needs no help; an entry without a refresh token
        # cannot be helped either.
        return None
    try:
        renewed = (refresher or _refresh_via_provider)(refresh_token, key[0])
    except Exception:  # health: allow BLE001 -- renewal must never break a turn
        logger.warning("turn token: refresh for %s failed; the turn will fail closed",
                       key[2] or key[0], exc_info=True)
        return None
    if renewed is None:
        logger.info("turn token: %s's refresh token was rejected; the turn will fail closed",
                    key[2] or key[0])
        return None
    # Compare-and-set: the IdP call above ran outside the lock, and a *browser* refresh or a
    # login may have landed meanwhile with a newer credential. Writing ours unconditionally
    # would clobber that newer state with a token that is already superseded — and, for a
    # rotating provider, put a spent refresh token back in the store. So only replace an entry
    # that is still the one this call started from; otherwise the newer writer wins and we
    # report what the store now holds.
    if not _replace_if_unchanged(key, expected=entry, renewed=renewed):
        logger.info("turn token: %s's credential changed during renewal; keeping the newer one",
                    key[2] or key[0])
    return recall(key)


def _replace_if_unchanged(key: tuple[str, str, str], *, expected: Any, renewed: Any) -> bool:
    """Write ``renewed`` only when the entry is still ``expected`` (identity compare).

    Returns False when a concurrent writer won, leaving its value in place. Compares the tuple
    object, not its fields: ``remember`` always installs a fresh tuple, so identity is an exact
    "has anything written here since I read it" test with no token comparison in it.
    """
    with _store_lock:
        current = _store.get(key)
        if current is not expected:
            return False
        _store[key] = (_token_from_session(renewed, key), time.time())
        _prune_store_locked()
    return True


def _refresh_via_provider(refresh_token: str, provider_hint: str) -> Any:
    """Rotate through the registered dashboard providers; the Session, or ``None``.

    Imported lazily: ``agent`` must stay importable without the dashboard-auth plugin layer, and
    this path only runs inside a turn (i.e. in the gateway, which has it loaded).
    """
    from hermes_cli.dashboard_auth.refresh_singleflight import refresh_session_coalesced

    refreshed = refresh_session_coalesced(
        refresh_token, provider_hint, phase="turn start", log=logger)
    return refreshed[0] if refreshed is not None else None


def _prune_store_locked() -> None:
    """Drop entries that can no longer produce a usable token; cap the rest. Caller holds the lock.

    An entry past its access token's expiry is NOT dead: its refresh token can still renew it,
    and that is exactly the long-idle case this store exists to serve. Dropping such an entry
    would strand the browser until a request happened to arrive and refresh it independently.
    Only an entry with neither a usable access token nor a refresh token is worthless.
    """
    moment = time.time()
    for key in [k for k, (token, _) in _store.items()
                if not token.is_fresh(now=moment) and not token.refresh_token]:
        del _store[key]
    while len(_store) > MAX_STORE_ENTRIES:
        # Prefer dropping expired-but-renewable entries over live ones.
        drained = [k for k, (token, _) in _store.items() if not token.is_fresh(now=moment)]
        victim = min(drained, key=lambda k: _store[k][1]) if drained else \
            min(_store, key=lambda k: _store[k][1])
        del _store[victim]


# ---- session bridge: gateway session id -> login identity -----------------

_sessions_lock = threading.Lock()
#: session id -> {"user_id", "provider", "issuer"}
_login_identities: dict[str, dict[str, str]] = {}


def _session_key(session_id: Any) -> str:
    return str(session_id or "")


def note_login(session_id: Any, identity: Any) -> None:
    """Record that ``session_id`` is being driven by the login described by ``identity``.

    ``identity`` is the server-minted WS/RPC dict (``{user_id, provider, issuer}``). Anything
    that is not a usable identity clears the entry rather than leaving a stale one: a session
    must never keep acting as a user who is no longer driving it.
    """
    key = _session_key(session_id)
    if not key:
        return
    triple = _triple(identity)
    with _sessions_lock:
        if triple is None:
            _login_identities.pop(key, None)
            return
        # Re-insert so the order is least-RECENTLY-used, not oldest-inserted: a session that
        # keeps submitting turns must not be rotated out behind a stream of new ones.
        _login_identities.pop(key, None)
        _login_identities[key] = triple
        while len(_login_identities) > MAX_SESSION_ENTRIES:
            _evict_one_mapping_locked()


def _evict_one_mapping_locked() -> None:
    """Drop the mapping whose eviction costs least. Caller holds ``_sessions_lock``.

    Eviction is not housekeeping here — it decides a security outcome. ``bind_session_token``
    answers ``None`` (= "no login drove this turn") for a session with no mapping, and an
    anonymous turn may fall back to the shared service credential, so dropping a *live* login's
    mapping would silently run that person's query as the integration account.

    A mapping whose login no longer holds a usable token is therefore the only safe victim: the
    turn it drives would fail closed either way (the placeholder bound from the remaining key is
    unusable, which is what raises the "sign in again" refusal). Oldest-used order breaks ties.

    Lock order is ``_sessions_lock`` then ``_store_lock`` — always this way around: no path
    takes ``_store_lock`` and then reaches into this map.
    """
    for session_id, triple in _login_identities.items():
        if recall(key_for_identity(triple)) is None:
            del _login_identities[session_id]
            return
    del _login_identities[next(iter(_login_identities))]


def _triple(identity: Any) -> Optional[dict[str, str]]:
    """Normalise an identity dict to ``{user_id, provider, issuer}``, or ``None``.

    A private/native grant (the Desktop, the TUI, a managed worker) carries provider ``local``
    and a ``user_id`` naming an OS user, which can never match a login's store key — correctly,
    since an OS user holds no OAuth token. When such a grant forwarded a *dashboard* login, the
    subject is that login's namespaced form (``auth:v1:[provider, issuer, user]``); decomposing
    it here is what makes the forwarded identity land on the same key the token was stored under,
    so the Desktop path resolves instead of binding an always-stale placeholder.
    """
    if not isinstance(identity, dict):
        return None
    subject = str(identity.get("user_id") or "").strip()
    provider = str(identity.get("provider") or "").strip()
    if not subject or not provider:
        return None
    issuer = str(identity.get("issuer") or "").strip()
    if provider == "local":
        from gateway.session_identity import AUTH_SUBJECT_PREFIX, parse_authenticated_subject

        # Only a forwarded dashboard login is re-keyed; a plain OS-user grant is left as-is
        # (``parse_authenticated_subject`` answers None for it).
        if subject.startswith(AUTH_SUBJECT_PREFIX):
            parts = parse_authenticated_subject(subject)
            if parts is not None and parts[0] and parts[2].strip():
                return {"user_id": parts[2].strip(), "provider": parts[0], "issuer": parts[1]}
    return {"user_id": subject, "provider": provider, "issuer": issuer}


def identity_for(session_id: Any) -> Optional[dict[str, str]]:
    """The login identity last seen driving ``session_id``, or ``None``."""
    key = _session_key(session_id)
    if not key:
        return None
    with _sessions_lock:
        return _login_identities.get(key)


def clear_sessions() -> None:
    """Test-only: drop every mapping."""
    with _sessions_lock:
        _login_identities.clear()


# ---- per-turn binding -----------------------------------------------------

_current: contextvars.ContextVar[Optional[TurnToken]] = contextvars.ContextVar(
    "turn_token", default=None)


def bind_turn_token(token: Optional[TurnToken]) -> contextvars.Token:
    """Bind the token for the current context; pass the result to :func:`reset_turn_token`."""
    return _current.set(token)


def reset_turn_token(token) -> None:
    """Undo a :func:`bind_turn_token`."""
    _current.reset(token)


def _placeholder(key: tuple[str, str, str]) -> TurnToken:
    """The identity-only token bound when a login exists but no credential does.

    Binding ``None`` here would make the turn anonymous, and an anonymous turn may use the shared
    service credential — so a user whose token merely expired or was rejected would silently run
    their query as the integration account. ``current_turn_token`` still reports this as unusable
    (``is_fresh`` is false), which is what makes a user-scoped MCP call fail closed instead.
    """
    return TurnToken(access_token="", issuer=key[1], subject=key[2], expires_at=0)


def _bind_plan(session_id: Any) -> tuple[Optional[tuple[str, str, str]], bool]:
    """``(key, needs_renewal)`` — the cheap, purely in-memory half of binding a turn.

    ``key`` is ``None`` when the session maps to no login at all (nothing to act as); otherwise
    ``needs_renewal`` says whether the caller has to reach the IdP, which is the only part that
    costs a network round trip. Splitting it out is what lets the sync and async binders share
    one decision and differ only in *where* that round trip runs.
    """
    key = key_for_identity(identity_for(session_id))
    return key, key is not None and recall(key) is None


def _renewal_result(key: tuple[str, str, str]) -> TurnToken:
    """The token to bind after a renewal attempt: the renewed one, or the placeholder."""
    return refresh_store_entry(key) or _placeholder(key)


def bind_session_token(session_id: Any) -> contextvars.Token:
    """Bind ``session_id``'s access token to the CURRENT context; returns the reset token.

    Token-shaped (not a scope object) so a caller whose turn body spans many statements can
    reset it in its existing ``finally`` next to the other turn ContextVars, without
    re-indenting the whole body. Binding happens at execution time, not submit time: the token
    is resolved against the *current* store, so a refresh that landed while the turn waited in
    the queue is already visible. The executor copies this context onto the tool worker, which
    is how a deeply nested MCP call reaches the token with no argument threading.

    Four outcomes, and the two in the middle are the whole point:

    * a live token — the normal case;
    * a **renewed** token — the stored access token was gone or stale, and the login's refresh
      token produced a new one. A browser left idle for hours has had no request to trigger the
      gate's refresh, so this turn is the first chance, and refusing it would demand a re-login
      the user does not owe;
    * an **expired placeholder** when renewal is impossible too (see :func:`_placeholder`);
    * ``None`` only when the session id maps to no login at all.

    This form may block (renewal is network I/O), so it belongs on a turn's own worker thread —
    which is where both callers run. An event-loop caller must use
    :func:`bind_session_token_async`, which makes the identical decision.
    """
    key, needs_renewal = _bind_plan(session_id)
    if key is None:
        return bind_turn_token(None)
    return bind_turn_token(_renewal_result(key) if needs_renewal else recall(key))


async def bind_session_token_async(session_id: Any) -> contextvars.Token:
    """Async :func:`bind_session_token`: identical decision, renewal on a worker thread.

    The renewal drives the provider's refresh grant, which is blocking network I/O. Async
    callers (the gateway's admission drain) must not run it on the event loop, so this offloads
    that one step and binds the result in the caller's context — ``ContextVar.set`` has to
    happen on the caller's task, which is why the split is here rather than inside a thread.
    """
    import asyncio

    key, needs_renewal = _bind_plan(session_id)
    if key is None:
        return bind_turn_token(None)
    if not needs_renewal:
        return bind_turn_token(recall(key))
    return bind_turn_token(await asyncio.to_thread(_renewal_result, key))


@contextlib.contextmanager
def turn_token_scope(token: Optional[TurnToken]) -> Iterator[None]:
    """Scope form of :func:`bind_turn_token` (``None`` = explicitly no identity)."""
    handle = bind_turn_token(token)
    try:
        yield
    finally:
        reset_turn_token(handle)


def current_turn_token() -> Optional[TurnToken]:
    """The running turn's access token, or ``None`` outside a signed-in turn."""
    return _current.get()