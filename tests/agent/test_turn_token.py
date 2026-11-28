"""Per-turn access token: freshness, the store, the session bridge, and per-turn binding.

The dashboard login proves *who* is asking; a downstream MCP server needs a token minted for
*that user*. This module owns the binding half of that hand-off.

The invariants here are security invariants, not conveniences:

* a credential is keyed by login, so two people never share one and two tabs of one person do;
* a turn that cannot produce a user token must fail **closed** — never look anonymous, because
  an anonymous turn is allowed to fall back to the shared service account;
* a stored refresh token is the only thing that can rescue a browser left idle for hours, so
  nothing may discard it;
* nothing a token touches may print it.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time

import pytest

from agent.turn_token import (
    MAX_SESSION_ENTRIES,
    MAX_STORE_ENTRIES,
    TurnToken,
    _store,
    backfill,
    bind_session_token,
    bind_session_token_async,
    bind_turn_token,
    clear_sessions,
    clear_store,
    current_turn_token,
    forget,
    forget_token,
    identity_for,
    identity_key,
    key_for_identity,
    key_for_session,
    note_login,
    recall,
    refresh_store_entry,
    remember,
    reset_turn_token,
    turn_token_scope,
)
from hermes_cli.dashboard_auth.base import Session

ISSUER = "https://sso.example/realms/ebmedical"


def _session(subject: str = "user-1", token: str = "at-1", *, expires_in: int = 600,
             refresh_token: str = "rt-1", provider: str = "keycloak",
             issuer: str = ISSUER) -> Session:
    return Session(user_id=subject, email="a@b.c", display_name="A", org_id="",
                   provider=provider, expires_at=int(time.time()) + expires_in,
                   access_token=token, refresh_token=refresh_token, issuer=issuer)


def _identity(subject: str = "user-1", *, provider: str = "keycloak",
              issuer: str = ISSUER) -> dict:
    return {"user_id": subject, "provider": provider, "issuer": issuer}


def _credential(token: str = "at", expires_in: int = 600, *, refresh_token: str = "") -> TurnToken:
    return TurnToken(access_token=token, issuer=ISSUER, subject="user-1",
                     expires_at=int(time.time()) + expires_in, refresh_token=refresh_token)


@pytest.fixture(autouse=True)
def _clean_stores():
    # The binding lives in a ContextVar, which outlives a test in the same thread: without this
    # reset, a test that binds and does not reset leaks its token into the next one.
    handle = bind_turn_token(None)
    clear_store()
    clear_sessions()
    yield
    reset_turn_token(handle)
    clear_store()
    clear_sessions()


# --------------------------------------------------------------------------- freshness

class TestFreshness:
    def test_fresh_when_expiry_is_comfortably_ahead(self):
        assert _credential().is_fresh() is True

    def test_stale_inside_the_skew_window(self):
        # A token expiring inside the safety margin would be rejected mid-flight by the
        # resource server, so it counts as unusable now.
        assert _credential(expires_in=5).is_fresh() is False

    def test_unknown_expiry_is_never_fresh(self):
        assert TurnToken(access_token="at", expires_at=0).is_fresh() is False

    def test_zero_expiry_in_the_past_is_never_fresh(self):
        assert TurnToken(access_token="at", expires_at=-1).is_fresh() is False

    def test_an_empty_token_is_never_fresh_however_late_it_expires(self):
        assert TurnToken(access_token="", expires_at=9_999_999_999).is_fresh() is False

    @pytest.mark.parametrize("expires_in, skew, expected", [
        (600, 60, True),
        (60, 60, False),   # exactly at the boundary is not "comfortably ahead"
        (61, 60, True),
    ])
    def test_skew_boundary(self, expires_in, skew, expected):
        assert _credential(expires_in=expires_in).is_fresh(skew_seconds=skew) is expected


# --------------------------------------------------------------------------- secrets

class TestTokensAreNeverPrinted:
    """A bearer that reaches a log line is a leaked bearer: the provider cannot revoke a
    signed JWT, so the only defence is that it is never rendered."""

    def test_neither_token_appears_in_repr(self):
        credential = _credential("AT-SECRET", refresh_token="RT-SECRET")
        rendered = repr(credential)
        assert "AT-SECRET" not in rendered
        assert "RT-SECRET" not in rendered

    def test_the_identity_still_shows_for_diagnostics(self):
        rendered = repr(_credential("AT-SECRET"))
        assert "user-1" in rendered  # whose token it is must stay debuggable


# --------------------------------------------------------------------------- keys

class TestKeys:
    def test_a_login_triple_is_the_key(self):
        assert identity_key(provider="keycloak", issuer=ISSUER, subject="u") == (
            "keycloak", ISSUER, "u")

    @pytest.mark.parametrize("subject", ["", "   ", None])
    def test_an_unidentifiable_login_has_no_key(self, subject):
        # The dangerous case: every anonymous login would collapse onto one key.
        assert identity_key(provider="keycloak", issuer=ISSUER, subject=subject) is None

    def test_provider_and_issuer_whitespace_is_normalised(self):
        assert identity_key(provider=" keycloak ", issuer=" iss ", subject=" u ") == (
            "keycloak", "iss", "u")

    def test_key_from_a_ws_identity_dict(self):
        assert key_for_identity(_identity()) == ("keycloak", ISSUER, "user-1")

    @pytest.mark.parametrize("identity", [None, "string", 42, [], {}])
    def test_unknown_identity_shapes_answer_none(self, identity):
        assert key_for_identity(identity) is None

    def test_key_from_a_session(self):
        assert key_for_session(_session()) == ("keycloak", ISSUER, "user-1")

    def test_a_session_without_a_subject_has_no_key(self):
        assert key_for_session(_session("")) is None


# --------------------------------------------------------------------------- the store

class TestRemember:
    def test_it_records_the_token_under_the_login(self):
        assert remember(_session(token="at-x")) is True
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "at-x"

    def test_a_session_without_a_token_removes_the_entry(self):
        """A stale credential must not outlive the login that replaced it."""
        remember(_session(token="old"))
        assert remember(_session(token="")) is False
        assert recall(("keycloak", ISSUER, "user-1")) is None

    def test_the_refresh_token_is_stored_too(self):
        remember(_session(token="at", refresh_token="rt-x"))
        assert recall(("keycloak", ISSUER, "user-1")).refresh_token == "rt-x"

    def test_a_later_write_replaces_an_earlier_one(self):
        remember(_session(token="at-1", refresh_token="rt-1"))
        remember(_session(token="at-2", refresh_token="rt-2"))
        stored = recall(("keycloak", ISSUER, "user-1"))
        assert (stored.access_token, stored.refresh_token) == ("at-2", "rt-2")


class TestRecall:
    def test_missing_key_is_none(self):
        assert recall(None) is None
        assert recall(("keycloak", ISSUER, "nobody")) is None

    def test_a_stale_entry_is_not_returned(self):
        remember(_session(expires_in=-900))
        assert recall(("keycloak", ISSUER, "user-1")) is None

    def test_two_logins_never_share(self):
        remember(_session("alice", "at-a"))
        remember(_session("bob", "at-b"))
        assert recall(("keycloak", ISSUER, "alice")).access_token == "at-a"
        assert recall(("keycloak", ISSUER, "bob")).access_token == "at-b"

    def test_one_login_in_two_providers_stays_separate(self):
        remember(_session(token="kc-at", provider="keycloak"))
        remember(_session(token="basic-at", provider="basic", issuer=""))
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "kc-at"
        assert recall(("basic", "", "user-1")).access_token == "basic-at"


class TestForget:
    def test_forget_drops_one_entry(self):
        remember(_session())
        forget(("keycloak", ISSUER, "user-1"))
        assert recall(("keycloak", ISSUER, "user-1")) is None

    def test_forgetting_a_missing_entry_is_a_no_op(self):
        forget(None)
        forget(("keycloak", ISSUER, "never-seen"))

    def test_sign_out_drops_the_entry_holding_that_bearer(self):
        remember(_session(token="at-to-revoke"))
        assert forget_token("at-to-revoke") is True
        assert recall(("keycloak", ISSUER, "user-1")) is None

    @pytest.mark.parametrize("bearer", ["", "not-a-token-we-store"])
    def test_an_unknown_bearer_frees_nothing(self, bearer):
        remember(_session(token="other"))
        assert forget_token(bearer) is False
        assert recall(("keycloak", ISSUER, "user-1")) is not None


class TestStoreBounds:
    def test_the_store_is_capped_and_keeps_the_newest(self):
        for index in range(MAX_STORE_ENTRIES + 20):
            remember(_session(f"u{index}", f"at-{index}"))
        assert len(_store) == MAX_STORE_ENTRIES
        newest = f"u{MAX_STORE_ENTRIES + 19}"
        assert recall(("keycloak", ISSUER, newest)) is not None
        assert recall(("keycloak", ISSUER, "u0")) is None

    def test_an_expired_entry_with_a_refresh_token_is_kept(self):
        """It is the long-idle case's only raw material; freshness alone must not evict it."""
        remember(_session(expires_in=-900, refresh_token="rt-good"))
        assert ("keycloak", ISSUER, "user-1") in _store

    def test_an_expired_entry_without_a_refresh_token_is_dropped(self):
        remember(_session(expires_in=-900, refresh_token=""))
        assert ("keycloak", ISSUER, "user-1") not in _store

    def test_under_capacity_pressure_expired_entries_go_first(self):
        remember(_session("u-live", expires_in=600))          # live
        remember(_session("u-dead", expires_in=-900))         # expired but renewable
        for index in range(MAX_STORE_ENTRIES):
            remember(_session(f"f{index}", expires_in=600))
        # The expired-but-renewable entry is the preferred victim.
        assert ("keycloak", ISSUER, "u-dead") not in _store


# --------------------------------------------------------------------------- backfill

class TestBackfill:
    """A restart empties the store while browser cookies stay valid. The next verified request
    must refill it, or the user is made to sign in again for no reason."""

    def test_an_empty_store_is_refilled(self):
        assert backfill(_session(token="cookie-at")) is True
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "cookie-at"

    def test_a_live_entry_is_never_overwritten(self):
        remember(_session(token="fresh-at"))
        assert backfill(_session(token="older-at")) is False
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "fresh-at"

    def test_a_stale_entry_is_replaced(self):
        remember(_session(token="dead-at", expires_in=5))  # inside the skew window
        assert backfill(_session(token="new-at")) is True
        assert recall(("keycloak", ISSUER, "user-1")).access_token == "new-at"

    def test_a_bearer_without_a_refresh_token_keeps_the_stored_one(self):
        """``verify_session`` re-verifies a bearer and returns ``refresh_token=""``.

        Writing that as-is would erase the only means of renewing a stale entry — destroying
        the very case the store exists to serve.
        """
        remember(_session(token="dead-at", expires_in=-900, refresh_token="rt-good"))
        assert backfill(_session(token="bearer-at", refresh_token="")) is True
        stored = recall(("keycloak", ISSUER, "user-1"))
        assert stored.access_token == "bearer-at"      # the newly verified bearer is live
        assert stored.refresh_token == "rt-good"       # ...and the old means of renewal survived

    def test_a_rotation_still_replaces_the_refresh_token(self):
        remember(_session(token="at-1", refresh_token="rt-1"))
        remember(_session(token="at-2", refresh_token="rt-2"))
        assert recall(("keycloak", ISSUER, "user-1")).refresh_token == "rt-2"

    def test_an_unusable_session_is_not_reported_as_written(self):
        """``True`` must mean "the store now holds a usable token for this login"."""
        assert backfill(_session(token="dead-at", expires_in=-900, refresh_token="")) is False
        assert ("keycloak", ISSUER, "user-1") not in _store

    @pytest.mark.parametrize("token", ["", None])
    def test_a_session_without_a_token_backfills_nothing(self, token):
        assert backfill(_session(token=token or "")) is False

    @pytest.mark.parametrize("subject", ["", "   "])
    def test_a_session_without_a_subject_backfills_nothing(self, subject):
        assert backfill(_session(subject, "at")) is False


# --------------------------------------------------------------------------- session bridge

class TestSessionBridge:
    """Which login drove which session — deliberately separate from session ownership."""

    def test_note_login_then_bind(self):
        note_login("sess-1", _identity())
        remember(_session(token="at-1"))
        bind_session_token("sess-1")
        assert current_turn_token().access_token == "at-1"

    def test_a_session_with_no_login_binds_nothing(self):
        """CLI, cron and messaging turns have no dashboard sign-in: there is no user to act as,
        so no token may be attached (``None``, not a placeholder)."""
        bind_session_token("sess-without-login")
        assert current_turn_token() is None

    def test_a_login_with_a_stale_token_binds_a_placeholder(self):
        """Not ``None``: an anonymous turn may use the service credential, so the turn must
        still look like *someone* and fail closed instead."""
        note_login("sess-1", _identity())
        remember(_session(expires_in=-900, refresh_token=""))
        bind_session_token("sess-1")
        bound = current_turn_token()
        assert bound is not None
        assert bound.access_token == ""
        assert bound.is_fresh() is False

    def test_a_session_switching_users_resolves_to_the_new_one(self):
        remember(_session("alice", "at-alice"))
        remember(_session("bob", "at-bob"))
        note_login("sess-1", _identity("alice"))
        note_login("sess-1", _identity("bob"))
        bind_session_token("sess-1")
        assert current_turn_token().access_token == "at-bob"

    @pytest.mark.parametrize("bad_identity", [None, "str", {}, {"user_id": ""},
                                              {"provider": "keycloak"}])
    def test_an_unusable_identity_clears_the_mapping(self, bad_identity):
        note_login("sess-1", _identity())
        note_login("sess-1", bad_identity)
        assert identity_for("sess-1") is None

    @pytest.mark.parametrize("session_id", ["", None])
    def test_note_login_without_a_session_id_is_ignored(self, session_id):
        note_login(session_id, _identity())
        assert identity_for(session_id) is None

    def test_an_identity_without_a_provider_is_not_a_login(self):
        # ``local`` with no provider is meaningless; a provider-less dict cannot key a login.
        note_login("sess-1", {"user_id": "u"})
        assert identity_for("sess-1") is None


class TestSessionMappingBounds:
    def test_the_mapping_is_bounded(self):
        for index in range(MAX_SESSION_ENTRIES + 25):
            note_login(f"s{index}", _identity(f"u{index}"))
        assert identity_for(f"s{MAX_SESSION_ENTRIES + 24}") is not None
        assert identity_for("s0") is None

    def test_re_noting_a_session_spares_it_from_eviction(self):
        """Eviction is least-RECENTLY-used, not oldest-inserted: a session that keeps submitting
        must not be rotated out behind a stream of new ones.

        The failure this guards is fail-OPEN: an evicted mapping makes the bind answer ``None``,
        and an anonymous turn may fall back to the shared service credential.
        """
        note_login("veteran", _identity("alice"))
        for index in range(MAX_SESSION_ENTRIES - 1):
            note_login(f"o{index}", _identity(f"u{index}"))
        note_login("veteran", _identity("alice"))    # submits again -> most recent
        note_login("newcomer", _identity("bob"))     # pushes the map over the limit
        assert identity_for("veteran") is not None

    def test_a_live_login_outlives_a_flood_of_tokenless_sessions(self):
        """Eviction prefers a mapping whose login holds no usable token — the security rule."""
        remember(_session("alice", "at-alice"))
        note_login("alice", _identity("alice"))
        for index in range(MAX_SESSION_ENTRIES * 2):   # signed out / never renewed
            note_login(f"dead{index}", _identity(f"nobody{index}"))
        bind_session_token("alice")
        assert current_turn_token().access_token == "at-alice"


# --------------------------------------------------------------------------- forwarded desktop

class TestForwardedDesktopLogin:
    """A native/Desktop grant carries provider ``local`` and an OS-user ``user_id``, which can
    never key a login — correctly, since an OS user holds no OAuth token. When that grant
    forwards a dashboard login, the subject is ``auth:v1:[provider, issuer, user]`` and must be
    decomposed so the turn lands on the same key the token was stored under."""

    def test_a_forwarded_login_resolves_the_stored_token(self):
        remember(_session("u1", "AT-u1"))
        forwarded = "auth:v1:" + json.dumps(["keycloak", ISSUER, "u1"])
        note_login("desk", {"user_id": forwarded, "provider": "local"})
        bind_session_token("desk")
        assert current_turn_token().access_token == "AT-u1"

    def test_a_plain_os_user_grant_invents_no_login(self):
        remember(_session("u1", "AT-u1"))
        note_login("desk", {"user_id": "osuser", "provider": "local"})
        bind_session_token("desk")
        bound = current_turn_token()
        assert bound is not None           # ``local`` is a real triple, so a placeholder binds
        assert bound.access_token == ""    # ...never someone else's token

    @pytest.mark.parametrize("subject", [
        "auth:v1:not-json",
        "auth:v1:[]",
        'auth:v1:["","","u1"]',            # no provider
        'auth:v1:["keycloak","i",""]',     # no user
    ])
    def test_a_malformed_namespaced_subject_is_left_alone(self, subject):
        note_login("desk", {"user_id": subject, "provider": "local"})
        assert identity_for("desk") == {"user_id": subject, "provider": "local", "issuer": ""}


# --------------------------------------------------------------------------- binding

class TestBinding:
    def test_no_binding_outside_a_turn(self):
        assert current_turn_token() is None

    def test_scope_binds_and_restores(self):
        credential = _credential()
        with turn_token_scope(credential):
            assert current_turn_token() is credential
        assert current_turn_token() is None

    def test_scope_none_clears_an_outer_binding(self):
        with turn_token_scope(_credential()):
            with turn_token_scope(None):
                assert current_turn_token() is None
            assert current_turn_token() is not None

    def test_explicit_reset_restores_the_previous_value(self):
        outer = bind_turn_token(_credential("outer"))
        inner = bind_turn_token(_credential("inner"))
        assert current_turn_token().access_token == "inner"
        reset_turn_token(inner)
        assert current_turn_token().access_token == "outer"
        reset_turn_token(outer)


class TestIsolation:
    """The bug class this guards: a turn must never carry another login's token."""

    def test_a_copied_context_carries_the_token_to_a_worker(self):
        # Tool executors run on workers whose context comes from copy_context(); that copy is
        # the ONLY thing carrying the token into an MCP call.
        seen: dict = {}

        def worker():
            seen["credential"] = current_turn_token()

        with turn_token_scope(_credential("worker-at")):
            context = contextvars.copy_context()
            thread = threading.Thread(target=lambda: context.run(worker))
            thread.start()
            thread.join()

        assert seen["credential"] is not None
        assert seen["credential"].access_token == "worker-at"

    def test_concurrent_turns_do_not_see_each_other(self):
        results: dict[str, str | None] = {}
        barrier = threading.Barrier(2)

        def turn(name: str, token: str):
            with turn_token_scope(_credential(token)):
                barrier.wait()  # both turns hold their binding at the same time
                credential = current_turn_token()
                results[name] = credential.access_token if credential else None

        threads = [threading.Thread(target=turn, args=("a", "at-a")),
                   threading.Thread(target=turn, args=("b", "at-b"))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results == {"a": "at-a", "b": "at-b"}

    def test_a_worker_outliving_its_turn_keeps_its_snapshot(self):
        # A long tool call must not start reading a newer turn's credential mid-flight.
        captured: dict = {}
        release = threading.Event()

        def slow_worker():
            captured["before"] = current_turn_token().access_token
            release.wait(5)
            captured["after"] = current_turn_token().access_token

        with turn_token_scope(_credential("turn-1")):
            context = contextvars.copy_context()
            thread = threading.Thread(target=lambda: context.run(slow_worker))
            thread.start()
            with turn_token_scope(_credential("turn-2")):  # a later turn begins
                release.set()
                thread.join()

        assert captured == {"before": "turn-1", "after": "turn-1"}


# --------------------------------------------------------------------------- renewal

class TestLongIdleRenewal:
    """A browser left alone for hours comes back to a dead access token.

    Nothing refreshed it: the gate renews only when a request arrives, and none has. The turn is
    therefore the first chance, and refusing it would demand a re-login the user does not owe —
    the refresh token is still good.
    """

    def _key(self, subject: str = "user-1") -> tuple[str, str, str]:
        return ("keycloak", ISSUER, subject)

    def test_a_healthy_entry_is_returned_without_renewing(self):
        remember(_session(token="at-live"))
        calls: list[str] = []

        def _refresh(refresh_token, provider):
            calls.append(refresh_token)
            return _session(token="at-unused")

        assert refresh_store_entry(self._key(), refresher=_refresh).access_token == "at-live"
        assert calls == []   # rotating again would burn an IdP round trip for nothing

    def test_a_stale_entry_is_renewed(self):
        remember(_session(token="at-dead", expires_in=-900, refresh_token="rt-good"))
        renewed = refresh_store_entry(
            self._key(), refresher=lambda rt, p: _session(token="at-fresh", refresh_token="rt-2"))
        assert renewed.access_token == "at-fresh"
        # The rotated credentials are stored, so the *next* idle return has them.
        stored = recall(self._key())
        assert (stored.access_token, stored.refresh_token) == ("at-fresh", "rt-2")

    def test_an_entry_without_a_refresh_token_cannot_be_renewed(self):
        remember(_session(token="at-dead", expires_in=-900, refresh_token=""))
        assert refresh_store_entry(self._key()) is None

    def test_a_rejected_refresh_token_answers_none(self):
        remember(_session(token="at-dead", expires_in=-900, refresh_token="rt-dead"))
        assert refresh_store_entry(self._key(), refresher=lambda rt, p: None) is None

    def test_a_raising_refresher_never_breaks_the_turn(self):
        remember(_session(token="at-dead", expires_in=-900, refresh_token="rt"))

        def _boom(refresh_token, provider):
            raise OSError("idp unreachable")

        assert refresh_store_entry(self._key(), refresher=_boom) is None

    def test_a_renewed_but_still_stale_token_binds_nothing(self):
        # An IdP reporting a token already inside the skew window is not usable yet; the entry
        # stays so a later attempt can try again.
        remember(_session(token="at-dead", expires_in=-900))
        renewed = refresh_store_entry(
            self._key(), refresher=lambda rt, p: _session(token="at-short", expires_in=5))
        assert renewed is None
        assert self._key() in _store

    def test_a_slow_renewal_does_not_clobber_a_newer_write(self):
        """The IdP call runs outside the lock, so a browser refresh can land while it is in
        flight. Writing ours unconditionally would replace a newer credential — and, for a
        rotating provider, put an already-spent refresh token back in the store."""
        remember(_session(token="at-dead", expires_in=-900, refresh_token="rt-0"))
        started, release = threading.Event(), threading.Event()

        def _slow_refresh(refresh_token, provider):
            started.set()
            release.wait(5)
            return _session(token="AT-FROM-TURN", refresh_token="RT-FROM-TURN")

        worker = threading.Thread(
            target=lambda: refresh_store_entry(self._key(), refresher=_slow_refresh))
        worker.start()
        assert started.wait(5)
        remember(_session(token="AT-FROM-BROWSER", refresh_token="RT-FROM-BROWSER"))
        release.set()
        worker.join(5)

        stored = recall(self._key())
        assert stored.access_token == "AT-FROM-BROWSER"
        assert stored.refresh_token == "RT-FROM-BROWSER"


class TestBindRenews:
    """The bind is what turns "the token expired while I was away" into a working turn."""

    def _expired_login(self, *, refresh_token: str = "rt-good") -> None:
        remember(_session(token="at-dead", expires_in=-900, refresh_token=refresh_token))
        note_login("sess-1", _identity())

    def test_a_stale_token_is_renewed_before_binding(self, monkeypatch):
        import agent.turn_token as store

        self._expired_login()
        monkeypatch.setattr(store, "_refresh_via_provider",
                            lambda rt, p: _session(token="at-fresh"))
        bind_session_token("sess-1")
        assert current_turn_token().access_token == "at-fresh"

    def test_a_rejected_renewal_still_fails_closed(self, monkeypatch):
        import agent.turn_token as store

        self._expired_login()
        monkeypatch.setattr(store, "_refresh_via_provider", lambda rt, p: None)
        bind_session_token("sess-1")
        bound = current_turn_token()
        assert bound is not None
        assert bound.access_token == ""   # refused, never the service account

    def test_the_async_bind_makes_the_same_decision(self, monkeypatch):
        import asyncio

        import agent.turn_token as store

        self._expired_login()
        monkeypatch.setattr(store, "_refresh_via_provider",
                            lambda rt, p: _session(token="at-fresh"))

        async def _drive():
            await store.bind_session_token_async("sess-1")
            return current_turn_token()   # read in the task that bound it

        assert asyncio.run(_drive()).access_token == "at-fresh"

    def test_the_async_bind_renews_on_a_worker_thread(self, monkeypatch):
        """The gateway's admission drain is async; the refresh grant is blocking network I/O."""
        import asyncio

        import agent.turn_token as store

        self._expired_login()
        loop_thread = threading.get_ident()
        seen: dict = {}

        def _refresh(refresh_token, provider):
            seen["ident"] = threading.get_ident()
            return _session(token="at-fresh")

        monkeypatch.setattr(store, "_refresh_via_provider", _refresh)
        asyncio.run(store.bind_session_token_async("sess-1"))
        assert seen["ident"] != loop_thread

    def test_the_async_bind_does_not_touch_the_provider_when_fresh(self, monkeypatch):
        import asyncio

        import agent.turn_token as store

        remember(_session(token="at-live"))
        note_login("sess-1", _identity())
        monkeypatch.setattr(store, "_refresh_via_provider",
                            lambda *a: pytest.fail("must not reach the IdP"))

        async def _drive():
            await store.bind_session_token_async("sess-1")
            return current_turn_token()

        assert asyncio.run(_drive()).access_token == "at-live"