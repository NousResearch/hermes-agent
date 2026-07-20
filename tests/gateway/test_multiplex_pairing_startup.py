"""Per-profile PairingStore availability at startup, and fail-closed lookup.

Two defects are pinned here, both about the window between "an adapter is live"
and "this gateway knows which pairing whitelist that adapter's profile uses":

1. ``_start_secondary_profile_adapters`` registered every profile's PairingStore
   only *after* the connect loop finished. Slack Socket Mode is accepting events
   the moment ``connect()`` returns, and the Slack custom-message seam authorizes
   ahead of the startup restore queue — so a correctly stamped secondary source
   could reach ``_is_user_authorized`` while ``pairing_stores`` was still empty.

2. ``_pairing_store_for`` then fell back to ``self.pairing_store`` — the DEFAULT
   profile's whitelist. That turns a startup race (or a failed store init, or a
   stale profile) into a cross-profile authorization grant.

These drive the REAL ``GatewayRunner`` methods and the REAL
``GatewayAuthorizationMixin._is_user_authorized`` against real ``PairingStore``
instances on disk, not a fake runner.
"""

import asyncio
import re
from unittest.mock import MagicMock, patch

import pytest

import gateway.pairing
from gateway.config import Platform
from gateway.pairing import MAX_PENDING_PER_PLATFORM, PairingStore
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _bare_runner(multiplex: bool = True):
    """A GatewayRunner with only the state the methods under test touch."""
    runner = object.__new__(GatewayRunner)
    runner.config = MagicMock(multiplex_profiles=multiplex)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_stores = {}
    runner._pairing_store_failed = set()
    runner.pairing_store = PairingStore()
    return runner


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # ``PAIRING_DIR`` is resolved at *import* time from the real HERMES_HOME, and
    # a profile-less ``PairingStore()`` — the global/default store these tests
    # now exercise directly — reads it instead of the env var. Without this the
    # suite reads and writes the developer's own ~/.hermes pairing files.
    monkeypatch.setattr(
        gateway.pairing, "PAIRING_DIR", home / "platforms" / "pairing"
    )
    return home


class TestStoreRegisteredBeforeConnect:
    """The store must exist before that profile's adapters go live."""

    def test_store_registered_before_profile_adapters_connect(self, hermes_home):
        """Observed at connect time, not after the loop.

        ``_start_one_profile_adapters`` stands in for the real connect; it
        records what ``pairing_stores`` looked like at the instant this
        profile's adapters would have started accepting traffic. Pre-fix that
        snapshot was empty for every profile.
        """
        runner = _bare_runner()
        seen_at_connect = {}

        async def _record_then_connect(profile_name, profile_home, claimed):
            seen_at_connect[profile_name] = set(runner.pairing_stores)
            runner._profile_adapters[profile_name] = {}
            return 1

        runner._start_one_profile_adapters = _record_then_connect
        runner._adapter_credential_fingerprint = lambda adapter: None

        with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
            ("coder", hermes_home / "profiles" / "coder"),
            ("ops", hermes_home / "profiles" / "ops"),
        ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            asyncio.run(runner._start_secondary_profile_adapters())

        assert "coder" in seen_at_connect["coder"], (
            "coder's PairingStore was still unregistered when its adapters "
            "connected — the startup race is back"
        )
        assert "ops" in seen_at_connect["ops"], (
            "ops's PairingStore was still unregistered when its adapters connected"
        )

    def test_store_registered_even_when_connect_fails(self, hermes_home):
        """A profile whose adapters all fail still gets its store.

        The safety-net pass after the loop covers it, so a later retry-connect
        cannot find the profile served but store-less.
        """
        runner = _bare_runner()

        async def _boom(profile_name, profile_home, claimed):
            raise RuntimeError("connect failed")

        runner._start_one_profile_adapters = _boom
        runner._adapter_credential_fingerprint = lambda adapter: None

        with patch("hermes_cli.profiles.profiles_to_serve", return_value=[
            ("coder", hermes_home / "profiles" / "coder"),
        ]), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            asyncio.run(runner._start_secondary_profile_adapters())

        assert "coder" in runner.pairing_stores
        assert "default" in runner.pairing_stores

    def test_ensure_pairing_store_is_idempotent(self, hermes_home):
        """Re-registering must not replace a live store instance."""
        runner = _bare_runner()

        assert runner._ensure_pairing_store("coder") is True
        first = runner.pairing_stores["coder"]
        assert runner._ensure_pairing_store("coder") is True
        assert runner.pairing_stores["coder"] is first

    def test_blank_profile_name_is_not_registered(self, hermes_home):
        """An empty/whitespace profile is not a profile — never register one."""
        runner = _bare_runner()

        assert runner._ensure_pairing_store("") is False
        assert runner._ensure_pairing_store("   ") is False
        assert runner._ensure_pairing_store(None) is False
        assert runner.pairing_stores == {}


class TestStoreInitFailure:
    """A store that fails to open must deny, never degrade to the default."""

    def test_init_failure_records_profile_and_denies(self, hermes_home):
        runner = _bare_runner()

        with patch("gateway.pairing.PairingStore", side_effect=OSError("disk on fire")):
            assert runner._ensure_pairing_store("coder") is False

        assert "coder" not in runner.pairing_stores
        assert "coder" in runner._pairing_store_failed

        # The authorization seam must NOT hand back the default store.
        source = SessionSource(platform=Platform.SLACK, chat_id="C1", profile="coder")
        assert runner._pairing_store_for(source) is None, (
            "a profile whose store failed to init fell back to the default "
            "profile's pairing whitelist"
        )

    def test_recovery_after_transient_failure(self, hermes_home):
        """Once the store opens, the profile is no longer marked failed."""
        runner = _bare_runner()

        with patch("gateway.pairing.PairingStore", side_effect=OSError("transient")):
            runner._ensure_pairing_store("coder")
        assert "coder" in runner._pairing_store_failed

        assert runner._ensure_pairing_store("coder") is True
        assert "coder" not in runner._pairing_store_failed
        source = SessionSource(platform=Platform.SLACK, chat_id="C1", profile="coder")
        assert runner._pairing_store_for(source) is runner.pairing_stores["coder"]


class TestPairingStoreForFailsClosed:
    """``_pairing_store_for`` selection rules, on the real mixin method."""

    def test_missing_store_under_multiplex_returns_none(self, hermes_home):
        runner = _bare_runner(multiplex=True)
        source = SessionSource(platform=Platform.SLACK, chat_id="C1", profile="ghost")

        assert runner._pairing_store_for(source) is None

    def test_registered_profile_gets_its_own_store(self, hermes_home):
        runner = _bare_runner(multiplex=True)
        runner._ensure_pairing_store("coder")
        runner._ensure_pairing_store("ops")

        coder = SessionSource(platform=Platform.SLACK, chat_id="C1", profile="coder")
        ops = SessionSource(platform=Platform.SLACK, chat_id="C1", profile="ops")

        assert runner._pairing_store_for(coder) is runner.pairing_stores["coder"]
        assert runner._pairing_store_for(ops) is runner.pairing_stores["ops"]
        assert runner.pairing_stores["coder"] is not runner.pairing_stores["ops"]

    def test_unstamped_source_under_multiplex_uses_default(self, hermes_home):
        """No profile at all is a different case from an unserved profile.

        An unstamped source has not asserted any profile identity, so the
        default store stays correct; the fail-closed rule targets sources that
        DO name a profile we cannot serve.
        """
        runner = _bare_runner(multiplex=True)
        source = SessionSource(platform=Platform.SLACK, chat_id="C1")

        assert runner._pairing_store_for(source) is runner.pairing_store

    def test_single_profile_gateway_unchanged(self, hermes_home):
        """Without multiplexing the global store answers everything."""
        runner = _bare_runner(multiplex=False)
        source = SessionSource(platform=Platform.SLACK, chat_id="C1", profile="anything")

        assert runner._pairing_store_for(source) is runner.pairing_store


class TestAuthorizationIntegration:
    """End-to-end through the real ``_is_user_authorized``."""

    @pytest.fixture(autouse=True)
    def _no_env_allowlists(self, monkeypatch):
        # Strip every allowlist so the pairing grant is the only thing that can
        # authorize — otherwise these tests could pass for the wrong reason.
        for var in (
            "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
            "SLACK_ALLOWED_USERS", "SLACK_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(var, raising=False)

    def _runner_with_default_grant(self, user_id="U_PAIRED"):
        runner = _bare_runner(multiplex=True)
        runner._ensure_pairing_store("default")
        runner.pairing_stores["default"]._approve_user("slack", user_id)
        return runner

    def test_default_grant_does_not_authorize_unregistered_profile(self, hermes_home):
        """The core cross-profile leak.

        ``U_PAIRED`` is paired on the DEFAULT profile only. A source stamped for
        a secondary profile with no registered store must not inherit that
        grant — pre-fix, ``_pairing_store_for`` returned the default store and
        ``is_approved`` said yes.
        """
        runner = self._runner_with_default_grant()
        source = SessionSource(
            platform=Platform.SLACK, chat_id="C1",
            user_id="U_PAIRED", chat_type="dm", profile="coder",
        )

        assert runner._is_user_authorized(source) is False, (
            "a default-profile pairing grant authorized a secondary profile's "
            "traffic — cross-profile pairing leak"
        )

    def test_default_grant_still_authorizes_default_profile(self, hermes_home):
        """The fail-closed rule must not break the legitimate case."""
        runner = self._runner_with_default_grant()
        source = SessionSource(
            platform=Platform.SLACK, chat_id="C1",
            user_id="U_PAIRED", chat_type="dm", profile="default",
        )

        assert runner._is_user_authorized(source) is True

    def test_grant_is_isolated_to_its_own_profile(self, hermes_home):
        """A secondary profile's own grant authorizes only that profile."""
        runner = _bare_runner(multiplex=True)
        runner._ensure_pairing_store("default")
        runner._ensure_pairing_store("coder")
        runner.pairing_stores["coder"]._approve_user("slack", "U_CODER")

        def _source(profile):
            return SessionSource(
                platform=Platform.SLACK, chat_id="C1",
                user_id="U_CODER", chat_type="dm", profile=profile,
            )

        assert runner._is_user_authorized(_source("coder")) is True
        assert runner._is_user_authorized(_source("default")) is False

    def test_startup_race_window_denies(self, hermes_home):
        """A stamped source arriving before its store exists is denied.

        This is the shape of the original race: the adapter is live, the store
        is not yet registered. Denying is correct; inheriting the default
        profile's whitelist is the bug.
        """
        runner = self._runner_with_default_grant()
        # 'coder' is served but its store has not been registered yet.
        runner._profile_adapters["coder"] = {}
        source = SessionSource(
            platform=Platform.SLACK, chat_id="C1",
            user_id="U_PAIRED", chat_type="dm", profile="coder",
        )

        assert runner._is_user_authorized(source) is False


class TestDefaultProfileStoreAliasesGlobal:
    """``default`` must resolve to the legacy global store, not a new subtree.

    The "default" profile IS ``~/.hermes`` — there is no ``profiles/default/``
    directory. ``PairingStore(profile="default")`` therefore opens a *different*
    directory from the one ``hermes pairing approve`` and the dashboard write,
    so a code issued under the default profile could never be approved.
    """

    def test_default_store_is_the_global_store(self, hermes_home):
        runner = _bare_runner()

        assert runner._ensure_pairing_store("default") is True
        assert runner.pairing_stores["default"] is runner.pairing_store, (
            "the default profile got its own PairingStore instead of the "
            "legacy global one the CLI and dashboard use"
        )

    def test_default_store_uses_the_legacy_directory(self, hermes_home):
        """Pin the directory, not just object identity."""
        runner = _bare_runner()
        runner._ensure_pairing_store("default")

        assert runner.pairing_stores["default"].profile is None
        assert "profiles/default" not in str(runner.pairing_stores["default"]._dir)

    def test_code_issued_for_default_is_visible_to_the_cli_store(self, hermes_home):
        """Round-trip: generated on the runner, read back on a fresh global store.

        A separate ``PairingStore()`` is exactly what ``hermes pairing approve``
        constructs, so this proves the CLI can see the code.
        """
        runner = _bare_runner()
        runner._ensure_pairing_store("default")
        source = SessionSource(
            platform=Platform.SLACK, chat_id="D1",
            user_id="U_NEW", chat_type="dm", profile="default",
        )

        code = runner._pairing_store_for(source).generate_code(
            "slack", "U_NEW", "newcomer"
        )
        assert code

        # A bare ``PairingStore()`` is exactly what the CLI constructs. Codes are
        # stored salted+hashed, so its own verification is the only honest way
        # to ask "can this store approve that code".
        cli_view = PairingStore()
        approved = cli_view.approve_code("slack", code)
        assert approved and approved.get("user_id") == "U_NEW", (
            "a default-profile pairing code was invisible to the CLI's global store"
        )

    def test_secondary_profile_still_gets_its_own_store(self, hermes_home):
        """The aliasing is default-only — secondaries stay isolated."""
        runner = _bare_runner()
        runner._ensure_pairing_store("default")
        runner._ensure_pairing_store("coder")

        assert runner.pairing_stores["coder"] is not runner.pairing_store
        assert runner.pairing_stores["coder"].profile == "coder"


class TestPairingCodeGenerationUsesTheAuthStore:
    """Generation and rate-limiting must hit the store auth reads.

    ``_handle_message`` wrote ``self.pairing_store`` (the global one) while
    ``_is_user_authorized`` read ``_pairing_store_for(source)``. For a secondary
    profile those are different files: the code landed in the default profile's
    pending list, so approving it never granted access, and the rate-limit
    counter was written where nobody looked.

    These drive the REAL ``GatewayRunner._handle_message`` pairing branch.
    """

    @pytest.fixture(autouse=True)
    def _no_env_allowlists(self, monkeypatch):
        for var in (
            "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
            "SLACK_ALLOWED_USERS", "SLACK_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(var, raising=False)

    def _runner_for_dm(self, multiplex=True):
        runner = _bare_runner(multiplex=multiplex)
        runner._startup_restore_in_progress = False
        runner._scale_to_zero_note_real_inbound = lambda: None
        runner._get_unauthorized_dm_behavior = lambda platform, profile=None: "pair"
        runner.sent = []

        adapter = MagicMock()

        async def _send(chat_id, text, *a, **kw):
            runner.sent.append(text)

        adapter.send = _send
        runner._adapter_for_source = lambda source: adapter
        return runner

    @staticmethod
    def _dm_event(profile):
        from gateway.platforms.base import MessageEvent

        return MessageEvent(
            text="hello?",
            source=SessionSource(
                platform=Platform.SLACK, chat_id="D1",
                user_id="U_NEW", user_name="newcomer",
                chat_type="dm", profile=profile,
            ),
        )

    @staticmethod
    def _issued_code(runner):
        """Pull the code out of the pairing message the user was sent."""
        assert runner.sent, "no pairing message was sent"
        match = re.search(r"pairing code: `([^`]+)`", runner.sent[0])
        assert match, f"unexpected pairing message: {runner.sent[0]!r}"
        return match.group(1)

    @staticmethod
    def _can_approve(store, code, user_id="U_NEW"):
        """Whether ``code`` is redeemable in ``store``.

        Pending codes are stored salted+hashed, so there is no field to compare
        — asking the store to verify it is the only honest check, and it is also
        the exact operation ``hermes pairing approve`` performs. A miss returns
        None without mutating anything, so this is safe on the negative side.
        """
        entry = store.approve_code("slack", code)
        return bool(entry) and entry.get("user_id") == user_id

    def test_secondary_code_lands_in_that_profiles_store(self, hermes_home):
        """The core defect: written to default, read from 'coder'."""
        runner = self._runner_for_dm()
        runner._ensure_pairing_store("default")
        runner._ensure_pairing_store("coder")

        asyncio.run(runner._handle_message(self._dm_event("coder")))
        code = self._issued_code(runner)

        # Negative side first — a miss does not mutate, so order is free.
        assert not self._can_approve(runner.pairing_store, code), (
            "the code was written to the default profile's store"
        )
        assert self._can_approve(runner.pairing_stores["coder"], code), (
            "the code was not written to the profile's own store"
        )

    def test_secondary_code_approves_and_then_authorizes(self, hermes_home):
        """End-to-end: the issued code actually grants access to that profile.

        Pre-fix this was the user-visible symptom — approving the code the bot
        handed out did nothing, because the approval and the check were looking
        at different files.
        """
        runner = self._runner_for_dm()
        runner._ensure_pairing_store("coder")

        asyncio.run(runner._handle_message(self._dm_event("coder")))
        code = self._issued_code(runner)

        assert self._can_approve(runner.pairing_stores["coder"], code)

        source = SessionSource(
            platform=Platform.SLACK, chat_id="D1",
            user_id="U_NEW", chat_type="dm", profile="coder",
        )
        assert runner._is_user_authorized(source) is True

    def test_default_profile_path_is_unchanged(self, hermes_home):
        """The legacy global path must behave exactly as before."""
        runner = self._runner_for_dm()
        runner._ensure_pairing_store("default")

        asyncio.run(runner._handle_message(self._dm_event("default")))
        code = self._issued_code(runner)

        assert self._can_approve(PairingStore(), code), (
            "the default profile stopped writing the legacy global store the "
            "CLI and dashboard read"
        )

    def test_unstamped_source_on_single_profile_gateway_unchanged(self, hermes_home):
        """No multiplexing, no profile: still the global store."""
        runner = self._runner_for_dm(multiplex=False)

        asyncio.run(runner._handle_message(self._dm_event(None)))
        code = self._issued_code(runner)

        assert self._can_approve(runner.pairing_store, code)

    def test_no_store_for_profile_issues_no_code(self, hermes_home):
        """Fail closed: an unserved/failed profile gets no pairing code.

        There is no file an approval could land in, so a code would be a dead
        end — and handing one out under multiplexing would imply a grant path
        that does not exist.
        """
        runner = self._runner_for_dm()
        runner._ensure_pairing_store("default")
        # 'ghost' has no registered store.

        asyncio.run(runner._handle_message(self._dm_event("ghost")))

        assert runner.sent == [], "a pairing code was offered with no store to hold it"
        default_pending = runner.pairing_store._load_json(
            runner.pairing_store._pending_path("slack")
        )
        assert not default_pending, (
            "an unserved profile's pairing code was written to the default store"
        )

    def test_rate_limit_is_recorded_in_the_profiles_own_store(self, hermes_home):
        """The rate limit must count where the next request will read it."""
        runner = self._runner_for_dm()
        runner._ensure_pairing_store("coder")
        store = runner.pairing_stores["coder"]

        # Exhaust the per-platform pending budget so generate_code returns None
        # and the rejection branch records the rate limit.
        for i in range(MAX_PENDING_PER_PLATFORM):
            store.generate_code("slack", f"U_OTHER{i}", "filler")

        asyncio.run(runner._handle_message(self._dm_event("coder")))

        assert runner.sent and "Too many pairing requests" in runner.sent[0]
        assert store._is_rate_limited("slack", "U_NEW") is True, (
            "the rate limit was recorded somewhere the next request won't read"
        )
        assert runner.pairing_store._is_rate_limited("slack", "U_NEW") is False, (
            "the rate limit was written to the default profile's store"
        )
