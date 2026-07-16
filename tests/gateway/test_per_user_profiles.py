"""Tests for per-user profile isolation (gateway.per_user_profiles).

Covers the four moving pieces:
  * derivation      — gateway/user_profiles.derive_user_profile_name / is_user_profile_name
  * provisioning    — gateway/user_profiles.ensure_user_profile (idempotent, race-safe, strips .env)
  * config          — GatewayConfig round-trips the three keys (top-level + nested)
  * wiring          — session-key namespacing, home resolution + provisioning, scope selection,
                      and the off-by-default passthrough.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.session import SessionSource, build_session_key, SessionStore
from gateway.run import GatewayRunner, _profile_home_only_scope, _profile_runtime_scope
from gateway import user_profiles
from gateway.config import GatewayConfig, Platform


def _src(user_id="1271274566", platform="telegram", **kw):
    return SessionSource(
        platform=MagicMock(value=platform),
        chat_id=kw.pop("chat_id", user_id),
        chat_type=kw.pop("chat_type", "dm"),
        user_id=user_id,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #
class TestDerivation:
    def test_basic_name(self):
        assert user_profiles.derive_user_profile_name(_src("1271274566")) == "u-telegram-1271274566"

    def test_prefix_configurable(self):
        assert user_profiles.derive_user_profile_name(_src("42"), prefix="usr") == "usr-telegram-42"

    def test_no_user_id_returns_none(self):
        s = SessionSource(platform=MagicMock(value="telegram"), chat_id="c", user_id=None)
        assert user_profiles.derive_user_profile_name(s) is None

    def test_user_id_alt_preferred(self):
        s = _src("rawid", user_id_alt="altid")
        assert user_profiles.derive_user_profile_name(s) == "u-telegram-altid"

    def test_platform_namespacing(self):
        tg = user_profiles.derive_user_profile_name(_src("777", platform="telegram"))
        sc = user_profiles.derive_user_profile_name(_src("777", platform="spacechat"))
        assert tg != sc
        assert tg == "u-telegram-777" and sc == "u-spacechat-777"

    def test_deterministic(self):
        a = user_profiles.derive_user_profile_name(_src("abc"))
        b = user_profiles.derive_user_profile_name(_src("abc"))
        assert a == b

    def test_nonconforming_uid_hashed(self):
        # An email-like / uppercase uid can't sit in a profile-id verbatim.
        name = user_profiles.derive_user_profile_name(_src("Alice@Example.COM"))
        assert name is not None
        assert name.startswith("u-telegram-")
        # falls back to a 16-hex digest, deterministic
        assert name == user_profiles.derive_user_profile_name(_src("Alice@Example.COM"))
        from hermes_cli.profiles import validate_profile_name
        validate_profile_name(name)  # must not raise

    def test_long_uid_hashed_and_valid(self):
        name = user_profiles.derive_user_profile_name(_src("x" * 200))
        assert name is not None and len(name) <= 64
        from hermes_cli.profiles import validate_profile_name
        validate_profile_name(name)

    def test_distinct_raw_ids_that_sanitize_alike_stay_distinct(self):
        a = user_profiles.derive_user_profile_name(_src("a.b"))
        b = user_profiles.derive_user_profile_name(_src("ab"))
        # "a.b" sanitizes to "ab" → must hash so it doesn't collide with "ab"
        assert a != b

    def test_derived_name_is_valid_profile_id(self):
        from hermes_cli.profiles import validate_profile_name
        validate_profile_name(user_profiles.derive_user_profile_name(_src("1271274566")))

    def test_is_user_profile_name(self):
        assert user_profiles.is_user_profile_name("u-telegram-1")
        assert user_profiles.is_user_profile_name("usr-x-1", prefix="usr")
        assert not user_profiles.is_user_profile_name("default")
        assert not user_profiles.is_user_profile_name("coder")
        assert not user_profiles.is_user_profile_name(None)


# --------------------------------------------------------------------------- #
# Provisioning
# --------------------------------------------------------------------------- #
class TestProvisioning:
    def test_existing_profile_fast_path(self):
        with patch("hermes_cli.profiles.profile_exists", return_value=True) as pe, \
             patch("hermes_cli.profiles.create_profile") as cp:
            assert user_profiles.ensure_user_profile("u-telegram-1") is False
            cp.assert_not_called()

    def test_creates_from_template(self, tmp_path):
        calls = {}

        def fake_create(name, clone_from=None, clone_config=False, no_alias=False, **kw):
            calls["name"] = name
            calls["clone_from"] = clone_from
            calls["clone_config"] = clone_config
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
            return tmp_path / name

        with patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.create_profile", side_effect=fake_create), \
             patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: tmp_path / n):
            created = user_profiles.ensure_user_profile("u-telegram-1", template="corp")
        assert created is True
        assert calls == {"name": "u-telegram-1", "clone_from": "corp", "clone_config": True}

    def test_template_defaults_to_active(self):
        with patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.create_profile") as cp, \
             patch("hermes_cli.profiles.get_profile_dir", return_value=Path("/nope")):
            user_profiles.ensure_user_profile("u-telegram-1", template=None)
            assert cp.call_args.kwargs["clone_from"] == "default"

    def test_strips_seeded_env(self, tmp_path):
        prof = tmp_path / "u-telegram-1"
        prof.mkdir()
        (prof / ".env").write_text("OPENAI_API_KEY=leaked\n")

        def fake_create(name, **kw):
            return prof

        with patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.create_profile", side_effect=fake_create), \
             patch("hermes_cli.profiles.get_profile_dir", return_value=prof):
            user_profiles.ensure_user_profile("u-telegram-1")
        assert not (prof / ".env").exists()  # copied secret removed

    def test_create_race_swallows_fileexists(self):
        with patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.create_profile", side_effect=FileExistsError), \
             patch("hermes_cli.profiles.get_profile_dir", return_value=Path("/nope")):
            assert user_profiles.ensure_user_profile("u-telegram-1") is False


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_defaults_off(self):
        cfg = GatewayConfig.from_dict({})
        assert cfg.per_user_profiles is False
        assert cfg.per_user_profile_template == ""
        assert cfg.per_user_profile_prefix == "u"

    def test_top_level(self):
        cfg = GatewayConfig.from_dict({
            "per_user_profiles": True,
            "per_user_profile_template": "corp",
            "per_user_profile_prefix": "usr",
        })
        assert cfg.per_user_profiles is True
        assert cfg.per_user_profile_template == "corp"
        assert cfg.per_user_profile_prefix == "usr"

    def test_nested_gateway_form(self):
        cfg = GatewayConfig.from_dict({"gateway": {"per_user_profiles": True}})
        assert cfg.per_user_profiles is True

    def test_blank_prefix_falls_back(self):
        cfg = GatewayConfig.from_dict({"per_user_profiles": True, "per_user_profile_prefix": "  "})
        assert cfg.per_user_profile_prefix == "u"

    def test_to_dict_roundtrip(self):
        cfg = GatewayConfig.from_dict({"per_user_profiles": True, "per_user_profile_template": "corp"})
        d = cfg.to_dict()
        assert d["per_user_profiles"] is True
        assert d["per_user_profile_template"] == "corp"
        assert GatewayConfig.from_dict(d).per_user_profiles is True


# --------------------------------------------------------------------------- #
# Session-key namespacing
# --------------------------------------------------------------------------- #
class TestSessionKeyIsolation:
    def _store(self, **cfg):
        store = SessionStore.__new__(SessionStore)
        store.config = SimpleNamespace(
            multiplex_profiles=cfg.get("multiplex_profiles", False),
            per_user_profiles=cfg.get("per_user_profiles", False),
        )
        return store

    def test_off_returns_none(self):
        store = self._store()
        assert store._resolve_profile_for_key(_src("1", **{})) is None

    def test_per_user_reads_stamped_profile(self):
        store = self._store(per_user_profiles=True)
        s = _src("1")
        s.profile = "u-telegram-1"
        assert store._resolve_profile_for_key(s) == "u-telegram-1"

    def test_per_user_no_profile_stays_shared(self):
        # per-user mode + unstamped source (no user_id) → None (shared agent:main),
        # NOT the active profile (that's the multiplex-only fallback).
        store = self._store(per_user_profiles=True)
        s = SessionSource(platform=MagicMock(value="telegram"), chat_id="c", user_id=None)
        assert store._resolve_profile_for_key(s) is None

    def test_two_users_get_distinct_session_keys(self):
        a = build_session_key(_src("111"), profile="u-telegram-111")
        b = build_session_key(_src("222"), profile="u-telegram-222")
        assert a != b
        assert a.startswith("agent:u-telegram-111:")
        assert b.startswith("agent:u-telegram-222:")

    def test_default_namespace_unchanged_when_off(self):
        # profile=None → legacy agent:main namespace, byte-identical to before.
        key = build_session_key(_src("111"), profile=None)
        assert key.startswith("agent:main:")


# --------------------------------------------------------------------------- #
# Runner wiring: derivation, provisioning-on-resolve, scope selection
# --------------------------------------------------------------------------- #
@pytest.fixture
def runner():
    r = MagicMock(spec=GatewayRunner)
    r._profile_name_for_source = GatewayRunner._profile_name_for_source.__get__(r)
    r._resolve_profile_home_for_source = GatewayRunner._resolve_profile_home_for_source.__get__(r)
    r._profile_scope_for_source = GatewayRunner._profile_scope_for_source.__get__(r)
    return r


class TestRunnerWiring:
    def test_profile_name_derived_when_no_route(self, runner):
        runner.config = SimpleNamespace(
            multiplex_profiles=False, per_user_profiles=True,
            profile_routes=[], per_user_profile_prefix="u",
        )
        assert runner._profile_name_for_source(_src("1271274566")) == "u-telegram-1271274566"

    def test_profile_name_none_when_all_off(self, runner):
        runner.config = SimpleNamespace(
            multiplex_profiles=False, per_user_profiles=False, profile_routes=[],
        )
        assert runner._profile_name_for_source(_src("1")) is None

    def test_route_wins_over_per_user(self, runner):
        route = SimpleNamespace(profile="team")
        runner.config = SimpleNamespace(
            multiplex_profiles=False, per_user_profiles=True,
            profile_routes=[route], per_user_profile_prefix="u",
        )
        with patch("gateway.profile_routing.match_profile_route", return_value=route):
            assert runner._profile_name_for_source(_src("1")) == "team"

    def test_home_resolution_provisions_missing_user_profile(self, runner):
        runner.config = SimpleNamespace(
            multiplex_profiles=False, per_user_profiles=True,
            profile_routes=[], per_user_profile_prefix="u",
            per_user_profile_template="",
        )
        s = _src("1")
        s.profile = "u-telegram-1"
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.get_profile_dir", return_value=Path("/h/profiles/u-telegram-1")), \
             patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("gateway.user_profiles.ensure_user_profile", return_value=True) as ep:
            result = runner._resolve_profile_home_for_source(s)
        ep.assert_called_once()
        assert result == Path("/h/profiles/u-telegram-1")

    def test_home_resolution_falls_back_for_nonuser_missing_profile(self, runner):
        # An explicit but missing operator profile (not a u- name) still falls back.
        runner.config = SimpleNamespace(
            multiplex_profiles=False, per_user_profiles=True,
            profile_routes=[], per_user_profile_prefix="u",
        )
        s = _src("1")
        s.profile = "coder"
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.get_profile_dir", return_value=Path("/h/profiles/coder")), \
             patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("hermes_constants.get_hermes_home", return_value=Path("/h")), \
             patch("gateway.user_profiles.ensure_user_profile") as ep:
            result = runner._resolve_profile_home_for_source(s)
        ep.assert_not_called()
        assert result == Path("/h")

    def test_scope_nullcontext_when_off(self, runner):
        runner.config = SimpleNamespace(multiplex_profiles=False, per_user_profiles=False)
        from contextlib import nullcontext
        assert isinstance(runner._profile_scope_for_source(_src("1")), type(nullcontext()))

    def test_scope_home_only_for_per_user(self, runner):
        runner.config = SimpleNamespace(multiplex_profiles=False, per_user_profiles=True)
        runner._resolve_profile_home_for_source = MagicMock(return_value=Path("/h/profiles/u-telegram-1"))
        with patch("gateway.run._profile_home_only_scope") as home_only, \
             patch("gateway.run._profile_runtime_scope") as full:
            runner._profile_scope_for_source(_src("1"))
        home_only.assert_called_once_with(Path("/h/profiles/u-telegram-1"))
        full.assert_not_called()

    def test_scope_full_for_multiplex_precedence(self, runner):
        runner.config = SimpleNamespace(multiplex_profiles=True, per_user_profiles=True)
        runner._resolve_profile_home_for_source = MagicMock(return_value=Path("/h/profiles/x"))
        with patch("gateway.run._profile_home_only_scope") as home_only, \
             patch("gateway.run._profile_runtime_scope") as full:
            runner._profile_scope_for_source(_src("1"))
        full.assert_called_once_with(Path("/h/profiles/x"))
        home_only.assert_not_called()


# --------------------------------------------------------------------------- #
# Home-only scope keeps secrets shared
# --------------------------------------------------------------------------- #
class TestHomeOnlyScopeSecrets:
    def test_home_override_set_secrets_untouched(self, tmp_path):
        from hermes_constants import get_hermes_home
        from agent.secret_scope import current_secret_scope, is_multiplex_active
        assert current_secret_scope() is None
        with _profile_home_only_scope(tmp_path):
            assert get_hermes_home() == tmp_path
            # No secret scope installed → get_secret keeps reading os.environ.
            assert current_secret_scope() is None
            assert is_multiplex_active() is False
        # override reset on exit
        assert get_hermes_home() != tmp_path


# --------------------------------------------------------------------------- #
# Collision hardening (case-fold / whitespace / punctuation)
# --------------------------------------------------------------------------- #
class TestCollisionHardening:
    def test_case_distinct_ids_do_not_collide(self):
        # A username platform where 'Bob' and 'bob' are distinct users.
        bob = user_profiles.derive_user_profile_name(_src("Bob"))
        low = user_profiles.derive_user_profile_name(_src("bob"))
        assert bob != low
        assert low == "u-telegram-bob"          # lossless → readable
        assert bob.startswith("u-telegram-")     # uppercase → hashed
        from hermes_cli.profiles import validate_profile_name
        validate_profile_name(bob)

    def test_whitespace_only_uid_is_none(self):
        for ws in ("\t", " ", "\n", "  \r\n "):
            assert user_profiles.derive_user_profile_name(_src(ws)) is None

    def test_punctuation_only_uid_hashed_not_degenerate(self):
        name = user_profiles.derive_user_profile_name(_src("!!!"))
        assert name is not None
        assert name != "u-telegram-"             # never a degenerate shared id
        assert name.startswith("u-telegram-")
        from hermes_cli.profiles import validate_profile_name
        validate_profile_name(name)

    def test_lowercase_username_lossless(self):
        assert user_profiles.derive_user_profile_name(_src("alice_99")) == "u-telegram-alice_99"


# --------------------------------------------------------------------------- #
# Provisioning: partial-failure cleanup + lock eviction
# --------------------------------------------------------------------------- #
class TestProvisioningHardening:
    def test_partial_create_failure_cleans_up_and_reraises(self, tmp_path):
        prof = tmp_path / "u-telegram-1"
        prof.mkdir()  # simulate a half-built dir left by a failed copytree
        with patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.create_profile", side_effect=OSError("disk full")), \
             patch("hermes_cli.profiles.get_profile_dir", return_value=prof), \
             patch("shutil.rmtree") as rmtree:
            with pytest.raises(OSError):
                user_profiles.ensure_user_profile("u-telegram-1")
        rmtree.assert_called_once()  # partial dir removed so it isn't treated as provisioned

    def test_lock_evicted_after_provision(self):
        user_profiles._provision_locks.pop("u-telegram-ev", None)
        with patch("hermes_cli.profiles.profile_exists", return_value=False), \
             patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
             patch("hermes_cli.profiles.create_profile"), \
             patch("hermes_cli.profiles.get_profile_dir", return_value=Path("/nope")):
            user_profiles.ensure_user_profile("u-telegram-ev")
        assert "u-telegram-ev" not in user_profiles._provision_locks


# --------------------------------------------------------------------------- #
# Adapter resolution (the reply-delivery blocker) + session recovery
# --------------------------------------------------------------------------- #
class TestAdapterAndRecovery:
    def test_authorization_adapter_falls_through_to_shared_bot_for_user_profile(self):
        # Under per_user_profiles the derived u- profile has no secondary adapter;
        # resolution must fall through to the single shared bot adapter, not None
        # (else every reply is dropped — the blocker).
        runner = MagicMock(spec=GatewayRunner)
        runner._authorization_adapter = GatewayRunner._authorization_adapter.__get__(runner)
        runner._adapter_for_source = GatewayRunner._adapter_for_source.__get__(runner)
        sentinel = object()
        runner.adapters = {Platform.TELEGRAM: sentinel}
        runner._profile_adapters = {}
        runner.config = SimpleNamespace(per_user_profiles=True, per_user_profile_prefix="u")

        s = SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="1",
                          profile="u-telegram-1")
        assert runner._adapter_for_source(s) is sentinel

    def test_authorization_adapter_still_fails_closed_for_unknown_multiplex_profile(self):
        # A non-u- secondary profile with no registry entry must still fail closed.
        runner = MagicMock(spec=GatewayRunner)
        runner._authorization_adapter = GatewayRunner._authorization_adapter.__get__(runner)
        runner.adapters = {Platform.TELEGRAM: object()}
        runner._profile_adapters = {}
        runner.config = SimpleNamespace(per_user_profiles=True, per_user_profile_prefix="u")
        assert runner._authorization_adapter(Platform.TELEGRAM, "coder") is None

    def test_recovery_allows_same_user_differently_keyed_row(self):
        store = SessionStore.__new__(SessionStore)
        store.config = SimpleNamespace(multiplex_profiles=False)
        store._recovered_row_allowed_for_active_profile = \
            SessionStore._recovered_row_allowed_for_active_profile.__get__(store)
        req = "agent:u-telegram-1:telegram:dm:1"
        same_user = {"session_key": "agent:u-telegram-1:telegram:dm:1:topic"}
        other_user = {"session_key": "agent:u-telegram-2:telegram:dm:2"}
        assert store._recovered_row_allowed_for_active_profile(
            requested_session_key=req, recovered=same_user) is True
        assert store._recovered_row_allowed_for_active_profile(
            requested_session_key=req, recovered=other_user) is False


# --------------------------------------------------------------------------- #
# multiplex + per_user both on: per-user derivation is suppressed
# --------------------------------------------------------------------------- #
class TestMultiplexPrecedence:
    def test_per_user_derivation_suppressed_under_multiplex(self, runner):
        runner.config = SimpleNamespace(
            multiplex_profiles=True, per_user_profiles=True,
            profile_routes=[], per_user_profile_prefix="u",
        )
        # No route configured → multiplex would stamp via its adapter handler, not
        # here; per-user derivation must NOT fire, so this returns None.
        assert runner._profile_name_for_source(_src("1271274566")) is None
