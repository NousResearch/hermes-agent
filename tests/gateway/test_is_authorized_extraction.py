"""Extraction tests for ``gateway.authz_mixin.is_authorized`` (spec §3).

Two invariants this refactor must hold, on top of every existing
``_is_user_authorized`` test continuing to pass unchanged:

1. Parity: ``GatewayRunner._is_user_authorized`` (the thin wrapper) and the
   module-level pure function it delegates to must agree, for the same
   inputs, across every branch of the decision tree — the wrapper's only job
   is binding ``self``, not altering behavior.
2. Profile independence: ``is_authorized`` must never consult
   ``_HERMES_HOME_OVERRIDE`` (directly or via ``hermes_constants``). Per-chat
   authorization is a process-global decision by design (spec's stated
   non-goal: "fixing" cross-profile authorization into being profile-aware);
   this test proves the pure function structurally cannot regress that,
   since there's nothing profile-aware in it to begin with.
"""

from types import SimpleNamespace

import pytest

import hermes_constants
from gateway.authz_mixin import is_authorized
from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_BOTS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "WECOM_ALLOWED_USERS",
        "WECOM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_runner(*, paired: bool = False, adapter: SimpleNamespace | None = None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: paired)
    runner.adapters = {Platform.WECOM: adapter} if adapter is not None else {}
    runner._profile_adapters = {}
    return runner


# --------------------------------------------------------------------------
# 1. Parity: wrapper vs. pure function, across the decision tree
# --------------------------------------------------------------------------

PARITY_CASES = [
    # (label, source_kwargs, paired, allowed_users_env)
    ("homeassistant-bypass", dict(platform=Platform.HOMEASSISTANT, chat_id="ha"), False, None),
    ("no-user-id-denied", dict(platform=Platform.TELEGRAM, chat_id="1", user_id=None), False, None),
    ("pairing-store-approved", dict(platform=Platform.TELEGRAM, chat_id="1", user_id="42"), True, None),
    ("env-allowlist-match", dict(platform=Platform.TELEGRAM, chat_id="1", user_id="42"), False, "42"),
    ("env-allowlist-miss", dict(platform=Platform.TELEGRAM, chat_id="1", user_id="99"), False, "42"),
    (
        "group-chat-allowlist",
        dict(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id=None),
        False,
        None,
    ),
    (
        "role-authorized",
        dict(platform=Platform.DISCORD, chat_id="1", user_id="7", role_authorized=True),
        False,
        None,
    ),
]


@pytest.mark.parametrize("label,source_kwargs,paired,allowed_users_env", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_wrapper_matches_pure_function(monkeypatch, label, source_kwargs, paired, allowed_users_env):
    if label == "group-chat-allowlist":
        monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100")
    if allowed_users_env is not None:
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", allowed_users_env)

    source = SessionSource(**{"user_name": None, **source_kwargs})
    runner = _make_runner(paired=paired)

    wrapper_result = runner._is_user_authorized(source)
    pure_result = is_authorized(
        source,
        pairing_is_approved=lambda platform_name, uid: paired,
    )
    assert wrapper_result == pure_result, label


def test_wrapper_matches_pure_function_for_adapter_own_policy(monkeypatch):
    """Own-policy adapter branch (WECOM dm_policy=allowlist, no env allowlist)."""
    adapter = SimpleNamespace(
        send=None,
        enforces_own_access_policy=True,
        _dm_policy="allowlist",
        _group_policy="pairing",
    )
    runner = _make_runner(paired=False, adapter=adapter)
    source = SessionSource(
        platform=Platform.WECOM, chat_id="dm-1", chat_type="dm", user_id="anyone", user_name="anyone",
    )

    wrapper_result = runner._is_user_authorized(source)
    pure_result = is_authorized(
        source,
        pairing_is_approved=lambda *_a: False,
        adapter_enforces_own_access_policy=lambda *_a: True,
        adapter_dm_policy=lambda *_a: "allowlist",
    )
    assert wrapper_result is True
    assert wrapper_result == pure_result


def test_wrapper_lazily_defers_pairing_store_lookup_like_original():
    """A branch resolved before the pairing-store check must never touch it.

    Regression guard for the extraction bug caught during this refactor: the
    wrapper originally passed ``self.pairing_store`` as an eagerly-evaluated
    keyword argument, which broke bare runners (built via
    ``object.__new__``) that never set ``pairing_store`` at all — even though
    the request never needed a pairing-store lookup because an earlier
    branch (the chat-scoped group allowlist) already resolved it.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)  # deliberately no .pairing_store
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id=None, user_name=None,
    )
    import os

    os.environ["TELEGRAM_GROUP_ALLOWED_CHATS"] = "-100"
    try:
        assert runner._is_user_authorized(source) is True
    finally:
        del os.environ["TELEGRAM_GROUP_ALLOWED_CHATS"]


# --------------------------------------------------------------------------
# 2. Profile independence: is_authorized must never read _HERMES_HOME_OVERRIDE
# --------------------------------------------------------------------------

def test_is_authorized_never_reads_hermes_home_override(monkeypatch):
    def _boom(*_a, **_kw):
        raise AssertionError(
            "is_authorized must not consult _HERMES_HOME_OVERRIDE — "
            "authorization is process-global by design, not profile-scoped"
        )

    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", _boom)

    token = hermes_constants.set_hermes_home_override("/tmp/some-other-profile-home")
    try:
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="42", user_name=None,
        )
        result = is_authorized(source, pairing_is_approved=lambda *_a: True)
        assert result is True
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_is_authorized_result_identical_across_hermes_home_overrides(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="42", user_name=None,
    )

    results = []
    for override in (None, "/tmp/profile-a-home", "/tmp/profile-b-home"):
        token = hermes_constants.set_hermes_home_override(override)
        try:
            results.append(is_authorized(source, pairing_is_approved=lambda *_a: False))
        finally:
            hermes_constants.reset_hermes_home_override(token)

    assert results == [True, True, True]
