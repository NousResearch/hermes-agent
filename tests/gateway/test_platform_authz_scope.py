"""Platform authorization reads must honor the per-profile secret scope (#93522).

Under ``gateway.multiplex_profiles`` every secondary profile is constructed
inside ``_profile_runtime_scope`` and its ``.env`` lives in that scope —
``gateway/run.py`` explicitly does NOT mutate ``os.environ`` with it. Raw
``os.getenv`` authorization reads therefore (a) silently miss the secondary
profile's own allowlists (fail-closed no-replies) and (b) leak the default
profile's ``GATEWAY_ALLOW_ALL_USERS`` / allowlists into secondary profiles
(fail-open admissions).

Covers the scoped env helpers of weixin / yuanbao / signal / wecom plus the
``_open_dm_opted_in`` gates built on them. Canonical shape reference:
QQ's ``_resolve_qq_secret``.
"""

import contextlib
import os

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope


@contextlib.contextmanager
def _scope(secrets):
    token = set_secret_scope(secrets)
    try:
        yield
    finally:
        reset_secret_scope(token)


@pytest.fixture
def profile_scope():
    """Install a secondary-profile secret scope; os.environ stays 'default'."""
    with _scope(
        {
            "WEIXIN_DM_POLICY": "allowlist",
            "WEIXIN_ALLOWED_USERS": "wx-alice",
            "WEIXIN_ALLOW_ALL_USERS": "",
            "YUANBAO_DM_POLICY": "open",
            "SIGNAL_ALLOWED_USERS": "sig-bob",
            "WECOM_DM_POLICY": "allowlist",
            "WECOM_ALLOWED_USERS": "wecom-carol",
        }
    ):
        yield


# ─────────────────────────────────────────────────────────────────────
# Scoped helpers: read the scope, never leak os.environ past it
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "module_path,helper_name,var",
    [
        ("gateway.platforms.weixin", "_wx_secret", "WEIXIN_ALLOWED_USERS"),
        ("gateway.platforms.yuanbao", "_yb_secret", "YUANBAO_DM_POLICY"),
        ("gateway.platforms.signal", "_sig_secret", "SIGNAL_ALLOWED_USERS"),
        ("plugins.platforms.wecom.adapter", "_get_scoped_secret", "WECOM_ALLOWED_USERS"),
    ],
)
def test_helper_reads_profile_scope_first(module_path, helper_name, var, monkeypatch):
    """Scope value wins over anything in os.environ."""
    monkeypatch.setenv(var, "FROM-DEFAULT-ENV")
    module = __import__(module_path, fromlist=[helper_name])
    helper = getattr(module, helper_name)

    with _scope({var: "from-profile-scope"}):
        assert helper(var, "") == "from-profile-scope"


@pytest.fixture
def multiplex_on(monkeypatch):
    """Simulate a multiplexed deployment: scoped misses fail closed.

    With multiplexing off, a scope miss deliberately falls through to
    os.environ (the cron-overlay contract) — the cross-profile leak only
    exists while the multiplexer is active, which is exactly the #93522
    scenario.
    """
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)


@pytest.mark.parametrize(
    "module_path,helper_name",
    [
        ("gateway.platforms.weixin", "_wx_secret"),
        ("gateway.platforms.yuanbao", "_yb_secret"),
        ("gateway.platforms.signal", "_sig_secret"),
        ("plugins.platforms.wecom.adapter", "_get_scoped_secret"),
    ],
)
def test_helper_does_not_leak_default_env_into_scoped_miss(
    module_path, helper_name, multiplex_on, monkeypatch
):
    """Scoped miss must return the default, NOT fall through to os.environ.

    This is the fail-closed half of #93522: the default profile's
    ``*_ALLOW_ALL_USERS=true`` in process env must not answer a secondary
    profile's gate.
    """
    monkeypatch.setenv("WEIXIN_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("YUANBAO_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "*")
    monkeypatch.setenv("WECOM_ALLOW_ALL_USERS", "true")

    module = __import__(module_path, fromlist=[helper_name])
    helper = getattr(module, helper_name)

    with _scope({}):  # scope installed, keys absent
        assert helper("WEIXIN_ALLOW_ALL_USERS", "") == ""
        assert helper("YUANBAO_ALLOW_ALL_USERS", "") == ""
        assert helper("WECOM_ALLOW_ALL_USERS", "") == ""
        # signal's "*" default means open at adapter level — the scoped
        # read must still not see the unscoped value:
        assert helper("SIGNAL_ALLOWED_USERS", "restricted-default") == "restricted-default"


@pytest.mark.parametrize(
    "module_path,helper_name,var",
    [
        ("gateway.platforms.weixin", "_wx_secret", "WEIXIN_ALLOWED_USERS"),
        ("gateway.platforms.yuanbao", "_yb_secret", "YUANBAO_DM_POLICY"),
        ("gateway.platforms.signal", "_sig_secret", "SIGNAL_ALLOWED_USERS"),
        ("plugins.platforms.wecom.adapter", "_get_scoped_secret", "WECOM_ALLOWED_USERS"),
    ],
)
def test_helper_falls_back_to_environ_without_scope(module_path, helper_name, var, monkeypatch):
    """Single-profile (no scope installed): os.environ remains authoritative."""
    monkeypatch.setenv(var, "legacy-env-value")
    module = __import__(module_path, fromlist=[helper_name])
    helper = getattr(module, helper_name)
    assert helper(var, "") == "legacy-env-value"


# ─────────────────────────────────────────────────────────────────────
# Admission gates built on the helpers
# ─────────────────────────────────────────────────────────────────────


def test_weixin_open_gate_ignores_default_env_under_scope(profile_scope, multiplex_on, monkeypatch):
    """Default profile sets GATEWAY_ALLOW_ALL_USERS=true; secondary scope does
    NOT carry it — the gate must stay closed."""
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    from gateway.platforms.weixin import WeixinAdapter

    adapter = WeixinAdapter.__new__(WeixinAdapter)
    assert adapter._open_dm_opted_in() is False


def test_weixin_open_gate_honors_scope_opt_in():
    from gateway.platforms.weixin import WeixinAdapter

    with _scope({"WEIXIN_ALLOW_ALL_USERS": "yes"}):
        adapter = WeixinAdapter.__new__(WeixinAdapter)
        assert adapter._open_dm_opted_in() is True


def test_yuanbao_access_policy_gate_ignores_default_env(profile_scope, multiplex_on, monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    from gateway.platforms.yuanbao import AccessPolicy

    policy = AccessPolicy(
        dm_policy="open", dm_allow_from=[], group_policy="pairing", group_allow_from=[]
    )
    assert policy._open_dm_opted_in() is False


def test_wecom_open_gate_ignores_default_env_under_scope(profile_scope, multiplex_on, monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter.__new__(WeComAdapter)
    assert adapter._open_dm_opted_in() is False


def test_startup_guard_uses_scoped_gateway_flag(multiplex_on):
    """_own_policy_open_startup_violation must consult the scope for
    GATEWAY_ALLOW_ALL_USERS, not raw os.environ (#93522)."""
    from gateway.platforms.base import Platform
    from gateway.run import _own_policy_open_startup_violation

    class _PlatformConfig:
        enabled = True
        extra = {"dm_policy": "open"}

    class _Config:
        platforms = {Platform.YUANBAO: _PlatformConfig()}

    cfg = _Config()
    with _scope({"GATEWAY_ALLOW_ALL_USERS": "true"}):
        # Scope says opted-in even though os.environ doesn't -> no violation.
        assert _own_policy_open_startup_violation(cfg) is None

    with _scope({}):
        # Scope lacks the opt-in while os.environ has it -> violation (the
        # default profile's flag must not satisfy this profile's check).
        monkey_env_backup = os.environ.get("GATEWAY_ALLOW_ALL_USERS")
        os.environ["GATEWAY_ALLOW_ALL_USERS"] = "true"
        try:
            reason = _own_policy_open_startup_violation(cfg)
            assert reason is not None and "yuanbao" in reason.lower()
        finally:
            if monkey_env_backup is None:
                os.environ.pop("GATEWAY_ALLOW_ALL_USERS", None)
            else:
                os.environ["GATEWAY_ALLOW_ALL_USERS"] = monkey_env_backup
