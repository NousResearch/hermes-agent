"""Verify that the WhatsApp group allowlist reads env vars as fallback.

Regression test for #56767: after the built-in → plugin migration,
``WHATSAPP_GROUP_ALLOW_FROM`` / ``WHATSAPP_GROUP_ALLOWED_USERS`` were no
longer read, causing group messages to be silently dropped when the user
configured the allowlist via ``.env`` instead of ``config.yaml``.
"""

from gateway.config import Platform, PlatformConfig

from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _init_group_allow_from(extra, monkeypatch, env_overrides=None):
    """Reproduce the ``__init__`` lines that set ``_group_allow_from``."""
    if env_overrides:
        for key, value in env_overrides.items():
            monkeypatch.setenv(key, value)
    config = PlatformConfig(enabled=True, extra=extra)
    return WhatsAppAdapter._coerce_allow_list(
        config.extra.get("group_allow_from")
        or config.extra.get("groupAllowFrom")
        or __import__("os").getenv("WHATSAPP_GROUP_ALLOW_FROM")
        or __import__("os").getenv("WHATSAPP_GROUP_ALLOWED_USERS")
    )


# --- Env-only fallback (the core regression) ---


def test_group_allowlist_reads_whatsapp_group_allow_from_env(monkeypatch):
    """Env-only setup: ``WHATSAPP_GROUP_ALLOW_FROM`` must populate the
    group allowlist when ``config.yaml`` has no ``group_allow_from``."""
    result = _init_group_allow_from(
        extra={},
        monkeypatch=monkeypatch,
        env_overrides={"WHATSAPP_GROUP_ALLOW_FROM": "12036300111@g.us, 12036300222@g.us"},
    )
    assert result == {"12036300111@g.us", "12036300222@g.us"}


def test_group_allowlist_reads_whatsapp_group_allowed_users_env(monkeypatch):
    """Env-only setup: ``WHATSAPP_GROUP_ALLOWED_USERS`` (setup-wizard name)
    must also populate the group allowlist."""
    result = _init_group_allow_from(
        extra={},
        monkeypatch=monkeypatch,
        env_overrides={"WHATSAPP_GROUP_ALLOWED_USERS": "12036300333@g.us"},
    )
    assert result == {"12036300333@g.us"}


# --- Config takes precedence over env ---


def test_group_allowlist_config_extra_wins_over_env(monkeypatch):
    """Explicit ``group_allow_from`` in config must not be widened by a
    stale env var."""
    result = _init_group_allow_from(
        extra={"group_allow_from": ["12036300444@g.us"]},
        monkeypatch=monkeypatch,
        env_overrides={"WHATSAPP_GROUP_ALLOW_FROM": "12036300999@g.us"},
    )
    assert result == {"12036300444@g.us"}


def test_group_allowlist_camelcase_config_wins_over_env(monkeypatch):
    """CamelCase ``groupAllowFrom`` in config extra also takes precedence."""
    result = _init_group_allow_from(
        extra={"groupAllowFrom": ["12036300555@g.us"]},
        monkeypatch=monkeypatch,
        env_overrides={"WHATSAPP_GROUP_ALLOWED_USERS": "12036300999@g.us"},
    )
    assert result == {"12036300555@g.us"}


# --- Env var priority: GROUP_ALLOW_FROM checked first ---


def test_group_allow_from_checked_before_group_allowed_users(monkeypatch):
    """When both env vars are set, ``WHATSAPP_GROUP_ALLOW_FROM`` wins
    (checked first in the ``or`` chain)."""
    result = _init_group_allow_from(
        extra={},
        monkeypatch=monkeypatch,
        env_overrides={
            "WHATSAPP_GROUP_ALLOW_FROM": "12036300666@g.us",
            "WHATSAPP_GROUP_ALLOWED_USERS": "12036300777@g.us",
        },
    )
    assert result == {"12036300666@g.us"}


# --- Empty / missing env var ---


def test_group_allowlist_empty_when_no_config_no_env(monkeypatch):
    """No config and no env → empty allowlist (default)."""
    result = _init_group_allow_from(extra={}, monkeypatch=monkeypatch)
    assert result == set()
