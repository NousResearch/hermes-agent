"""Credential isolation for the QQ (qqbot) gateway adapter.

Covers the multiplex credential-collision class (same class as the
WeChat/weixin adapter tracked in #59662): the QQ adapter resolves its ``QQ_*``
settings through the active profile secret scope rather than raw ``os.getenv``,
so a secondary profile whose secret lives in its own ``.env`` (installed as an
isolated scope, not into ``os.environ``) does not fall back to the
default/primary profile's value.

Also guards the primary/active profile: it is constructed without a scope and
legitimately owns ``os.environ``, so the resolver must fall back to
``os.environ`` there (not fail closed) even when multiplexing is active —
otherwise the active profile's adapter would raise ``UnscopedSecretError`` on
construction and fail to start.
"""
import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig
from gateway.platforms.qqbot.adapter import QQAdapter


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _make_adapter(extra=None):
    return QQAdapter(PlatformConfig(enabled=True, extra=extra or {}))


class TestQQCredentialScope:
    def test_credentials_read_scope_not_environ(self, monkeypatch):
        # os.environ holds another profile's values; the scoped values must win.
        monkeypatch.setenv("QQ_APP_ID", "global-app")
        monkeypatch.setenv("QQ_CLIENT_SECRET", "global-secret")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(
            {"QQ_APP_ID": "profileA-app", "QQ_CLIENT_SECRET": "profileA-secret"}
        )
        try:
            adapter = _make_adapter()
        finally:
            ss.reset_secret_scope(tok)
        assert adapter._app_id == "profileA-app"
        assert adapter._client_secret == "profileA-secret"

    def test_two_profiles_isolated(self):
        ss.set_multiplex_active(True)
        tok_a = ss.set_secret_scope({"QQ_CLIENT_SECRET": "secret-A"})
        try:
            a = _make_adapter()
        finally:
            ss.reset_secret_scope(tok_a)
        tok_b = ss.set_secret_scope({"QQ_CLIENT_SECRET": "secret-B"})
        try:
            b = _make_adapter()
        finally:
            ss.reset_secret_scope(tok_b)
        assert a._client_secret == "secret-A"
        assert b._client_secret == "secret-B"

    def test_single_profile_still_reads_environ(self, monkeypatch):
        # No scope + multiplex inactive (default single-profile deployment):
        # legacy os.environ behavior is preserved — no regression.
        monkeypatch.setenv("QQ_CLIENT_SECRET", "legacy-secret")
        adapter = _make_adapter()
        assert adapter._client_secret == "legacy-secret"

    def test_active_profile_no_scope_reads_environ_without_raising(self, monkeypatch):
        # The primary/active profile is built with NO scope while multiplexing
        # is active. A bare get_secret() would fail closed (UnscopedSecretError)
        # and break its startup; the resolver must fall back to os.environ.
        monkeypatch.setenv("QQ_APP_ID", "primary-app")
        monkeypatch.setenv("QQ_CLIENT_SECRET", "primary-secret")
        ss.set_multiplex_active(True)
        assert ss.current_secret_scope() is None  # no scope installed
        adapter = _make_adapter()  # must not raise
        assert adapter._app_id == "primary-app"
        assert adapter._client_secret == "primary-secret"

    def test_explicit_config_extra_takes_precedence(self, monkeypatch):
        # An explicit value in config.extra still wins over env/scope.
        monkeypatch.setenv("QQ_CLIENT_SECRET", "env-secret")
        adapter = _make_adapter(extra={"client_secret": "explicit"})
        assert adapter._client_secret == "explicit"


class TestQQSttConfigScope:
    def test_stt_api_key_reads_scope(self, monkeypatch):
        monkeypatch.setenv("QQ_STT_API_KEY", "global-stt-key")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"QQ_STT_API_KEY": "profileA-stt-key"})
        try:
            adapter = _make_adapter()
            stt = adapter._resolve_stt_config()
        finally:
            ss.reset_secret_scope(tok)
        assert stt is not None
        assert stt["api_key"] == "profileA-stt-key"
