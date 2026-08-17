"""Weixin adapter secret-scope regression tests.

The adapter's WEIXIN_* credential reads must follow the Slack pattern
(#59739): under multiplexing a SCOPED miss is authoritative (no borrow from
``os.environ`` — that would be a cross-profile leak), while an UNSCOPED read
(default-profile startup/send path) falls back to ``os.environ``, which is
that profile's own value, instead of raising ``UnscopedSecretError``.
"""

import pytest

from agent import secret_scope
from gateway.config import PlatformConfig
from gateway.platforms.weixin import WeixinAdapter, _wx_secret


@pytest.fixture()
def multiplex_on():
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(previous)


class TestWxSecretHelper:
    def test_scoped_read_uses_scope_value(self, multiplex_on, monkeypatch):
        monkeypatch.setenv("WEIXIN_TOKEN", "default-profile-token")
        token = secret_scope.set_secret_scope({"WEIXIN_TOKEN": "scoped-token"})
        try:
            assert _wx_secret("WEIXIN_TOKEN") == "scoped-token"
        finally:
            secret_scope.reset_secret_scope(token)

    def test_scoped_miss_does_not_borrow_environ(self, multiplex_on, monkeypatch):
        """A secondary profile without WEIXIN_TOKEN must NOT inherit the
        default profile's process-env token."""
        monkeypatch.setenv("WEIXIN_TOKEN", "default-profile-token")
        token = secret_scope.set_secret_scope({"OTHER_KEY": "x"})
        try:
            assert _wx_secret("WEIXIN_TOKEN", "") == ""
            assert _wx_secret("WEIXIN_TOKEN") is None
        finally:
            secret_scope.reset_secret_scope(token)

    def test_unscoped_read_falls_back_to_environ(self, multiplex_on, monkeypatch):
        """The default profile's adapter runs unscoped under multiplexing;
        os.environ is its own value — fall back instead of raising."""
        monkeypatch.setenv("WEIXIN_TOKEN", "default-profile-token")
        token = secret_scope.set_secret_scope(None)
        try:
            assert _wx_secret("WEIXIN_TOKEN") == "default-profile-token"
        finally:
            secret_scope.reset_secret_scope(token)


class TestWeixinAdapterConstructionScope:
    def test_multiplex_scoped_construction_reads_scope_not_environ(
        self, multiplex_on, monkeypatch
    ):
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "env-account")
        monkeypatch.setenv("WEIXIN_TOKEN", "env-token")
        token = secret_scope.set_secret_scope(
            {
                "WEIXIN_ACCOUNT_ID": "scoped-account",
                "WEIXIN_TOKEN": "scoped-token",
            }
        )
        try:
            adapter = WeixinAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._account_id == "scoped-account"
        assert adapter._token == "scoped-token"

    def test_multiplex_scoped_miss_yields_empty_not_environ_borrow(
        self, multiplex_on, monkeypatch
    ):
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "env-account")
        monkeypatch.setenv("WEIXIN_TOKEN", "env-token")
        token = secret_scope.set_secret_scope({"SOMETHING_ELSE": "x"})
        try:
            adapter = WeixinAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._account_id == ""
        assert adapter._token == ""

    def test_multiplex_unscoped_construction_falls_back_to_environ(
        self, multiplex_on, monkeypatch
    ):
        """Regression for the bare get_secret reads: default-profile adapter
        construction under multiplexing must not raise UnscopedSecretError."""
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "env-account")
        monkeypatch.setenv("WEIXIN_TOKEN", "env-token")
        token = secret_scope.set_secret_scope(None)
        try:
            adapter = WeixinAdapter(PlatformConfig(enabled=True))
        finally:
            secret_scope.reset_secret_scope(token)
        assert adapter._account_id == "env-account"
        assert adapter._token == "env-token"
