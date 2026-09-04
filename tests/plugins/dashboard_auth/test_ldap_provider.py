"""Tests for the LdapAuthProvider plugin (LDAP bind auth, signed sessions).

Loads the plugin module directly (bundled backend plugin, not a package on
the import path) and exercises construction validation, the stateless
session-token lifecycle, the two bind modes (via ldap3 MOCK_SYNC), group
restriction, refresh directory checks, and the register(ctx) entry point.
"""

from __future__ import annotations

import logging
import secrets
import time
from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.ldap as ldap_plugin
from hermes_cli.dashboard_auth import (
    InvalidCredentialsError,
    ProviderError,
    RefreshExpiredError,
    assert_protocol_compliance,
)

SECRET = secrets.token_bytes(32)


def make_provider(**overrides):
    kwargs = dict(
        server_url="ldaps://ldap.example.com",
        secret=SECRET,
        user_dn_template="uid={username},ou=people,dc=example,dc=com",
    )
    kwargs.update(overrides)
    return ldap_plugin.LdapAuthProvider(**kwargs)


class TestProtocolAndConstruction:
    def test_protocol_compliance(self):
        assert_protocol_compliance(ldap_plugin.LdapAuthProvider)

    def test_supports_password_flag(self):
        p = make_provider()
        assert p.supports_password is True
        assert p.name == "ldap"

    def test_oauth_methods_are_stubs(self):
        p = make_provider()
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="http://x/cb")
        with pytest.raises(NotImplementedError):
            p.complete_login(code="c", state="s", code_verifier="v",
                             redirect_uri="http://x/cb")

    def test_rejects_short_secret(self):
        with pytest.raises(ValueError, match="secret"):
            make_provider(secret=b"short")

    def test_rejects_bad_scheme(self):
        with pytest.raises(ValueError, match="server_url"):
            make_provider(server_url="http://ldap.example.com")

    def test_rejects_plain_ldap_without_tls(self):
        with pytest.raises(ValueError, match="allow_insecure"):
            make_provider(server_url="ldap://ldap.example.com")

    def test_plain_ldap_allowed_with_start_tls(self):
        make_provider(server_url="ldap://ldap.example.com", start_tls=True)

    def test_plain_ldap_allowed_with_allow_insecure(self):
        make_provider(server_url="ldap://ldap.example.com",
                      allow_insecure=True)

    def test_rejects_both_bind_modes(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            make_provider(
                user_dn_template="uid={username},dc=example,dc=com",
                user_search_base="ou=people,dc=example,dc=com",
            )

    def test_rejects_no_bind_mode(self):
        with pytest.raises(ValueError, match="bind mode"):
            make_provider(user_dn_template="")

    def test_rejects_template_without_placeholder(self):
        with pytest.raises(ValueError, match="{username}"):
            make_provider(user_dn_template="uid=admin,dc=example,dc=com")

    def test_rejects_dn_template_with_stray_placeholder(self):
        # Contains {username}, so the presence check passes — but
        # .format(username=...) would raise KeyError('dept') at the FIRST
        # login, 500-ing the login endpoint. Must fail at construction.
        with pytest.raises(ValueError, match="dept"):
            make_provider(
                user_dn_template="uid={username},ou={dept},dc=example,dc=com"
            )

    def test_rejects_search_filter_with_stray_placeholder(self):
        with pytest.raises(ValueError, match="user_search_filter"):
            make_provider(
                user_dn_template="",
                user_search_base="ou=people,dc=example,dc=com",
                user_search_filter="(&(uid={username})(x={y}))",
            )


class TestSessionTokens:
    def test_mint_verify_roundtrip(self):
        p = make_provider()
        s = p._mint_session("alice", "uid=alice,ou=people,dc=example,dc=com",
                            {"email": "alice@example.com", "display": "Alice"})
        assert s.provider == "ldap"
        assert s.user_id == "alice"
        assert s.email == "alice@example.com"
        assert s.display_name == "Alice"
        got = p.verify_session(access_token=s.access_token)
        assert got is not None
        assert got.user_id == "alice"
        assert got.email == "alice@example.com"

    def test_verify_rejects_tampered_token(self):
        p = make_provider()
        s = p._mint_session("alice", "uid=alice,dc=example,dc=com", {})
        assert p.verify_session(access_token=s.access_token[:-4] + "AAAA") is None

    def test_verify_rejects_wrong_secret(self):
        p1 = make_provider()
        p2 = make_provider(secret=secrets.token_bytes(32))
        s = p1._mint_session("alice", "uid=alice,dc=example,dc=com", {})
        assert p2.verify_session(access_token=s.access_token) is None

    def test_verify_rejects_refresh_token_as_access(self):
        p = make_provider()
        s = p._mint_session("alice", "uid=alice,dc=example,dc=com", {})
        assert p.verify_session(access_token=s.refresh_token) is None

    def test_verify_rejects_expired(self):
        p = make_provider(session_ttl_seconds=60)
        payload = {"sub": "alice", "dn": "x", "em": "", "nm": "alice",
                   "kind": "access", "exp": int(time.time()) - 1}
        token = ldap_plugin._sign(payload, SECRET)
        assert p.verify_session(access_token=token) is None

    def test_refresh_token_only_in_direct_mode(self):
        # Direct mode has no service credentials → refresh is token-only.
        p = make_provider()
        s = p._mint_session("alice", "uid=alice,dc=example,dc=com",
                            {"email": "a@x.com", "display": "Alice"})
        s2 = p.refresh_session(refresh_token=s.refresh_token)
        assert s2.user_id == "alice"
        assert s2.email == "a@x.com"
        assert p.verify_session(access_token=s2.access_token) is not None

    def test_refresh_rejects_garbage(self):
        p = make_provider()
        with pytest.raises(RefreshExpiredError):
            p.refresh_session(refresh_token="not-a-token")
        with pytest.raises(RefreshExpiredError):
            p.refresh_session(refresh_token="")

    def test_revoke_never_raises(self):
        p = make_provider()
        assert p.revoke_session(refresh_token="anything") is None


# ---------------------------------------------------------------------------
# LDAP I/O tests — use ldap3's offline MOCK_SYNC strategy (no real server).
# ---------------------------------------------------------------------------

ldap3 = pytest.importorskip("ldap3")

BASE_DN = "dc=example,dc=com"
ALICE_DN = f"uid=alice,ou=people,{BASE_DN}"
GROUP_DN = f"cn=hermes-users,ou=groups,{BASE_DN}"

MOCK_ENTRIES = {
    ALICE_DN: {
        "objectClass": ["inetOrgPerson"],
        "uid": ["alice"],
        "cn": ["Alice Adams"],
        "mail": ["alice@example.com"],
        "userPassword": ["s3cret"],
    },
    f"uid=bob,ou=people,{BASE_DN}": {
        "objectClass": ["inetOrgPerson"],
        "uid": ["bob"],
        "cn": ["Bob Brown"],
        "mail": ["bob@example.com"],
        "userPassword": ["hunter2"],
    },
    f"cn=hermes,ou=svc,{BASE_DN}": {
        "objectClass": ["simpleSecurityObject"],
        "cn": ["hermes"],
        "userPassword": ["svc-secret"],
    },
    GROUP_DN: {
        "objectClass": ["groupOfNames"],
        "cn": ["hermes-users"],
        "member": [ALICE_DN],
    },
}


def mock_factory(entries=MOCK_ENTRIES):
    """connection_factory backed by ldap3's offline mock directory."""
    server = ldap3.Server("fake_ldap_server")

    def factory(*, user, password):
        conn = ldap3.Connection(
            server,
            user=user or None,
            password=password or None,
            client_strategy=ldap3.MOCK_SYNC,
            raise_exceptions=False,
        )
        for dn, attrs in entries.items():
            conn.strategy.add_entry(dn, attrs)
        return conn

    return factory


def broken_factory(*, user, password):
    """Factory simulating an unreachable directory."""
    from ldap3.core.exceptions import LDAPSocketOpenError

    raise LDAPSocketOpenError("connection refused")


class TestDirectBindLogin:
    def make(self, **overrides):
        return make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            connection_factory=mock_factory(),
            **overrides,
        )

    def test_valid_credentials_mint_session(self):
        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        assert s.user_id == "alice"
        assert s.provider == "ldap"
        assert p.verify_session(access_token=s.access_token) is not None

    def test_direct_mode_has_no_email(self):
        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        assert s.email == ""
        assert s.display_name == "alice"

    def test_wrong_password_rejected(self):
        p = self.make()
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="alice", password="wrong")

    def test_unknown_user_rejected(self):
        p = self.make()
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="mallory", password="x")

    def test_empty_password_rejected_before_bind(self):
        # An empty password is an ANONYMOUS bind on real LDAP servers —
        # must be rejected before any bind is attempted.
        calls = []
        inner = mock_factory()

        def counting_factory(*, user, password):
            calls.append(user)
            return inner(user=user, password=password)

        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            connection_factory=counting_factory,
        )
        for pw in ("", "   ", "\t"):
            with pytest.raises(InvalidCredentialsError):
                p.complete_password_login(username="alice", password=pw)
        assert calls == []  # no bind ever attempted

    def test_empty_username_rejected(self):
        p = self.make()
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="", password="s3cret")

    def test_username_is_rdn_escaped(self):
        # A username with DN metacharacters must not smuggle extra RDNs
        # into the template. "alice,ou=admins" would, unescaped, bind as
        # uid=alice,ou=admins,ou=people,... — escaped, it's a single
        # (nonexistent) RDN value and the login fails.
        p = self.make()
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(
                username="alice,ou=admins", password="s3cret"
            )

    def test_directory_down_raises_provider_error(self):
        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            connection_factory=broken_factory,
        )
        with pytest.raises(ProviderError):
            p.complete_password_login(username="alice", password="s3cret")

    def test_transport_failure_closes_the_connection(self):
        # ldap3 raises LDAPSocketOpenError from a failed connect()/TLS wrap
        # WITHOUT closing the socket it already opened. If the provider
        # doesn't close it, a down directory leaks one fd per hit on this
        # unauthenticated endpoint.
        from ldap3.core.exceptions import LDAPSocketOpenError

        conn = MagicMock()
        conn.bind.side_effect = LDAPSocketOpenError("connection refused")

        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            connection_factory=lambda *, user, password: conn,
        )
        with pytest.raises(ProviderError):
            p.complete_password_login(username="alice", password="s3cret")
        conn.unbind.assert_called_once_with()


class TestDefaultFactory:
    """Cover the REAL connection factory's TLS policy and timeouts.

    Every other test injects a fake ``connection_factory``, so nothing
    else exercises ``_default_factory`` — the highest-consequence
    security code in the plugin. Without these assertions an edit
    flipping ``CERT_REQUIRED`` to ``CERT_NONE`` would keep the suite
    green while silently disabling certificate validation.
    """

    @pytest.fixture
    def patched(self, monkeypatch):
        """Patch ldap3's Tls/Server/Connection and record their kwargs."""
        calls: dict = {}

        def fake_tls(**kwargs):
            calls["tls"] = kwargs
            return "TLS_OBJ"

        def fake_server(url, **kwargs):
            calls["server"] = dict(kwargs, url=url)
            return "SERVER_OBJ"

        def fake_connection(server, **kwargs):
            calls["conn"] = dict(kwargs, server=server)
            return calls.setdefault("conn_obj", MagicMock())

        monkeypatch.setattr(ldap3, "Tls", fake_tls)
        monkeypatch.setattr(ldap3, "Server", fake_server)
        monkeypatch.setattr(ldap3, "Connection", fake_connection)
        return calls

    def test_ldaps_validates_certificates(self, patched):
        import ssl

        p = make_provider(
            server_url="ldaps://ldap.example.com",
            ca_certs_file="/etc/ssl/private-ca.pem",
            timeout_seconds=7.0,
        )
        conn = p._default_factory(user="uid=alice", password="s3cret")

        assert patched["tls"] == {
            "validate": ssl.CERT_REQUIRED,
            "ca_certs_file": "/etc/ssl/private-ca.pem",
        }
        assert patched["server"]["tls"] == "TLS_OBJ"
        assert patched["server"]["connect_timeout"] == 7.0
        assert patched["server"]["get_info"] is ldap3.NONE
        assert patched["conn"]["client_strategy"] is ldap3.SYNC
        # int, not float: ldap3 2.9.1 struct.pack('LL', ...)s this value on
        # POSIX, so a float breaks every real connection at open().
        assert patched["conn"]["receive_timeout"] == 7
        assert isinstance(patched["conn"]["receive_timeout"], int)
        assert patched["conn"]["raise_exceptions"] is False
        assert patched["conn"]["auto_bind"] is False
        assert patched["conn"]["auto_referrals"] is False
        assert patched["conn"]["user"] == "uid=alice"
        # ldaps:// is TLS from the first byte — no StartTLS upgrade.
        conn.open.assert_not_called()
        conn.start_tls.assert_not_called()

    def test_unset_ca_certs_file_becomes_none(self, patched):
        p = make_provider(server_url="ldaps://ldap.example.com")
        p._default_factory(user=None, password=None)
        # "" would make ldap3 try to load a file named "" — must be None
        # so it falls back to the system trust store.
        assert patched["tls"]["ca_certs_file"] is None
        assert patched["conn"]["auto_referrals"] is False

    def test_referral_chasing_is_disabled(self, patched):
        # ldap3's default (auto_referrals=True) follows a referral the
        # DIRECTORY supplies and re-binds with the SAME credentials — the
        # service account in _search_user/_user_still_present, the end
        # user's own DN and password in _user_in_group — against a host
        # named by directory data. A plain ldap:// referral out of an
        # ldaps:// deployment would then bind in the clear. Must be off
        # in every configuration, which is why it is also asserted in
        # each variant above and below.
        for kwargs in (
            {"server_url": "ldaps://ldap.example.com"},
            {"server_url": "ldap://ldap.example.com", "start_tls": True},
            {"server_url": "ldap://ldap.example.com", "allow_insecure": True},
        ):
            patched.clear()
            make_provider(**kwargs)._default_factory(user=None, password=None)
            assert patched["conn"]["auto_referrals"] is False

    def test_start_tls_over_plain_ldap_attaches_tls_and_upgrades(
        self, patched
    ):
        import ssl

        p = make_provider(
            server_url="ldap://ldap.example.com", start_tls=True
        )
        conn = p._default_factory(user=None, password=None)

        assert patched["tls"]["validate"] == ssl.CERT_REQUIRED
        assert patched["server"]["tls"] == "TLS_OBJ"
        assert patched["conn"]["auto_referrals"] is False
        conn.open.assert_called_once_with()
        conn.start_tls.assert_called_once_with()

    def test_allow_insecure_plain_ldap_has_no_tls(self, patched):
        p = make_provider(
            server_url="ldap://ldap.example.com", allow_insecure=True
        )
        conn = p._default_factory(user=None, password=None)

        assert "tls" not in patched  # ldap3.Tls never constructed
        assert patched["server"]["tls"] is None
        assert patched["conn"]["auto_referrals"] is False
        conn.open.assert_not_called()
        conn.start_tls.assert_not_called()

    def test_start_tls_failure_closes_the_socket(self, patched, monkeypatch):
        # open() connects the socket; if start_tls() then fails, ldap3
        # leaves it open. The factory must close it before propagating.
        from ldap3.core.exceptions import LDAPStartTLSError

        conn = MagicMock()
        conn.start_tls.side_effect = LDAPStartTLSError("handshake failed")
        monkeypatch.setattr(ldap3, "Connection", lambda *a, **k: conn)

        p = make_provider(
            server_url="ldap://ldap.example.com", start_tls=True
        )
        with pytest.raises(LDAPStartTLSError):
            p._default_factory(user=None, password=None)
        conn.unbind.assert_called_once_with()


class TestSearchThenBindLogin:
    def make(self, **overrides):
        kwargs = dict(
            server_url="ldaps://ldap.example.com",
            secret=SECRET,
            bind_dn=f"cn=hermes,ou=svc,{BASE_DN}",
            bind_password="svc-secret",
            user_search_base=f"ou=people,{BASE_DN}",
            connection_factory=mock_factory(),
        )
        kwargs.update(overrides)
        return ldap_plugin.LdapAuthProvider(**kwargs)

    def test_valid_credentials_mint_session_with_attrs(self):
        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        assert s.user_id == "alice"
        assert s.email == "alice@example.com"
        assert s.display_name == "Alice Adams"

    def test_wrong_password_rejected(self):
        p = self.make()
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="alice", password="wrong")

    def test_unknown_user_rejected_generically(self):
        p = self.make()
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="mallory", password="x")

    def test_unknown_user_still_attempts_dummy_bind(self):
        # Timing pad: the factory must see exactly one extra bind attempt
        # (the dummy DN) after the search misses.
        binds = []
        inner = mock_factory()

        def counting_factory(*, user, password):
            binds.append(user)
            return inner(user=user, password=password)

        p = self.make(connection_factory=counting_factory)
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="mallory", password="x")
        # First bind: service account (search); second: dummy pad.
        assert binds[0] == f"cn=hermes,ou=svc,{BASE_DN}"
        assert binds[1] == ldap_plugin._DUMMY_BIND_DN

    def test_service_bind_failure_is_provider_error(self):
        p = self.make(bind_password="wrong-svc-password")
        with pytest.raises(ProviderError):
            p.complete_password_login(username="alice", password="s3cret")

    def test_filter_injection_is_escaped(self):
        # "*)(uid=*" unescaped would wildcard-match every user; escaped
        # per RFC 4515 it matches nothing.
        p = self.make()
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="*)(uid=*", password="x")

    def test_custom_search_filter(self):
        p = self.make(user_search_filter="(mail={username})")
        s = p.complete_password_login(
            username="alice@example.com", password="s3cret"
        )
        assert s.email == "alice@example.com"

    def test_directory_down_is_provider_error(self):
        p = self.make(connection_factory=broken_factory)
        with pytest.raises(ProviderError):
            p.complete_password_login(username="alice", password="s3cret")

    def test_anonymous_search_when_bind_dn_empty(self):
        # MOCK_SYNC allows anonymous binds, standing in for a directory
        # that permits anonymous search.
        p = self.make(bind_dn="", bind_password="")
        s = p.complete_password_login(username="alice", password="s3cret")
        assert s.user_id == "alice"

    def test_multiple_matches_rejected(self):
        # A filter matching several entries must never let a bind against
        # ANY of them succeed. This filter matches both alice and bob
        # whichever username is supplied, so BOTH logins must be rejected
        # even though each password is correct.
        #
        # Both directions are asserted on purpose: ldap3's mock returns
        # matched entries in an unspecified order, so a test that tried
        # only one user would pass spuriously whenever the *other* entry
        # sorted first (its password wouldn't match, hiding the fact that
        # the multi-match guard was gone). Checking both pins the guard
        # regardless of ordering. The filter keeps the {username}
        # placeholder because the constructor requires it.
        p = self.make(
            user_search_filter="(|(uid={username})(uid=alice)(uid=bob))"
        )
        assert p._search_user("alice") == (None, {})
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="alice", password="s3cret")
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="bob", password="hunter2")


class TestGroupRestriction:
    def test_bob_logs_in_when_no_group_required(self):
        # Positive control for the rejection tests below: bob's password
        # really is correct and his DN really does bind. Without this,
        # every "bob is rejected" assertion would still pass if bob
        # simply could not log in at all and the group check were gone.
        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            connection_factory=mock_factory(),
        )
        s = p.complete_password_login(username="bob", password="hunter2")
        assert s.user_id == "bob"

    def test_unique_member_arm_grants_access(self):
        # groupOfUniqueNames stores membership in uniqueMember, not
        # member — the filter's second arm. Bob is in this group; alice
        # (a member of the groupOfNames one) is not.
        group_dn = f"cn=unique-users,ou=groups,{BASE_DN}"
        entries = dict(MOCK_ENTRIES)
        entries[group_dn] = {
            "objectClass": ["groupOfUniqueNames"],
            "cn": ["unique-users"],
            "uniqueMember": [f"uid=bob,ou=people,{BASE_DN}"],
        }
        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            require_group=group_dn,
            connection_factory=mock_factory(entries),
        )
        assert p.complete_password_login(
            username="bob", password="hunter2"
        ).user_id == "bob"
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="alice", password="s3cret")

    def test_member_uid_arm_grants_access(self):
        # posixGroup stores bare usernames in memberUid, not DNs — the
        # filter's third arm, matched on the username rather than the DN.
        group_dn = f"cn=posix-users,ou=groups,{BASE_DN}"
        entries = dict(MOCK_ENTRIES)
        entries[group_dn] = {
            "objectClass": ["posixGroup"],
            "cn": ["posix-users"],
            "memberUid": ["bob"],
        }
        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            require_group=group_dn,
            connection_factory=mock_factory(entries),
        )
        assert p.complete_password_login(
            username="bob", password="hunter2"
        ).user_id == "bob"
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="alice", password="s3cret")

    def test_member_allowed(self):
        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            require_group=GROUP_DN,
            connection_factory=mock_factory(),
        )
        s = p.complete_password_login(username="alice", password="s3cret")
        assert s.user_id == "alice"

    def test_non_member_rejected_generically(self):
        # bob's password is correct but he is not in hermes-users →
        # the SAME generic error as a wrong password (no group oracle).
        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            require_group=GROUP_DN,
            connection_factory=mock_factory(),
        )
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="bob", password="hunter2")

    def test_group_check_in_search_mode(self):
        p = ldap_plugin.LdapAuthProvider(
            server_url="ldaps://ldap.example.com",
            secret=SECRET,
            bind_dn=f"cn=hermes,ou=svc,{BASE_DN}",
            bind_password="svc-secret",
            user_search_base=f"ou=people,{BASE_DN}",
            require_group=GROUP_DN,
            connection_factory=mock_factory(),
        )
        assert p.complete_password_login(
            username="alice", password="s3cret"
        ).user_id == "alice"
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="bob", password="hunter2")


def denied_search_factory(result):
    """Factory whose connections bind fine but whose ``search()`` fails.

    Stands in for an ACL denial (insufficientAccessRights) or a
    nonexistent search base. Connections are built with
    ``raise_exceptions=False``, so ldap3 reports those as a *falsy*
    search plus a populated ``conn.result`` — never an exception — which
    the MOCK_SYNC directory cannot reproduce.
    """

    class _Conn:
        entries: list = []

        def __init__(self):
            self.result = result

        def bind(self):
            return True

        def search(self, **kwargs):
            return False

        def unbind(self):
            return None

    return lambda *, user, password: _Conn()


class TestFailedProbesAreLogged:
    """A denied probe rejects EVERY login — it must not do so silently."""

    DENIED = {"result": 50, "description": "insufficientAccessRights"}

    def test_denied_user_search_warns(self, caplog):
        p = ldap_plugin.LdapAuthProvider(
            server_url="ldaps://ldap.example.com",
            secret=SECRET,
            bind_dn=f"cn=hermes,ou=svc,{BASE_DN}",
            bind_password="svc-secret",
            user_search_base=f"ou=people,{BASE_DN}",
            connection_factory=denied_search_factory(self.DENIED),
        )
        caplog.set_level(logging.WARNING, logger=ldap_plugin.__name__)
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="alice", password="s3cret")
        assert "user search" in caplog.text
        assert "insufficientAccessRights" in caplog.text
        assert f"ou=people,{BASE_DN}" in caplog.text
        # Credentials must never reach the log.
        assert "s3cret" not in caplog.text
        assert "alice" not in caplog.text

    def test_denied_group_probe_warns(self, caplog):
        p = make_provider(
            user_dn_template="uid={username},ou=people," + BASE_DN,
            require_group=GROUP_DN,
            connection_factory=denied_search_factory(self.DENIED),
        )
        caplog.set_level(logging.WARNING, logger=ldap_plugin.__name__)
        with pytest.raises(InvalidCredentialsError):
            p.complete_password_login(username="alice", password="s3cret")
        assert "require_group" in caplog.text
        assert GROUP_DN in caplog.text
        assert "insufficientAccessRights" in caplog.text
        assert "s3cret" not in caplog.text


def search_failure_factory(exc):
    """Factory whose connections bind fine but raise inside ``search()``.

    Returns ``(factory, made)`` where ``made`` collects every connection
    handed out, so a test can assert it was unbound. The MOCK_SYNC
    directory cannot reach the refresh probe's exception arms at all —
    it is built with ``raise_exceptions=False``, so LDAP result codes
    arrive as a falsy search rather than an exception — hence this stub.
    """
    made = []

    class _Conn:
        entries: list = []

        def __init__(self):
            self.unbound = False

        def bind(self):
            return True

        def search(self, **kwargs):
            raise exc

        def unbind(self):
            self.unbound = True

    def factory(*, user, password):
        conn = _Conn()
        made.append(conn)
        return conn

    return factory, made


class TestRefreshDirectoryCheck:
    def make(self, entries=MOCK_ENTRIES, **overrides):
        kwargs = dict(
            server_url="ldaps://ldap.example.com",
            secret=SECRET,
            bind_dn=f"cn=hermes,ou=svc,{BASE_DN}",
            bind_password="svc-secret",
            user_search_base=f"ou=people,{BASE_DN}",
            connection_factory=mock_factory(entries),
        )
        kwargs.update(overrides)
        return ldap_plugin.LdapAuthProvider(**kwargs)

    def test_refresh_ok_while_user_exists(self):
        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        s2 = p.refresh_session(refresh_token=s.refresh_token)
        assert s2.user_id == "alice"
        assert s2.email == "alice@example.com"

    def test_refresh_rejected_after_user_removed(self):
        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        # Simulate account deletion: swap in a directory without alice.
        gone = {k: v for k, v in MOCK_ENTRIES.items() if k != ALICE_DN}
        p._factory = mock_factory(gone)
        with pytest.raises(RefreshExpiredError):
            p.refresh_session(refresh_token=s.refresh_token)

    def test_refresh_directory_down_is_provider_error(self):
        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        p._factory = broken_factory
        with pytest.raises(ProviderError):
            p.refresh_session(refresh_token=s.refresh_token)

    def test_refresh_transport_failure_mid_probe_is_provider_error(self):
        # The service bind SUCCEEDS and the socket then dies during the
        # existence probe. That is an outage, not a deleted account:
        # answering "user gone" here would raise RefreshExpiredError and
        # log every active user out whenever the directory blipped. The
        # contract says an unreachable directory is a 503 (ProviderError)
        # with the session cookies left intact.
        from ldap3.core.exceptions import LDAPSocketReceiveError

        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        factory, made = search_failure_factory(
            LDAPSocketReceiveError("connection reset by peer")
        )
        p._factory = factory
        with pytest.raises(ProviderError):
            p.refresh_session(refresh_token=s.refresh_token)
        # The failing probe must not leak its connection.
        assert made and all(c.unbound for c in made)

    def test_refresh_no_such_object_result_means_user_gone(self):
        # A connection_factory that opted into raise_exceptions=True
        # surfaces the noSuchObject *result code* as an exception instead
        # of a falsy search — still "user gone", so the session expires.
        from ldap3.core.exceptions import LDAPNoSuchObjectResult

        p = self.make()
        s = p.complete_password_login(username="alice", password="s3cret")
        factory, made = search_failure_factory(
            LDAPNoSuchObjectResult(description="noSuchObject")
        )
        p._factory = factory
        with pytest.raises(RefreshExpiredError):
            p.refresh_session(refresh_token=s.refresh_token)
        assert made and all(c.unbound for c in made)

    def test_refresh_check_can_be_disabled(self):
        p = self.make(verify_user_on_refresh=False)
        s = p.complete_password_login(username="alice", password="s3cret")
        p._factory = broken_factory  # directory gone — token-only refresh
        s2 = p.refresh_session(refresh_token=s.refresh_token)
        assert s2.user_id == "alice"


LDAP_ENV_VARS = (
    "HERMES_DASHBOARD_LDAP_SERVER_URL",
    "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE",
    "HERMES_DASHBOARD_LDAP_BIND_DN",
    "HERMES_DASHBOARD_LDAP_BIND_PASSWORD",
    "HERMES_DASHBOARD_LDAP_USER_SEARCH_BASE",
    "HERMES_DASHBOARD_LDAP_USER_SEARCH_FILTER",
    "HERMES_DASHBOARD_LDAP_REQUIRE_GROUP",
    "HERMES_DASHBOARD_LDAP_START_TLS",
    "HERMES_DASHBOARD_LDAP_ALLOW_INSECURE",
    "HERMES_DASHBOARD_LDAP_CA_CERTS_FILE",
    "HERMES_DASHBOARD_LDAP_SECRET",
    "HERMES_DASHBOARD_LDAP_TTL_SECONDS",
)


class TestRegister:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for var in LDAP_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        # Config-file section must not leak in from the host machine.
        monkeypatch.setattr(
            ldap_plugin, "_load_config_ldap_section", lambda: {}
        )
        # register() must not hit the network installer in tests.
        monkeypatch.setattr(ldap_plugin, "_ensure_ldap3", lambda: None)

    def test_skips_without_server_url(self):
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "server_url" in ldap_plugin.LAST_SKIP_REASON

    def test_skips_without_bind_mode(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_SERVER_URL", "ldaps://ldap.example.com"
        )
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "user_dn_template" in ldap_plugin.LAST_SKIP_REASON

    def test_registers_direct_mode_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_SERVER_URL", "ldaps://ldap.example.com"
        )
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE",
            "uid={username},ou=people,dc=example,dc=com",
        )
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args[0][0]
        assert provider.name == "ldap"
        assert ldap_plugin.LAST_SKIP_REASON == ""

    def test_registers_search_mode_from_config(self, monkeypatch):
        monkeypatch.setattr(
            ldap_plugin,
            "_load_config_ldap_section",
            lambda: {
                "server_url": "ldaps://ldap.example.com",
                "bind_dn": "cn=hermes,ou=svc,dc=example,dc=com",
                "bind_password": "svc-secret",
                "user_search_base": "ou=people,dc=example,dc=com",
                "require_group": "cn=hermes-users,ou=groups,dc=example,dc=com",
                "display_name": "PESCO Active Directory",
            },
        )
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args[0][0]
        assert provider.display_name == "PESCO Active Directory"

    def test_env_wins_over_config(self, monkeypatch):
        monkeypatch.setattr(
            ldap_plugin,
            "_load_config_ldap_section",
            lambda: {
                "server_url": "ldaps://config-host.example.com",
                "user_dn_template": "uid={username},dc=example,dc=com",
            },
        )
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_SERVER_URL", "ldaps://env-host.example.com"
        )
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args[0][0]
        assert provider._server_url == "ldaps://env-host.example.com"

    def test_construction_error_becomes_skip_reason(self, monkeypatch):
        # plain ldap:// with no start_tls / allow_insecure → skip, not crash.
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_SERVER_URL", "ldap://ldap.example.com"
        )
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE",
            "uid={username},dc=example,dc=com",
        )
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "allow_insecure" in ldap_plugin.LAST_SKIP_REASON

    def test_stray_placeholder_becomes_skip_reason(self, monkeypatch):
        # A second placeholder in the DN template is a config error, not
        # a 500 at the first login — register() must skip, not raise.
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_SERVER_URL", "ldaps://ldap.example.com"
        )
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE",
            "uid={username},ou={dept},dc=example,dc=com",
        )
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "user_dn_template" in ldap_plugin.LAST_SKIP_REASON
        assert "dept" in ldap_plugin.LAST_SKIP_REASON

    def test_hardening_env_vars_reach_the_provider(self, monkeypatch):
        for var, val in (
            ("HERMES_DASHBOARD_LDAP_SERVER_URL", "ldap://ldap.example.com"),
            (
                "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE",
                "uid={username},dc=x",
            ),
            ("HERMES_DASHBOARD_LDAP_START_TLS", "1"),
            ("HERMES_DASHBOARD_LDAP_ALLOW_INSECURE", "1"),
            ("HERMES_DASHBOARD_LDAP_CA_CERTS_FILE", "/x/ca.pem"),
            ("HERMES_DASHBOARD_LDAP_REQUIRE_GROUP", "cn=g,dc=x"),
        ):
            monkeypatch.setenv(var, val)
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args[0][0]
        assert provider._start_tls is True
        assert provider._ca_certs_file == "/x/ca.pem"
        assert provider._require_group == "cn=g,dc=x"

    def test_allow_insecure_env_permits_plain_ldap(self, monkeypatch):
        # START_TLS unset, ALLOW_INSECURE=1 → plain ldap:// registers.
        # Paired with test_construction_error_becomes_skip_reason (same
        # config with NEITHER set → skip reason naming allow_insecure),
        # this pins that the env var is what unlocks it.
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_SERVER_URL", "ldap://ldap.example.com"
        )
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE", "uid={username},dc=x"
        )
        monkeypatch.setenv("HERMES_DASHBOARD_LDAP_ALLOW_INSECURE", "1")
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args[0][0]
        assert provider._start_tls is False
        assert ldap_plugin.LAST_SKIP_REASON == ""

    def test_missing_ldap3_becomes_skip_reason(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_SERVER_URL", "ldaps://ldap.example.com"
        )
        monkeypatch.setenv(
            "HERMES_DASHBOARD_LDAP_USER_DN_TEMPLATE",
            "uid={username},dc=example,dc=com",
        )

        def boom():
            raise RuntimeError("lazy install disabled")

        monkeypatch.setattr(ldap_plugin, "_ensure_ldap3", boom)
        ctx = MagicMock()
        ldap_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "ldap3" in ldap_plugin.LAST_SKIP_REASON
