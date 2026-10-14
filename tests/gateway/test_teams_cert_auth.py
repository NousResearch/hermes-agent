"""Certificate client-credentials (client_assertion) for the Teams adapter.

Secret auth stays the client_secret form. A configured certificate replaces it.
Neither secret nor certificate is a named error, not a silent empty secret.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
from types import SimpleNamespace

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from gateway.config import PlatformConfig
from plugins.platforms.teams import cert_auth
from plugins.platforms.teams.adapter import (
    TeamsAdapter, _MISSING_TEAMS_AUTH, _bf_token_request, _env_enablement, _standalone_send,
    interactive_setup, validate_config,
)

_BF_SCOPE = "https://api.botframework.com/.default"
_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"


def _material(tmp_path):
    """RSA key plus a one-day self-signed cert. Returns (key_path, cert_path, sha1, sha256)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "teams-test")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256()))
    der = cert.public_bytes(serialization.Encoding.DER)
    key_path = tmp_path / "key.pem"
    cert_path = tmp_path / "cert.pem"
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return key_path, cert_path, hashlib.sha1(der, usedforsecurity=False).digest(), hashlib.sha256(der).digest(), key


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _clear_teams_env(monkeypatch):
    for name in (
        "TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID",
        "TEAMS_CLIENT_CERT_PATH", "TEAMS_CLIENT_KEY_PATH", "TEAMS_CLIENT_CERT_THUMBPRINT",
    ):
        monkeypatch.delenv(name, raising=False)


class TestClientAssertion:
    def test_cert_and_key_sign_an_x5t_assertion(self, tmp_path):
        key_path, cert_path, sha1, _sha256, key = _material(tmp_path)
        prepared = cert_auth.prepare_certificate(str(cert_path), str(key_path), "")
        assertion = cert_auth.build_client_assertion(
            prepared, tenant_id="tenant", client_id="client", now=1_700_000_000)
        header = jwt.get_unverified_header(assertion)
        claims = jwt.decode(
            assertion, key.public_key(), algorithms=["RS256"],
            audience=_TOKEN_URL.format(tenant="tenant"), options={"verify_exp": False})
        assert header["alg"] == "RS256"
        assert header["x5t"] == _b64(sha1)
        assert "x5t#S256" not in header
        assert claims["iss"] == claims["sub"] == "client"
        assert claims["exp"] - claims["nbf"] == 600
        assert claims["jti"]

    def test_combined_pem_needs_no_separate_key(self, tmp_path):
        key_path, cert_path, sha1, _sha256, _key = _material(tmp_path)
        combined = tmp_path / "combined.pem"
        combined.write_bytes(cert_path.read_bytes() + b"\n" + key_path.read_bytes())
        prepared = cert_auth.prepare_certificate(str(combined), "", "")
        assert prepared.x5t_header == "x5t"
        assert prepared.x5t_value == _b64(sha1)

    def test_key_and_thumbprint_without_a_cert_file(self, tmp_path):
        key_path, _cert_path, sha1, _sha256, _key = _material(tmp_path)
        thumb = ":".join(f"{byte:02X}" for byte in sha1)
        prepared = cert_auth.prepare_certificate("", str(key_path), thumb)
        assert prepared.x5t_value == _b64(sha1)

    def test_sha256_thumbprint_uses_s256_header(self, tmp_path):
        key_path, cert_path, _sha1, sha256, _key = _material(tmp_path)
        prepared = cert_auth.prepare_certificate(str(cert_path), str(key_path), sha256.hex())
        assert prepared.x5t_header == "x5t#S256"
        assert prepared.x5t_value == _b64(sha256)

    def test_thumbprint_must_match_the_certificate(self, tmp_path):
        key_path, cert_path, _sha1, _sha256, _key = _material(tmp_path)
        with pytest.raises(ValueError, match="does not match"):
            cert_auth.prepare_certificate(str(cert_path), str(key_path), "ab" * 20)

    def test_encrypted_key_is_rejected(self, tmp_path):
        key_path, cert_path, _sha1, _sha256, key = _material(tmp_path)
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"pw")))
        with pytest.raises(ValueError, match="encrypted"):
            cert_auth.prepare_certificate(str(cert_path), str(key_path), "")

    def test_missing_key_names_the_setting(self, tmp_path):
        _key_path, cert_path, _sha1, _sha256, _key = _material(tmp_path)
        with pytest.raises(ValueError, match="TEAMS_CLIENT_KEY_PATH"):
            cert_auth.prepare_certificate(str(cert_path), "", "")

    def test_key_without_cert_or_thumbprint_names_the_setting(self, tmp_path):
        key_path, _cert_path, _sha1, _sha256, _key = _material(tmp_path)
        with pytest.raises(ValueError, match="TEAMS_CLIENT_CERT_THUMBPRINT"):
            cert_auth.prepare_certificate("", str(key_path), "")

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(ValueError, match="TEAMS_CLIENT_CERT_PATH"):
            cert_auth.prepare_certificate(str(tmp_path / "missing.pem"), "", "")


class TestTokenForm:
    def test_secret_form_is_unchanged(self):
        url, form = _bf_token_request("tenant", "client-id", "secret")
        assert url == _TOKEN_URL.format(tenant="tenant")
        assert form == {
            "grant_type": "client_credentials", "client_id": "client-id",
            "client_secret": "secret", "scope": _BF_SCOPE,
        }

    def test_certificate_replaces_the_secret(self, tmp_path):
        key_path, cert_path, sha1, _sha256, key = _material(tmp_path)
        _url, form = _bf_token_request(
            "tenant", "client-id", "secret", cert_path=str(cert_path), key_path=str(key_path))
        assert "client_secret" not in form
        assert form["grant_type"] == "client_credentials"
        assert form["client_id"] == "client-id"
        assert form["scope"] == _BF_SCOPE
        assert form["client_assertion_type"] == "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
        header = jwt.get_unverified_header(form["client_assertion"])
        jwt.decode(
            form["client_assertion"], key.public_key(), algorithms=["RS256"],
            audience=_TOKEN_URL.format(tenant="tenant"))
        assert header["x5t"] == _b64(sha1)


class TestCredentialResolution:
    def test_neither_secret_nor_certificate_is_not_configured(self, monkeypatch):
        _clear_teams_env(monkeypatch)
        assert validate_config(PlatformConfig(enabled=True)) is False
        assert _env_enablement() is None

    def test_certificate_env_enables_without_a_secret(self, monkeypatch):
        _clear_teams_env(monkeypatch)
        monkeypatch.setenv("TEAMS_CLIENT_ID", "id")
        monkeypatch.setenv("TEAMS_TENANT_ID", "tenant")
        monkeypatch.setenv("TEAMS_CLIENT_KEY_PATH", "/keys/app.pem")
        monkeypatch.setenv("TEAMS_CLIENT_CERT_THUMBPRINT", "ab" * 20)
        assert validate_config(PlatformConfig(enabled=True)) is True
        seed = _env_enablement()
        assert seed["client_id"] == "id"
        assert seed["client_key_path"] == "/keys/app.pem"
        assert "client_secret" not in seed

    def test_extra_certificate_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("TEAMS_CLIENT_CERT_PATH", "/env/cert.pem")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "from-env")
        adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={
            "client_id": "id", "tenant_id": "tenant", "client_cert_path": "/yaml/cert.pem",
            "client_key_path": "/yaml/key.pem",
        }))
        assert adapter._cert_path == "/yaml/cert.pem"
        assert adapter._key_path == "/yaml/key.pem"
        assert adapter._uses_cert is True
        assert adapter._client_secret == "from-env"

    def test_certificate_path_follows_the_profile_scope(self, monkeypatch):
        from agent import secret_scope as ss

        monkeypatch.setenv("TEAMS_CLIENT_ID", "default-id")
        monkeypatch.setenv("TEAMS_CLIENT_SECRET", "default-secret")
        monkeypatch.setenv("TEAMS_TENANT_ID", "default-tenant")
        monkeypatch.setenv("TEAMS_CLIENT_CERT_PATH", "/default/cert.pem")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({
            "TEAMS_CLIENT_ID": "bot2", "TEAMS_TENANT_ID": "tenant2",
            "TEAMS_CLIENT_CERT_PATH": "/secondary/cert.pem",
            "TEAMS_CLIENT_KEY_PATH": "/secondary/key.pem",
        })
        try:
            adapter = TeamsAdapter(PlatformConfig(enabled=True))
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)
        assert (adapter._client_id, adapter._tenant_id) == ("bot2", "tenant2")
        assert adapter._client_secret == ""
        assert adapter._cert_path == "/secondary/cert.pem"
        assert adapter._key_path == "/secondary/key.pem"


def _app_namespace(captured: dict):
    class _App:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.id = kwargs.get("client_id")

        def on_message(self, func):
            return func

        def on_card_action(self, func):
            return func

        async def initialize(self):
            return None

    class _Application:
        def __init__(self, **_kwargs):
            self.router = SimpleNamespace(add_get=lambda *_a, **_k: None)

    return _App, _Application


def _patch_connect(monkeypatch, teams, captured):
    app_cls, application = _app_namespace(captured)
    monkeypatch.setattr(teams, "check_teams_requirements", lambda: True)
    monkeypatch.setattr(teams, "AIOHTTP_AVAILABLE", True)
    monkeypatch.setattr(teams, "App", app_cls)
    monkeypatch.setattr(teams, "ClientOptions", lambda **kw: kw)
    monkeypatch.setattr(teams, "web", SimpleNamespace(Application=application))

    async def _bind(*_a, **_k):
        return object()

    monkeypatch.setattr("gateway.platforms.shared_ingress.bind_listener", _bind)


class TestConnectAndTokens:
    @pytest.mark.asyncio
    async def test_connect_refuses_when_neither_auth_method_is_set(self, monkeypatch):
        import plugins.platforms.teams.adapter as teams

        _clear_teams_env(monkeypatch)
        _patch_connect(monkeypatch, teams, {})
        adapter = TeamsAdapter(PlatformConfig(enabled=True))
        assert await adapter.connect() is False
        assert adapter._fatal_error_code == "MISSING_CREDENTIALS"
        assert adapter._fatal_error_message == _MISSING_TEAMS_AUTH
        assert "TEAMS_CLIENT_SECRET" in adapter._fatal_error_message
        assert "TEAMS_CLIENT_CERT_PATH" in adapter._fatal_error_message

    @pytest.mark.asyncio
    async def test_secret_connect_still_passes_the_client_secret(self, monkeypatch):
        import plugins.platforms.teams.adapter as teams

        captured = {}
        _patch_connect(monkeypatch, teams, captured)
        adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={
            "client_id": "id", "client_secret": "secret", "tenant_id": "tenant"}))
        assert await adapter.connect() is True
        assert captured["client_secret"] == "secret"
        assert "token" not in captured

    @pytest.mark.asyncio
    async def test_certificate_connect_uses_the_token_callback(self, monkeypatch, tmp_path):
        import plugins.platforms.teams.adapter as teams

        key_path, cert_path, _sha1, _sha256, _key = _material(tmp_path)
        captured = {}
        _patch_connect(monkeypatch, teams, captured)
        adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={
            "client_id": "id", "client_secret": "leftover-secret", "tenant_id": "tenant",
            "client_cert_path": str(cert_path), "client_key_path": str(key_path),
        }))
        assert await adapter.connect() is True
        assert "client_secret" not in captured
        assert captured["token"] == adapter._sdk_token

    @pytest.mark.asyncio
    async def test_unreadable_certificate_fails_connect_with_the_path(self, monkeypatch, tmp_path):
        import plugins.platforms.teams.adapter as teams

        _patch_connect(monkeypatch, teams, {})
        adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={
            "client_id": "id", "tenant_id": "tenant",
            "client_cert_path": str(tmp_path / "no-such-cert.pem"),
        }))
        assert await adapter.connect() is False
        assert "TEAMS_CLIENT_CERT_PATH" in adapter._fatal_error_message
        assert adapter._app is None

    @pytest.mark.asyncio
    async def test_attachment_token_posts_the_assertion(self, monkeypatch, tmp_path):
        key_path, cert_path, _sha1, _sha256, _key = _material(tmp_path)
        adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={
            "client_id": "bot-id", "client_secret": "secret", "tenant_id": "tenant",
            "client_cert_path": str(cert_path), "client_key_path": str(key_path),
        }))
        posts = []

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"access_token": "tok-cert", "expires_in": 3600}

        class _Client:
            def __init__(self, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

            async def post(self, url, data=None):
                posts.append((url, dict(data or {})))
                return _Resp()

        monkeypatch.setattr("httpx.AsyncClient", _Client)
        assert await adapter._get_botframework_token() == "tok-cert"
        assert await adapter._get_botframework_token() == "tok-cert"
        assert len(posts) == 1
        assert posts[0][0] == _TOKEN_URL.format(tenant="tenant")
        assert "client_secret" not in posts[0][1]
        assert posts[0][1]["client_assertion_type"].endswith("jwt-bearer")
        assert posts[0][1]["scope"] == _BF_SCOPE

    @pytest.mark.asyncio
    async def test_standalone_send_uses_the_assertion(self, monkeypatch, tmp_path):
        import sys
        import types

        key_path, cert_path, _sha1, _sha256, _key = _material(tmp_path)
        _clear_teams_env(monkeypatch)
        seen = {}

        class _Resp:
            def __init__(self, payload):
                self.status = 200
                self._payload = payload

            async def json(self):
                return self._payload

            async def text(self):
                return ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

            def post(self, url, **kwargs):
                seen.setdefault("calls", []).append((url, kwargs))
                if "oauth2" in url:
                    return _Resp({"access_token": "the-token"})
                return _Resp({"id": "msg-1"})

        fake = types.SimpleNamespace(
            ClientSession=lambda timeout=None, **kwargs: _Session(),
            ClientTimeout=lambda total=None: None)
        monkeypatch.setitem(sys.modules, "aiohttp", fake)
        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={
                "client_id": "client-id", "tenant_id": "tenant",
                "client_cert_path": str(cert_path), "client_key_path": str(key_path),
            }),
            "19:abc@thread.skype", "hello cron")
        assert result == {"success": True, "message_id": "msg-1"}
        token_kwargs = seen["calls"][0][1]
        assert "client_secret" not in token_kwargs["data"]
        assert token_kwargs["data"]["client_assertion"]
        assert seen["calls"][1][1]["headers"]["Authorization"] == "Bearer the-token"

    @pytest.mark.asyncio
    async def test_standalone_send_names_both_options_when_unconfigured(self, monkeypatch):
        _clear_teams_env(monkeypatch)
        result = await _standalone_send(PlatformConfig(enabled=True), "19:abc@thread.skype", "hi")
        assert "error" in result
        assert "TEAMS_CLIENT_SECRET" in result["error"]
        assert "certificate" in result["error"].lower()


class TestCertificateSetup:
    def test_wizard_persists_a_certificate_and_clears_a_secret(self, tmp_path, monkeypatch):
        import hermes_cli.cli_output as cli_output_mod

        from hermes_cli.config import save_env_value

        hermes_home = tmp_path / "hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TEAMS_CLIENT_ID", raising=False)
        save_env_value("TEAMS_CLIENT_SECRET", "old-secret")
        answers = iter(["client-id", "/certs/app.pem", "/certs/app.key", "AB:CD", "tenant-id"])
        monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(answers))

        def _yes_no(question, *_a, **_kw):
            text = str(question).lower()
            if "certificate" in text:
                return True
            return False

        monkeypatch.setattr(cli_output_mod, "prompt_yes_no", _yes_no)
        monkeypatch.setattr(cli_output_mod, "print_info", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli_output_mod, "print_success", lambda *_a, **_kw: None)
        monkeypatch.setattr(cli_output_mod, "print_warning", lambda *_a, **_kw: None)

        interactive_setup()

        env_text = (hermes_home / ".env").read_text(encoding="utf-8")
        assert "TEAMS_CLIENT_ID=client-id" in env_text
        assert "TEAMS_TENANT_ID=tenant-id" in env_text
        assert "TEAMS_CLIENT_CERT_PATH=/certs/app.pem" in env_text
        assert "TEAMS_CLIENT_KEY_PATH=/certs/app.key" in env_text
        assert "TEAMS_CLIENT_CERT_THUMBPRINT=AB:CD" in env_text
        assert "TEAMS_CLIENT_SECRET" not in env_text
