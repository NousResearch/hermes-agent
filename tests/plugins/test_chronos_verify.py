"""Tests for the Chronos inbound cron-fire JWT verifier (Phase 4E.1).

These exercise REAL RS256 signing/verification (PyJWT[crypto] is a declared
dependency) against an inline PEM public key — no mocking of the crypto, since
this is a security boundary. The JWKS-URL path is covered separately by mocking
PyJWKClient's key resolution.
"""

import time

import pytest


@pytest.fixture(scope="module")
def rsa_keys():
    """An RS256 keypair: (private_pem, public_pem)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv, pub


def _mint(priv, claims):
    import jwt
    return jwt.encode(claims, priv, algorithm="RS256")


AUD = "agent:inst-123"
ISS = "https://portal.nousresearch.com"


def _base_claims(**over):
    now = int(time.time())
    c = {
        "aud": AUD,
        "iss": ISS,
        "purpose": "cron_fire",
        "iat": now,
        "nbf": now - 5,
        "exp": now + 300,
    }
    c.update(over)
    return c


def test_valid_token_returns_claims(rsa_keys):
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    token = _mint(priv, _base_claims())
    claims = verify_nas_fire_token(token=token, expected_audience=AUD,
                                   jwks_or_key=pub, issuer=ISS)
    assert claims is not None
    assert claims["purpose"] == "cron_fire"
    assert claims["aud"] == AUD


def test_wrong_audience_rejected(rsa_keys):
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    token = _mint(priv, _base_claims(aud="agent:someone-else"))
    assert verify_nas_fire_token(token=token, expected_audience=AUD,
                                 jwks_or_key=pub, issuer=ISS) is None


def test_missing_purpose_rejected(rsa_keys):
    """A general agent JWT (no purpose=cron_fire) can't fire jobs."""
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    claims = _base_claims()
    del claims["purpose"]
    token = _mint(priv, claims)
    assert verify_nas_fire_token(token=token, expected_audience=AUD,
                                 jwks_or_key=pub, issuer=ISS) is None


def test_wrong_purpose_rejected(rsa_keys):
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    token = _mint(priv, _base_claims(purpose="inference"))
    assert verify_nas_fire_token(token=token, expected_audience=AUD,
                                 jwks_or_key=pub, issuer=ISS) is None


def test_expired_token_rejected(rsa_keys):
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    now = int(time.time())
    token = _mint(priv, _base_claims(iat=now - 1000, nbf=now - 1000, exp=now - 600))
    assert verify_nas_fire_token(token=token, expected_audience=AUD,
                                 jwks_or_key=pub, issuer=ISS) is None


def test_wrong_issuer_rejected(rsa_keys):
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    token = _mint(priv, _base_claims(iss="https://evil.example"))
    assert verify_nas_fire_token(token=token, expected_audience=AUD,
                                 jwks_or_key=pub, issuer=ISS) is None


def test_tampered_signature_rejected(rsa_keys):
    """A token signed by a DIFFERENT key must fail signature verification."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    _, pub = rsa_keys
    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    attacker_priv = attacker.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    token = _mint(attacker_priv, _base_claims())
    # Verified against the REAL public key → signature mismatch → None.
    assert verify_nas_fire_token(token=token, expected_audience=AUD,
                                 jwks_or_key=pub, issuer=ISS) is None


def test_no_key_configured_refuses(rsa_keys):
    """No JWKS/key configured → refuse (never fall back to unsigned decode)."""
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, _ = rsa_keys
    token = _mint(priv, _base_claims())
    assert verify_nas_fire_token(token=token, expected_audience=AUD,
                                 jwks_or_key=None) is None


def test_empty_token_refused(rsa_keys):
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    _, pub = rsa_keys
    assert verify_nas_fire_token(token="", expected_audience=AUD, jwks_or_key=pub) is None


def test_jwks_url_path_resolves_key(rsa_keys, monkeypatch):
    """The JWKS-URL branch resolves the signing key via PyJWKClient."""
    from plugins.cron_providers.chronos import verify as verify_mod
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    token = _mint(priv, _base_claims())

    class FakeKey:
        key = pub

    class FakeJWKClient:
        def __init__(self, url):
            assert url == "https://portal.nousresearch.com/.well-known/jwks.json"

        def get_signing_key_from_jwt(self, tok):
            return FakeKey()

    monkeypatch.setattr("jwt.PyJWKClient", FakeJWKClient)
    # Isolate from the process-wide client cache (other tests may have populated it).
    monkeypatch.setattr(verify_mod, "_JWK_CLIENTS", {})
    claims = verify_nas_fire_token(
        token=token, expected_audience=AUD,
        jwks_or_key="https://portal.nousresearch.com/.well-known/jwks.json",
        issuer=ISS,
    )
    assert claims is not None and claims["purpose"] == "cron_fire"


def test_jwks_client_is_cached_across_calls(rsa_keys, monkeypatch):
    """Regression (Chronos relay 403/401 + 504 storm): the JWKS client must be
    constructed ONCE per URL and reused across fires, NOT rebuilt per call.

    A fresh PyJWKClient per fire discards its key cache and forces a synchronous
    JWKS HTTP GET on every fire; a burst of concurrent fires then fans out into N
    simultaneous fetches that the portal rate-limits (403 → agent 401) or that
    block the event loop past the relay's 30s timeout (504). Reusing one cached
    client keeps the steady state at zero fetches per fire. This test fails
    against the pre-fix code (construct_count == N) and passes with the cache
    (construct_count == 1).
    """
    from plugins.cron_providers.chronos import verify as verify_mod
    from plugins.cron_providers.chronos.verify import verify_nas_fire_token

    priv, pub = rsa_keys
    url = "https://portal.nousresearch.com/.well-known/jwks.json"

    counters = {"construct": 0, "fetch": 0}

    class FakeKey:
        key = pub

    class CountingJWKClient:
        def __init__(self, u):
            counters["construct"] += 1

        def get_signing_key_from_jwt(self, tok):
            counters["fetch"] += 1
            return FakeKey()

    monkeypatch.setattr("jwt.PyJWKClient", CountingJWKClient)
    # Start from an empty cache so this test's count is deterministic.
    monkeypatch.setattr(verify_mod, "_JWK_CLIENTS", {})

    for _ in range(5):
        token = _mint(priv, _base_claims())
        claims = verify_nas_fire_token(
            token=token, expected_audience=AUD, jwks_or_key=url, issuer=ISS,
        )
        assert claims is not None

    # The client is built once and reused; only the fetch (served from the
    # client's own cache in production) is per-call.
    assert counters["construct"] == 1, (
        f"expected 1 PyJWKClient construction, got {counters['construct']} "
        "(a fresh client per fire is the bug this guards against)"
    )


def test_get_fire_verifier_returns_nas_verifier():
    from plugins.cron_providers.chronos.verify import get_fire_verifier, verify_nas_fire_token

    assert get_fire_verifier() is verify_nas_fire_token
