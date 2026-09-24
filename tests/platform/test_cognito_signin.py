"""Sign-in through a load balancer and Cognito: the token is verified, not trusted.

Tokens here are signed with a real RSA key by the ``cryptography`` package, and the
verifier under test is NOVA's own standard-library RS256 check — so a passing test means
NOVA's check agrees with an independent implementation, not with itself.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import hashes  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import padding, rsa  # noqa: E402

from nova.control import ControlAPI  # noqa: E402
from nova.control.oidc import (  # noqa: E402
    ACCESS_TOKEN_HEADER,
    CognitoVerifier,
    NotInAGroup,
    TokenRejected,
    verify_rs256,
)
from nova.control.server import build_server  # noqa: E402

ISSUER = "https://cognito-idp.eu-west-2.amazonaws.com/eu-west-2_TestPool"
CLIENT = "test-client-id"
NOW = 1_800_000_000


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _uint(value: int) -> str:
    return _b64(value.to_bytes((value.bit_length() + 7) // 8, "big"))


@pytest.fixture(scope="module")
def key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(key):
    numbers = key.public_key().public_numbers()
    return {"keys": [{"kty": "RSA", "kid": "k1", "alg": "RS256", "use": "sig",
                      "n": _uint(numbers.n), "e": _uint(numbers.e)}]}


def token(key, *, groups=("nova-admin",), header=None, sign=True, **overrides):
    claims = {"iss": ISSUER, "token_use": "access", "client_id": CLIENT, "username": "yahye",
              "sub": "u-1", "exp": NOW + 3600, "iat": NOW - 10, "cognito:groups": list(groups)}
    claims.update(overrides)
    head = header or {"alg": "RS256", "kid": "k1"}
    signing_input = f"{_b64(json.dumps(head).encode())}.{_b64(json.dumps(claims).encode())}"
    signature = key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256()) if sign else b""
    return f"{signing_input}.{_b64(signature)}"


@pytest.fixture
def verifier(jwks):
    calls = []

    def fetch(url):
        calls.append(url)
        return jwks

    v = CognitoVerifier(issuer=ISSUER, client_id=CLIENT, fetch=fetch, clock=lambda: NOW,
                        logout_url="https://auth.example/logout?client_id=x")
    v.calls = calls
    return v


# -- the RS256 check itself --------------------------------------------------------------


def test_the_standard_library_check_agrees_with_an_independent_implementation(key):
    message = b"header.payload"
    signature = key.sign(message, padding.PKCS1v15(), hashes.SHA256())
    numbers = key.public_key().public_numbers()
    assert verify_rs256(message, signature, numbers.n, numbers.e)
    assert not verify_rs256(b"header.payloaD", signature, numbers.n, numbers.e)
    tampered = bytes([signature[0] ^ 1]) + signature[1:]
    assert not verify_rs256(message, tampered, numbers.n, numbers.e)
    assert not verify_rs256(message, signature[:-1], numbers.n, numbers.e)


# -- tokens ------------------------------------------------------------------------------


def test_an_admin_is_signed_in_as_an_admin(key, verifier):
    principal = verifier.principal(token(key))
    assert (principal.name, principal.role, principal.via) == ("yahye", "admin", "cognito")
    assert verifier.calls == [f"{ISSUER}/.well-known/jwks.json"]


def test_a_viewer_is_a_viewer(key, verifier):
    assert verifier.principal(token(key, groups=("nova-viewer",))).role == "viewer"


def test_signed_in_but_in_no_group_is_refused_as_forbidden(key, verifier):
    with pytest.raises(NotInAGroup, match="nova-admin or nova-viewer"):
        verifier.principal(token(key, groups=()))


@pytest.mark.parametrize("change, words", [
    ({"iss": "https://cognito-idp.eu-west-2.amazonaws.com/eu-west-2_Other"}, "different user pool"),
    ({"client_id": "someone-else"}, "different app client"),
    ({"token_use": "id"}, "not an access token"),
    ({"exp": NOW - 3600}, "expired"),
    ({"nbf": NOW + 3600}, "future"),
])
def test_a_token_for_somewhere_or_somewhen_else_is_refused(key, verifier, change, words):
    with pytest.raises(TokenRejected, match=words):
        verifier.principal(token(key, **change))


def test_a_changed_claim_breaks_the_signature(key, verifier):
    head, payload, signature = token(key, groups=("nova-viewer",)).split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
    claims["cognito:groups"] = ["nova-admin"]
    forged = f"{head}.{_b64(json.dumps(claims).encode())}.{signature}"
    with pytest.raises(TokenRejected, match="signature"):
        verifier.principal(forged)


@pytest.mark.parametrize("alg", ["none", "HS256", "RS512", None])
def test_only_rs256_is_accepted(key, verifier, alg):
    with pytest.raises(TokenRejected, match="algorithm"):
        verifier.principal(token(key, header={"alg": alg, "kid": "k1"}, sign=False))


def test_a_key_the_pool_does_not_publish_is_refused_without_hammering_cognito(key, verifier):
    for _ in range(3):
        with pytest.raises(TokenRejected, match="does not publish"):
            verifier.principal(token(key, header={"alg": "RS256", "kid": "k-unknown"}))
    assert len(verifier.calls) == 1, "one fetch per refetch interval, however many bad tokens"


def test_garbage_is_refused_not_crashed_on(verifier):
    for bad in ("", "a.b", "a.b.c", "!!!.???.***"):
        with pytest.raises(TokenRejected):
            verifier.principal(bad)


def test_a_half_configured_sign_in_is_refused_at_startup():
    with pytest.raises(ValueError):
        CognitoVerifier(issuer=ISSUER, client_id="")
    with pytest.raises(ValueError):
        CognitoVerifier(issuer="https://evil.example/pool", client_id=CLIENT)


# -- through the server, as behind a load balancer ----------------------------------------


@pytest.fixture
def live(bundle, runtime, verifier):
    # behind_tls_proxy: loopback is not trusted, exactly as behind a load balancer on the host.
    server = build_server(ControlAPI(bundle, runtime), host="127.0.0.1", port=0,
                          behind_tls_proxy=True, oidc=verifier)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def call(url, *, access_token="", method="GET", body=None):
    request = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None)
    if access_token:
        request.add_header(ACCESS_TOKEN_HEADER, access_token)
    if body is not None:
        request.add_header("Content-Type", "application/json")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def test_no_sign_in_is_refused(live):
    status, _, _ = call(f"{live}/platform/v1/agents")
    assert status == 401


def test_a_signed_in_admin_reads_and_is_named(live, key):
    status, _, body = call(f"{live}/platform/v1/agents", access_token=token(key))
    assert status == 200
    status, _, body = call(f"{live}/platform/v1/whoami", access_token=token(key))
    assert status == 200 and json.loads(body) == {
        "name": "yahye", "role": "admin", "via": "cognito", "sign_out": "/logout"}


def test_a_viewer_cannot_write(live, key):
    status, _, body = call(f"{live}/platform/v1/agents/operations/update", method="POST",
                           access_token=token(key, groups=("nova-viewer",)), body={"fields": {}})
    assert status == 403


def test_signed_in_without_a_group_is_told_why(live, key):
    status, _, body = call(f"{live}/platform/v1/agents", access_token=token(key, groups=()))
    assert status == 403 and "no Control Centre group" in json.loads(body)["error"]["message"]


def test_a_forged_token_is_refused(live, key):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    status, _, _ = call(f"{live}/platform/v1/agents", access_token=token(other))
    assert status == 401


def test_sign_out_clears_the_load_balancer_session_and_ends_the_pools(live, key):
    status, headers, _ = call(f"{live}/logout", access_token=token(key))
    assert status == 302
    assert headers["Location"] == "https://auth.example/logout?client_id=x"


def test_a_non_loopback_bind_is_allowed_with_sign_in_and_still_needs_tls(bundle, runtime, verifier):
    from nova.errors import NovaError

    with pytest.raises(NovaError, match="TLS"):
        build_server(ControlAPI(bundle, runtime), host="0.0.0.0", port=0, oidc=verifier)
    server = build_server(ControlAPI(bundle, runtime), host="0.0.0.0", port=0,
                          behind_tls_proxy=True, oidc=verifier)
    server.server_close()
