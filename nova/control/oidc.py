"""Sign-in through an AWS load balancer and Amazon Cognito.

When the Control Centre sits behind an Application Load Balancer whose listener runs the
``authenticate-cognito`` action, the load balancer signs the person in (password, MFA) and
then forwards every request with the user pool's **access token** in
``x-amzn-oidc-accesstoken``. This module turns that header into a :class:`Principal`.

**The token is verified here, not trusted because of where it came from.** The instance's
security group admits only the load balancer, but a control that depends on a firewall rule
never being loosened is one mistake from "anyone who reaches the port is whoever they say".
So the signature is checked against the user pool's own published keys, and the issuer,
the app client, the token type and the expiry are checked against this deployment.

The check is RS256 (RSASSA-PKCS1-v1_5 with SHA-256), the algorithm Cognito signs access
tokens with. It needs only modular exponentiation and a fixed byte layout, so it is done
with the standard library: NOVA's dependency surface is the standard library and PyYAML,
and a cryptography package added for one check would be the largest thing in it.

**Roles come from Cognito groups** (``cognito:groups`` in the access token): membership of
the admin group makes an admin, of the viewer group a viewer. Signed in and in neither group
is refused — being able to sign in to the pool is not, by itself, a grant.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from nova.control.auth import Principal

#: The header the load balancer forwards the user pool's access token in.
ACCESS_TOKEN_HEADER = "x-amzn-oidc-accesstoken"

#: DER prefix of a SHA-256 DigestInfo (RFC 8017 §9.2, note 1).
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")

#: Clock skew tolerated on exp / nbf / iat.
LEEWAY_SECONDS = 60

#: How often an unknown key id may trigger a refetch of the key set. A token naming a key
#: the pool does not publish must not become a way to make this process hammer Cognito.
REFETCH_INTERVAL_SECONDS = 300


class TokenRejected(Exception):
    """A token that does not authenticate anyone. The message is safe to log."""


class NotInAGroup(TokenRejected):
    """Authenticated, and granted nothing: a 403, not a 401."""


def _b64url(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _int(segment: str) -> int:
    return int.from_bytes(_b64url(segment), "big")


def verify_rs256(signing_input: bytes, signature: bytes, n: int, e: int) -> bool:
    """RSASSA-PKCS1-v1_5 verification with SHA-256 (RFC 8017 §8.2.2)."""
    k = (n.bit_length() + 7) // 8
    if len(signature) != k:
        return False
    s = int.from_bytes(signature, "big")
    if s >= n:
        return False
    em = pow(s, e, n).to_bytes(k, "big")
    digest_info = _SHA256_DIGEST_INFO + hashlib.sha256(signing_input).digest()
    padding = k - len(digest_info) - 3
    if padding < 8:
        return False
    expected = b"\x00\x01" + b"\xff" * padding + b"\x00" + digest_info
    return em == expected


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 — https, fixed host
        return json.load(response)


@dataclass
class CognitoVerifier:
    """Verifies a user pool's access tokens and maps its groups to Control Centre roles."""

    #: ``https://cognito-idp.<region>.amazonaws.com/<user-pool-id>``
    issuer: str
    #: The app client the load balancer signs in through.
    client_id: str
    admin_groups: tuple[str, ...] = ("nova-admin",)
    viewer_groups: tuple[str, ...] = ("nova-viewer",)
    #: Where "Sign out" sends the browser after the load balancer's cookie is cleared.
    logout_url: str = ""
    fetch: Callable[[str], dict] = _fetch_json
    clock: Callable[[], float] = time.time
    _keys: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)
    _fetched_at: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.issuer.startswith("https://cognito-idp.") or not self.client_id:
            raise ValueError("an issuer (https://cognito-idp.<region>.amazonaws.com/<pool>) "
                             "and an app client id are both required")
        self.issuer = self.issuer.rstrip("/")

    # -- keys -----------------------------------------------------------------

    def _key(self, kid: str) -> tuple[int, int]:
        with self._lock:
            if kid not in self._keys and self.clock() - self._fetched_at >= REFETCH_INTERVAL_SECONDS:
                self._fetched_at = self.clock()
                document = self.fetch(f"{self.issuer}/.well-known/jwks.json")
                self._keys = {
                    str(jwk["kid"]): (_int(jwk["n"]), _int(jwk["e"]))
                    for jwk in document.get("keys", [])
                    if jwk.get("kty") == "RSA" and jwk.get("kid")
                }
            if kid not in self._keys:
                raise TokenRejected(f"the token is signed with key {kid!r}, which this user pool does not publish")
            return self._keys[kid]

    # -- the check ------------------------------------------------------------

    def claims(self, token: str) -> Mapping[str, Any]:
        """The token's claims, once every check has passed. Raises :class:`TokenRejected`."""
        parts = token.split(".")
        if len(parts) != 3:
            raise TokenRejected("not a JWT")
        try:
            header = json.loads(_b64url(parts[0]))
            claims = json.loads(_b64url(parts[1]))
            signature = _b64url(parts[2])
        except (ValueError, UnicodeDecodeError) as exc:
            raise TokenRejected(f"malformed token ({type(exc).__name__})") from None
        if header.get("alg") != "RS256":
            # Named, not merely "not verified": alg=none and HS256-with-the-public-key are
            # the classic ways a JWT check is talked out of checking.
            raise TokenRejected(f"algorithm {header.get('alg')!r} is not accepted; only RS256")
        n, e = self._key(str(header.get("kid") or ""))
        if not verify_rs256(f"{parts[0]}.{parts[1]}".encode("ascii"), signature, n, e):
            raise TokenRejected("the signature does not verify")

        now = self.clock()
        if claims.get("iss") != self.issuer:
            raise TokenRejected("issued by a different user pool")
        if claims.get("token_use") != "access":
            raise TokenRejected("not an access token")
        if claims.get("client_id") != self.client_id:
            raise TokenRejected("issued to a different app client")
        if not isinstance(claims.get("exp"), (int, float)) or now > claims["exp"] + LEEWAY_SECONDS:
            raise TokenRejected("expired")
        for early in ("nbf", "iat"):
            if isinstance(claims.get(early), (int, float)) and claims[early] > now + LEEWAY_SECONDS:
                raise TokenRejected(f"{early} is in the future")
        return claims

    def principal(self, token: str) -> Principal:
        """The signed-in person as a Control Centre principal. Raises :class:`TokenRejected`."""
        claims = self.claims(token)
        groups = set(claims.get("cognito:groups") or ())
        name = str(claims.get("username") or claims.get("sub") or "")
        if groups & set(self.admin_groups):
            role = "admin"
        elif groups & set(self.viewer_groups):
            role = "viewer"
        else:
            raise NotInAGroup(
                f"{name or 'this user'} is signed in but in no Control Centre group; an "
                f"administrator adds them to {self.admin_groups[0]} or {self.viewer_groups[0]}"
            )
        return Principal(name=name, role=role, via="cognito")
