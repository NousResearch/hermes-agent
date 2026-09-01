from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tools.mcp_oauth import HermesTokenStorage


class OAuthFailurePoint(str, Enum):
    PROTECTED_RESOURCE_DISCOVERY = "protected_resource_discovery"
    AUTHORIZATION_SERVER_DISCOVERY = "authorization_server_discovery"
    DYNAMIC_CLIENT_REGISTRATION = "dynamic_client_registration"
    AUTHORIZATION_URL_PUBLICATION = "authorization_url_publication"
    CALLBACK_RECEIPT = "callback_receipt"
    TOKEN_EXCHANGE = "token_exchange"
    TOKEN_PERSISTENCE = "token_persistence"
    MCP_INITIALIZATION = "mcp_initialization"


class InjectedOAuthFailure(RuntimeError):
    def __init__(self, point: OAuthFailurePoint, events: tuple[str, ...]):
        self.point = point
        self.events = events
        super().__init__(f"injected OAuth failure after {point.value}")


class KnownCredentialLoss(AssertionError):
    """Raised only after a test recognizes GH #76590's exact state shape."""


@dataclass(frozen=True)
class OAuthArtifactState:
    token: bytes | None
    client: bytes | None
    metadata: bytes | None

    @staticmethod
    def _label(value: bytes | None) -> str:
        if value is None:
            return "MISSING"
        if b"OLD_" in value:
            return "OLD"
        if b"old-auth.invalid" in value:
            return "OLD"
        if b"PARTIAL_" in value:
            return "PARTIAL"
        if b"NEW_" in value:
            return "NEW"
        return "OTHER"

    def labels(self) -> tuple[str, str, str]:
        return tuple(self._label(value) for value in (self.token, self.client, self.metadata))

    def safe_summary(self) -> str:
        token, client, metadata = self.labels()
        return f"token={token} client={client} metadata={metadata}"


_OLD_DOCUMENTS = {
    "token": {"access_token": "OLD_ACCESS_TOKEN_FOR_TEST_ONLY", "refresh_token": "OLD_REFRESH_TOKEN_FOR_TEST_ONLY", "token_type": "Bearer", "expires_in": 3600},
    "client": {"client_id": "OLD_CLIENT_ID_FOR_TEST_ONLY", "client_secret": "OLD_CLIENT_SECRET_FOR_TEST_ONLY", "redirect_uris": ["http://127.0.0.1:43111/callback"], "token_endpoint_auth_method": "client_secret_post"},
    "metadata": {"issuer": "https://old-auth.invalid", "authorization_endpoint": "https://old-auth.invalid/authorize", "token_endpoint": "https://old-auth.invalid/token"},
}


def _stable_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def seed_old_oauth_state(home: Path, server_name: str) -> OAuthArtifactState:
    storage = HermesTokenStorage(server_name, hermes_home=home)
    paths = (storage._tokens_path(), storage._client_info_path(), storage._meta_path())
    payloads = tuple(_stable_json(_OLD_DOCUMENTS[key]) for key in ("token", "client", "metadata"))
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    paths[0].parent.chmod(0o700)
    for path, payload in zip(paths, payloads, strict=True):
        path.write_bytes(payload)
        path.chmod(0o600)
    return OAuthArtifactState(*payloads)


def capture_oauth_state(home: Path, server_name: str) -> OAuthArtifactState:
    storage = HermesTokenStorage(server_name, hermes_home=home)

    def read(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    return OAuthArtifactState(read(storage._tokens_path()), read(storage._client_info_path()), read(storage._meta_path()))


def raise_known_mutation(
    *,
    before: OAuthArtifactState,
    after: OAuthArtifactState,
    expected_labels: tuple[str, str, str],
    surface: str,
    failure_point: OAuthFailurePoint,
) -> None:
    if after == before:
        return
    if after.labels() != expected_labels:
        raise AssertionError(
            "unexpected OAuth artifact state: "
            f"surface={surface} failure={failure_point.value} after={after.safe_summary()}"
        )
    raise KnownCredentialLoss(
        f"surface={surface} before={before.safe_summary()} "
        f"failure={failure_point.value} after={after.safe_summary()}"
    )
