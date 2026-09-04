from __future__ import annotations

import asyncio
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


class OAuthFailureKind(str, Enum):
    DEFINITIVE = "definitive"
    INDETERMINATE = "indeterminate"


class ProbeOutcome(str, Enum):
    AUTHENTICATED = "authenticated"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


class InjectedOAuthFailure(RuntimeError):
    def __init__(
        self,
        point: OAuthFailurePoint,
        events: tuple[str, ...],
        *,
        kind: OAuthFailureKind | None = None,
        retry_after: float | None = None,
        probe_outcome: ProbeOutcome | None = None,
    ):
        self.point = point
        self.events = events
        self.kind = kind
        self.retry_after = retry_after
        self.probe_outcome = probe_outcome
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

    def __repr__(self) -> str:
        return f"OAuthArtifactState({self.safe_summary()})"


_OLD_DOCUMENTS = {
    "token": {"access_token": "OLD_ACCESS_TOKEN_FOR_TEST_ONLY", "refresh_token": "OLD_REFRESH_TOKEN_FOR_TEST_ONLY", "token_type": "Bearer", "expires_in": 3600},
    "client": {"client_id": "OLD_CLIENT_ID_FOR_TEST_ONLY", "client_secret": "OLD_CLIENT_SECRET_FOR_TEST_ONLY", "redirect_uris": ["http://127.0.0.1:43111/callback"], "token_endpoint_auth_method": "client_secret_post"},
    "metadata": {"issuer": "https://old-auth.invalid", "authorization_endpoint": "https://old-auth.invalid/authorize", "token_endpoint": "https://old-auth.invalid/token"},
}


def _stable_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def seed_old_oauth_state(home: Path, server_name: str) -> OAuthArtifactState:
    storage = HermesTokenStorage(server_name, hermes_home=Path(home).resolve())
    paths = (storage._tokens_path(), storage._client_info_path(), storage._meta_path())
    payloads = tuple(_stable_json(_OLD_DOCUMENTS[key]) for key in ("token", "client", "metadata"))
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    paths[0].parent.chmod(0o700)
    for path, payload in zip(paths, payloads, strict=True):
        path.write_bytes(payload)
        path.chmod(0o600)
    return OAuthArtifactState(*payloads)


def capture_oauth_state(home: Path, server_name: str) -> OAuthArtifactState:
    storage = HermesTokenStorage(server_name, hermes_home=Path(home).resolve())

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


@dataclass(frozen=True)
class _DumpableOAuthRecord:
    payload: dict

    def model_dump(self, **_kwargs) -> dict:
        return dict(self.payload)


_KINDED_POINTS = {
    OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY,
    OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY,
    OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
    OAuthFailurePoint.TOKEN_EXCHANGE,
}


class FakeOAuthMCPPeer:
    def __init__(
        self,
        failure_point: OAuthFailurePoint | None,
        *,
        kind: OAuthFailureKind = OAuthFailureKind.DEFINITIVE,
        probe_outcome: ProbeOutcome = ProbeOutcome.REJECTED,
    ):
        self.failure_point = failure_point
        self.kind = kind
        self.probe_outcome = probe_outcome
        self.completed_events: list[str] = []
        self.connect_timeouts: list[float | None] = []
        self._new_token = _DumpableOAuthRecord(
            {
                "access_token": "NEW_ACCESS_TOKEN_FOR_TEST_ONLY",
                "refresh_token": "NEW_REFRESH_TOKEN_FOR_TEST_ONLY",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        )

    def _complete(self, point: OAuthFailurePoint, storage: HermesTokenStorage) -> None:
        if point is OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY:
            storage.save_oauth_metadata(
                _DumpableOAuthRecord(
                    {
                        "issuer": "https://partial-auth.invalid",
                        "authorization_endpoint": "https://partial-auth.invalid/authorize",
                        "token_endpoint": "https://partial-auth.invalid/token",
                        "marker": "PARTIAL_METADATA_FOR_TEST_ONLY",
                    }
                )
            )
        elif point is OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION:
            asyncio.run(
                storage.set_client_info(
                    _DumpableOAuthRecord(
                        {
                            "client_id": "PARTIAL_CLIENT_ID_FOR_TEST_ONLY",
                            "client_secret": "PARTIAL_CLIENT_SECRET_FOR_TEST_ONLY",
                            "redirect_uris": ["http://127.0.0.1:43112/callback"],
                            "token_endpoint_auth_method": "client_secret_post",
                        }
                    )
                )
            )
        elif point is OAuthFailurePoint.TOKEN_PERSISTENCE:
            asyncio.run(storage.set_tokens(self._new_token))
        self.completed_events.append(point.value)

    def probe(
        self,
        server_name: str,
        config: dict,
        connect_timeout: float | None = None,
        *,
        details: dict | None = None,
    ) -> list[tuple[str, str]]:
        del config, details
        self.connect_timeouts.append(connect_timeout)
        storage = HermesTokenStorage(server_name)
        if self.failure_point is None:
            for point in OAuthFailurePoint:
                self._complete(point, storage)
            return [("fake_tool", "Deterministic fake MCP tool")]
        for point in OAuthFailurePoint:
            self._complete(point, storage)
            if point is self.failure_point:
                raise self._injected_failure(point)
        return [("fake_tool", "Deterministic fake MCP tool")]

    def _injected_failure(self, point: OAuthFailurePoint) -> InjectedOAuthFailure:
        events = tuple(self.completed_events)
        if point in _KINDED_POINTS:
            retry_after = 1.0 if self.kind is OAuthFailureKind.INDETERMINATE else None
            return InjectedOAuthFailure(point, events, kind=self.kind, retry_after=retry_after)
        if point is OAuthFailurePoint.MCP_INITIALIZATION:
            return InjectedOAuthFailure(point, events, kind=None, probe_outcome=self.probe_outcome)
        return InjectedOAuthFailure(point, events)
