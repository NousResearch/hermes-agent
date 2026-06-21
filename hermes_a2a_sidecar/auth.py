from __future__ import annotations

import hmac
from dataclasses import dataclass

from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.server.routes.common import DefaultServerCallContextBuilder
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import DEFAULT_SKILLS, PeerPolicy, SidecarConfig


@dataclass(frozen=True, slots=True)
class PeerPrincipal:
    peer_id: str
    policy: PeerPolicy
    auth_method: str


class PeerUser(User):
    def __init__(self, principal: PeerPrincipal | None):
        self._principal = principal

    @property
    def is_authenticated(self) -> bool:
        return self._principal is not None

    @property
    def user_name(self) -> str:
        return self._principal.peer_id if self._principal else ""


class A2AServerCallContextBuilder(DefaultServerCallContextBuilder):
    """Add authenticated A2A peer policy to the official SDK call context."""

    def build(self, request: Request) -> ServerCallContext:
        ctx = super().build(request)
        principal = request.scope.get("a2a.peer")
        if principal:
            ctx.state["a2a_peer"] = principal
            ctx.user = PeerUser(principal)
        return ctx


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "").strip()
    if not header:
        return ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return ""
    return token.strip()


def authenticate_peer(request: Request, config: SidecarConfig) -> PeerPrincipal | None:
    token = _bearer_token(request)
    if token:
        import hashlib

        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        for peer_id, policy in config.peers.items():
            expected = policy.resolved_token_sha256()
            if expected and hmac.compare_digest(token_hash, expected):
                return PeerPrincipal(peer_id=peer_id, policy=policy, auth_method="bearer")

    # Perimeter auth bridge: only configure this when a trusted reverse proxy
    # (Cloudflare Access, oauth2-proxy, Tailscale serve, mTLS terminator) strips
    # spoofed incoming headers and injects an authenticated subject header.
    for peer_id, policy in config.peers.items():
        header_name = policy.trusted_subject_header
        expected = policy.trusted_subject
        if not header_name or not expected:
            continue
        actual = request.headers.get(header_name, "")
        if actual and hmac.compare_digest(actual, expected):
            return PeerPrincipal(peer_id=peer_id, policy=policy, auth_method="trusted_header")

    if config.allow_insecure_local and request.client:
        host = request.client.host
        if host in {"127.0.0.1", "::1", "localhost"}:
            policy = PeerPolicy(id="local-insecure", allowed_skills=list(DEFAULT_SKILLS))
            if config.peers:
                policy.allowed_skills = list(next(iter(config.peers.values())).allowed_skills)
            return PeerPrincipal(peer_id="local-insecure", policy=policy, auth_method="local-insecure")

    return None


class A2AAuthMiddleware(BaseHTTPMiddleware):
    """HTTP auth and payload-size gate for non-public A2A routes."""

    def __init__(self, app, config: SidecarConfig):  # noqa: ANN001
        super().__init__(app)
        self.config = config
        rpc_path = config.rpc_path if config.rpc_path.startswith("/") else f"/{config.rpc_path}"
        self.rpc_path = rpc_path.rstrip("/") or "/"

    def _is_public(self, request: Request) -> bool:
        if request.method == "GET" and request.url.path in {
            "/.well-known/agent-card.json",
            "/healthz",
        }:
            return True
        return False

    def _needs_auth(self, request: Request) -> bool:
        path = request.url.path.rstrip("/") or "/"
        return path == self.rpc_path or path.startswith(f"{self.rpc_path}/")

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        if self._is_public(request) or not self._needs_auth(request):
            return await call_next(request)

        principal = authenticate_peer(request, self.config)
        if principal is None:
            return JSONResponse(
                {"error": "unauthorized", "detail": "A2A peer authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                size = 0
            limit = principal.policy.max_payload_bytes
            if limit > 0 and size > limit:
                return JSONResponse(
                    {"error": "payload_too_large", "max_payload_bytes": limit},
                    status_code=413,
                )

        request.scope["a2a.peer"] = principal
        return await call_next(request)
