"""The fake NAS anonymous surface shared by the free-tier tests.

One ``FakePortal`` and one ``install_portal`` behind every ``portal`` fixture: the wire contract is
exercised through Hermes' real client code, never mocked away. Scenarios flip its behaviour
(``gate_closed``, ``dead_tokens``, a canned ``create_response`` / ``token_response``, or a
``raise_transport`` that makes the wire itself fail).
"""

from __future__ import annotations

import base64
import json
import time

import httpx

WELCOME = "https://welcome-api.nousresearch.com/v1"
PORTAL = "https://portal.example.test"


def make_jwt(**claims) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:1", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke", "exp": int(time.time()) + 900, **claims}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


class FakePortal:
    """Minimal NAS anonymous surface. Records every call; scenarios flip its behaviour."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.dead_tokens: set[str] = set()
        self.gate_closed = False
        self.minted = 0
        # What the token exchange names as the inference host; None = an older NAS that omits it.
        self.inference_base_url: str | None = WELCOME
        # One canned refusal in place of the happy path, and a wire failure in place of any answer.
        self.create_response: httpx.Response | None = None
        self.token_response: httpx.Response | None = None
        self.raise_transport: Exception | None = None
        # Browser challenge (``hermes_cli.anon_challenge``). ``challenge_required`` makes the exchange
        # answer 428 until the ticket settles; each status poll consumes one entry of
        # ``challenge_statuses`` (then "none": a passed challenge detaches its ticket), and the
        # exchange goes through once the list is drained unless ``challenge_never_clears``.
        self.challenge_required = False
        self.challenge_never_clears = False
        self.challenge_statuses: list[str] = []
        # Status polls answered with this HTTP error first (a 429 / 5xx blip), one per entry.
        self.challenge_status_errors: list[int] = []
        self.challenge_url = f"{PORTAL}/challenge?code=ticket"
        self.optional_challenge = False
        self.token_requests: list[dict] = []

    def creates(self) -> int:
        return [p for _, p in self.calls].count("/api/anonymous/create")

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if self.raise_transport is not None:
            raise self.raise_transport
        if path.startswith("/api/anonymous/") and not request.headers.get("x-anonymous-api-secret"):
            return httpx.Response(401, json={"error": "invalid_shared_secret"})
        if self.gate_closed:
            return httpx.Response(401, json={"error": "invalid_shared_secret"})
        if path == "/api/anonymous/create":
            if self.create_response is not None:
                return self.create_response
            self.minted += 1
            return httpx.Response(201, json={"user_id": f"nas_user:{self.minted}", "org_id": "nas_org:1",
                                             "token": f"anon_{self.minted:04d}", "idle_ttl_days": 14})
        if path == "/api/anonymous/challenge/status":
            if self.challenge_status_errors:
                return httpx.Response(self.challenge_status_errors.pop(0), json={"error": "blip"})
            if self.challenge_statuses:
                return httpx.Response(200, json={"status": self.challenge_statuses.pop(0)})
            if self.challenge_never_clears:
                return httpx.Response(200, json={"status": "pending"})
            self.challenge_required = False
            return httpx.Response(200, json={"status": "none"})
        if path == "/api/anonymous/token":
            self.token_requests.append(
                {"body": json.loads(request.content), "user_agent": request.headers.get("user-agent")})
            if self.token_response is not None:
                return self.token_response
            if self.challenge_required:
                return httpx.Response(428, json={
                    "error": "challenge_required", "message": "A quick check first.",
                    "challenges": [{"type": "browser", "url": self.challenge_url, "state": "pending",
                                    "required": True, "expires_in": 600, "interval": 2}],
                    "fallback_url": f"{PORTAL}/login"})
            token = json.loads(request.content)["token"]
            if token in self.dead_tokens:
                return httpx.Response(404, json={"error": "unknown_token"})
            body = {"access_token": make_jwt(), "token_type": "Bearer", "expires_in": 900,
                    "user_id": "nas_user:1", "org_id": "nas_org:1"}
            if self.inference_base_url:
                body["inference_base_url"] = self.inference_base_url
            if self.optional_challenge:
                body["challenges"] = [{"type": "browser", "url": self.challenge_url, "state": "pending",
                                       "required": False, "expires_in": 600, "interval": 2}]
            return httpx.Response(200, json=body)
        return httpx.Response(500, json={"error": f"unexpected {path}"})


def install_portal(monkeypatch, tmp_path, fake: FakePortal | None = None) -> FakePortal:
    """Route every Nous HTTP client at *fake*, isolate the stores, and reset the per-process memos.

    One transport seam: ``httpx.Client`` itself, which ``auth_nous._nous_http_client`` and
    ``resolve_nous_access_token`` both construct."""
    from hermes_cli import anon_auth, free_tier_bootstrap
    from hermes_cli import auth as auth_mod

    fake = fake or FakePortal()
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", PORTAL)
    monkeypatch.setenv("HERMES_ANON_API_SECRET", "test-secret")
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NOUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    real_client = httpx.Client

    class _RoutedClient(real_client):
        def __init__(self, *a, **kw):
            kw.pop("verify", None)
            kw["transport"] = httpx.MockTransport(fake.handler)
            super().__init__(*a, **kw)
    monkeypatch.setattr(httpx, "Client", _RoutedClient)
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)
    anon_auth.reset_mint_memo_for_tests()
    free_tier_bootstrap.reset_for_tests()
    from hermes_cli import anon_challenge
    anon_challenge.reset_for_tests()
    monkeypatch.setattr(anon_challenge, "_sleep", lambda _seconds: None)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    # resolve_nous_access_token memoises the last token for 5 s per profile home (dict); a token minted
    # by an earlier test must not be served to this one.
    monkeypatch.setattr(auth_mod, "_RESOLVE_TOKEN_CACHE", {})
    return fake
