"""Tests for optional-skills/blockchain/privy/scripts/privy_client.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import urllib.error
import urllib.request

import pytest  # ty: ignore[unresolved-import]

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "privy"
    / "scripts"
    / "privy_client.py"
)
SKILL_PATH = SCRIPT_PATH.parents[1] / "SKILL.md"


@pytest.fixture(scope="module")
def privy_client():
    spec = importlib.util.spec_from_file_location("privy_client", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_is_read_only_and_warns_against_signing():
    text = SKILL_PATH.read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    description = next(
        line.split(":", 1)[1].strip().strip('"')
        for line in frontmatter.splitlines()
        if line.startswith("description:")
    )
    author = next(
        line.split(":", 1)[1].strip()
        for line in frontmatter.splitlines()
        if line.startswith("author:")
    )

    assert "read-only" in text.lower()
    assert "does not sign" in text.lower()
    assert "human-approved" in text.lower()
    assert "PRIVY_APP_ID" in text
    assert "PRIVY_APP_SECRET" in text
    assert len(description) <= 60
    assert author.startswith("Patrick (@leprincep35700)")
    assert "## How to Run" in text
    assert "related_skills: [evm, solana]" in text
    assert "requires_toolsets: [terminal]" in text
    assert "required_environment_variables:" in frontmatter
    assert "name: PRIVY_APP_ID" in frontmatter
    assert "name: PRIVY_APP_SECRET" in frontmatter
    assert "credentials themselves are not read-only" in text.lower()
    assert "may authorize write operations elsewhere" in frontmatter.lower()
    assert "PRIVY_API_BASE" not in text
    assert "`base`" not in text.lower()
    assert "base/solana" not in text.lower()


def test_build_basic_auth_header_uses_app_id_and_secret(privy_client):
    assert (
        privy_client.build_basic_auth_header("app_123", "secret_456")
        == "Basic YXBwXzEyMzpzZWNyZXRfNDU2"
    )


def test_load_credentials_requires_app_id_and_secret(privy_client, monkeypatch):
    monkeypatch.delenv("PRIVY_APP_ID", raising=False)
    monkeypatch.delenv("PRIVY_APP_SECRET", raising=False)

    with pytest.raises(SystemExit) as exc:
        privy_client.load_credentials()

    assert "PRIVY_APP_ID" in str(exc.value)
    assert "PRIVY_APP_SECRET" in str(exc.value)


def test_extract_wallets_returns_only_wallet_linked_accounts(privy_client):
    user = {
        "id": "did:privy:abc",
        "linked_accounts": [
            {"type": "email", "address": "alice@example.com"},
            {
                "type": "wallet",
                "address": "0x1234567890abcdef1234567890abcdef12345678",
                "chain_type": "ethereum",
                "wallet_client_type": "privy",
                "connector_type": "embedded",
                "verified_at": 1710000000,
            },
            {
                "type": "wallet",
                "address": "So11111111111111111111111111111111111111112",
                "chain_type": "solana",
            },
        ],
    }

    assert privy_client.extract_wallets(user) == [
        {
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "chain_type": "ethereum",
            "wallet_client_type": "privy",
            "connector_type": "embedded",
            "verified_at": 1710000000,
        },
        {
            "address": "So11111111111111111111111111111111111111112",
            "chain_type": "solana",
            "wallet_client_type": None,
            "connector_type": None,
            "verified_at": None,
        },
    ]


def test_summarize_user_redacts_pii_but_keeps_wallets(privy_client):
    user = {
        "id": "did:privy:abc",
        "created_at": 1710000000,
        "linked_accounts": [
            {"type": "email", "address": "alice@example.com"},
            {"type": "phone", "phoneNumber": "+15551234567"},
            {
                "type": "wallet",
                "address": "0x1234567890abcdef1234567890abcdef12345678",
                "chain_type": "ethereum",
            },
        ],
    }

    summary = privy_client.summarize_user(user)

    assert summary["id"] == "did:privy:abc"
    assert summary["created_at"] == 1710000000
    assert summary["linked_account_types"] == ["email", "phone", "wallet"]
    assert summary["wallets"] == [
        {
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "chain_type": "ethereum",
            "wallet_client_type": None,
            "connector_type": None,
            "verified_at": None,
        }
    ]
    assert "alice@example.com" not in json.dumps(summary)
    assert "+15551234567" not in json.dumps(summary)


def test_build_users_query_encodes_listing_filters(privy_client):
    path = privy_client.build_users_query(limit=10, cursor="next cursor")
    assert path == "/v1/users?limit=10&cursor=next+cursor"


@pytest.mark.parametrize("limit", [0, 101])
def test_user_list_limit_is_bounded(privy_client, limit):
    with pytest.raises(ValueError, match="between 1 and 100"):
        privy_client.build_users_query(limit=limit)

    with pytest.raises(SystemExit):
        privy_client.build_parser().parse_args(["users", "--limit", str(limit)])


def test_arbitrary_api_base_environment_is_ignored(privy_client, monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"users":[]}'

    class Opener:
        def open(self, request, timeout):
            captured["request"] = request
            return Response()

    def fake_build_opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setenv("PRIVY_APP_ID", "app_123")
    monkeypatch.setenv("PRIVY_APP_SECRET", "secret_456")
    monkeypatch.setenv("PRIVY_API_BASE", "https://attacker.example")
    monkeypatch.setattr(privy_client.urllib.request, "build_opener", fake_build_opener)

    privy_client.request_json("/v1/users?limit=20", safe_route="/v1/users")

    assert captured["request"].full_url == "https://api.privy.io/v1/users?limit=20"
    assert any(
        isinstance(handler, privy_client.NoRedirectHandler)
        for handler in captured["handlers"]
    )


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_authenticated_redirects_are_never_followed_or_rebuilt(privy_client, status):
    request = urllib.request.Request(
        "https://api.privy.io/v1/users",
        headers={
            "Authorization": "Basic secret",
            "privy-app-id": "app_123",
        },
    )

    redirected = privy_client.NoRedirectHandler().redirect_request(
        request,
        None,
        status,
        "redirect",
        {},
        "https://attacker.example/collect",
    )

    assert redirected is None


def test_email_lookup_uses_documented_post_endpoint(privy_client, monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"id":"did:privy:abc","linked_accounts":[]}'

    class Opener:
        def open(self, request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    monkeypatch.setenv("PRIVY_APP_ID", "app_123")
    monkeypatch.setenv("PRIVY_APP_SECRET", "secret_456")
    monkeypatch.setattr(
        privy_client.urllib.request,
        "build_opener",
        lambda *_handlers: Opener(),
    )
    args = privy_client.build_parser().parse_args([
        "users",
        "--email",
        "alice@example.com",
    ])

    result = privy_client.cmd_users(args)

    request = captured["request"]
    assert request.get_method() == "POST"
    assert request.full_url == "https://api.privy.io/v1/users/email/address"
    assert json.loads(request.data) == {"address": "alice@example.com"}
    assert request.headers["Content-type"] == "application/json"
    assert result["users"][0]["id"] == "did:privy:abc"


@pytest.mark.parametrize(
    ("argv", "expected_error", "sensitive_values"),
    [
        (
            ["user", "did:privy:private-user"],
            "Privy API GET /v1/users/{user_id} failed",
            ["did:privy:private-user", "socket-secret"],
        ),
        (
            ["users", "--cursor", "private-cursor"],
            "Privy API GET /v1/users failed",
            ["private-cursor", "limit=20", "socket-secret"],
        ),
        (
            ["users", "--email", "private.person@example.com"],
            "Privy API POST /v1/users/email/address failed",
            ["private.person@example.com", "socket-secret"],
        ),
    ],
)
def test_request_failures_only_report_method_and_static_route(
    privy_client,
    monkeypatch,
    argv,
    expected_error,
    sensitive_values,
):
    monkeypatch.setenv("PRIVY_APP_ID", "app_123")
    monkeypatch.setenv("PRIVY_APP_SECRET", "secret_456")

    class FailingOpener:
        def open(self, *_args, **_kwargs):
            raise urllib.error.URLError("socket-secret")

    monkeypatch.setattr(
        privy_client.urllib.request,
        "build_opener",
        lambda *_handlers: FailingOpener(),
    )
    args = privy_client.build_parser().parse_args(argv)

    with pytest.raises(SystemExit) as exc:
        args.func(args)

    assert str(exc.value) == expected_error
    for value in sensitive_values:
        assert value not in str(exc.value)
