from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from hermes_cli.secure_github_broker import (
    ALLOWED_TOOL_NAMES,
    BrokerConfig,
    BrokerError,
    GitHubStagingBroker,
    create_mcp_server,
)


@pytest.fixture
def config(tmp_path: Path) -> BrokerConfig:
    return BrokerConfig(
        owner="private-staging",
        repository="ox-proposals",
        token="github_pat_staging_only_test_value",
        audit_path=tmp_path / "broker-audit.jsonl",
    )


def _broker(config: BrokerConfig, handler) -> GitHubStagingBroker:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.github.com",
        follow_redirects=False,
    )
    return GitHubStagingBroker(config, client=client)


def test_config_uses_only_dedicated_staging_token(tmp_path: Path) -> None:
    env = {
        "GH_TOKEN": "broad-host-token",
        "GITHUB_TOKEN": "another-ambient-token",
        "HERMES_STAGING_GITHUB_OWNER": "private-staging",
        "HERMES_STAGING_GITHUB_REPO": "ox-proposals",
        "HERMES_STAGING_GITHUB_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
    }
    with pytest.raises(BrokerError, match="HERMES_STAGING_GITHUB_TOKEN"):
        BrokerConfig.from_env(env)
    env["HERMES_STAGING_GITHUB_TOKEN"] = "github_pat_dedicated_staging_token_value"
    loaded = BrokerConfig.from_env(env)
    assert loaded.token == "github_pat_dedicated_staging_token_value"
    assert loaded.token != env["GH_TOKEN"]


def test_config_rejects_classic_broad_token_shape(tmp_path: Path) -> None:
    with pytest.raises(BrokerError, match="fine-grained"):
        BrokerConfig(
            owner="private-staging",
            repository="ox-proposals",
            token="ghp_broad_classic_token_value",
            audit_path=tmp_path / "audit.jsonl",
        )


def test_fixed_repository_is_compiled_into_request(config: BrokerConfig) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"name": "main"})

    broker = _broker(config, handler)
    broker.get_branch("main")
    assert seen[0].url == httpx.URL(
        "https://api.github.com/repos/private-staging/ox-proposals/branches/main"
    )
    assert "owner" not in GitHubStagingBroker.get_branch.__annotations__
    assert "repository" not in GitHubStagingBroker.get_branch.__annotations__


def test_repository_preflight_requires_exact_private_staging_identity(config: BrokerConfig) -> None:
    def allowed(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/actions/permissions"):
            return httpx.Response(200, json={"enabled": False})
        return httpx.Response(
            200,
            json={
                "full_name": "private-staging/ox-proposals",
                "private": True,
                "permissions": {"pull": True, "push": True, "admin": False},
            },
        )
    broker = _broker(
        config,
        allowed,
    )
    broker.verify_repository_boundary()

    for response in (
        {"full_name": "production-owner/private-repo", "private": True},
        {"full_name": "private-staging/ox-proposals", "private": False},
        {
            "full_name": "private-staging/ox-proposals",
            "private": True,
            "permissions": {"admin": True},
        },
    ):
        denied = _broker(config, lambda request, value=response: httpx.Response(200, json=value))
        with pytest.raises(BrokerError, match="boundary"):
            denied.verify_repository_boundary()


def test_repository_preflight_denies_enabled_actions(config: BrokerConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/actions/permissions"):
            return httpx.Response(200, json={"enabled": True})
        return httpx.Response(
            200,
            json={
                "full_name": "private-staging/ox-proposals",
                "private": True,
                "permissions": {"pull": True, "push": True, "admin": False},
            },
        )
    with pytest.raises(BrokerError, match="Actions"):
        _broker(config, handler).verify_repository_boundary()


@pytest.mark.parametrize(
    "value",
    ["../main", "refs/heads/main", "main?owner=real", "main#fragment", "", "a" * 256],
)
def test_branch_validation_denies_substitution(config: BrokerConfig, value: str) -> None:
    broker = _broker(config, lambda request: pytest.fail("network must not be reached"))
    with pytest.raises(BrokerError, match="branch"):
        broker.get_branch(value)


@pytest.mark.parametrize(
    "value", ["../secret", "/etc/passwd", ".git/config", "src/../secret", "a\\b.py"]
)
def test_path_validation_denies_escape(config: BrokerConfig, value: str) -> None:
    broker = _broker(config, lambda request: pytest.fail("network must not be reached"))
    with pytest.raises(BrokerError, match="path"):
        broker.get_file(value, "main")


@pytest.mark.parametrize(
    "value",
    [".github/workflows/ci.yml", ".github/actions/setup/action.yml", ".github/dependabot.yml"],
)
def test_write_denies_github_execution_control_paths_before_http(
    config: BrokerConfig, value: str
) -> None:
    broker = _broker(config, lambda request: pytest.fail("network must not be reached"))
    with pytest.raises(BrokerError, match="control path"):
        broker.put_file(value, "payload\n", "proposal", "hermes-proposal/task")


def test_writes_and_pull_heads_require_proposal_namespace_before_http(
    config: BrokerConfig,
) -> None:
    broker = _broker(config, lambda request: pytest.fail("network must not be reached"))
    with pytest.raises(BrokerError, match="proposal branch"):
        broker.put_file("src/logic.py", "payload\n", "proposal", "main")
    with pytest.raises(BrokerError, match="proposal branch"):
        broker.create_branch("main", "main")
    with pytest.raises(BrokerError, match="proposal branch"):
        broker.create_pull_request("proposal", "body", "main", "main")


def test_redirect_is_rejected_and_token_is_not_in_error(config: BrokerConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.invalid/upload"})

    broker = _broker(config, handler)
    with pytest.raises(BrokerError) as exc_info:
        broker.get_branch("main")
    assert "evil.invalid" not in str(exc_info.value)
    assert config.token not in str(exc_info.value)


def test_put_file_uses_fixed_contents_route_and_typed_payload(config: BrokerConfig) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(201, json={"commit": {"sha": "abc123"}})

    broker = _broker(config, handler)
    broker.put_file(
        "src/logic.py", "print('safe')\n", "proposal", "hermes-proposal/task"
    )
    assert captured["method"] == "PUT"
    assert captured["url"] == (
        "https://api.github.com/repos/private-staging/ox-proposals/contents/src/logic.py"
    )
    assert captured["json"] == {
        "branch": "hermes-proposal/task",
        "content": "cHJpbnQoJ3NhZmUnKQo=",
        "message": "proposal",
    }


def test_create_branch_reads_source_then_creates_fixed_ref(config: BrokerConfig) -> None:
    calls: list[tuple[str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, payload))
        if request.method == "GET":
            return httpx.Response(200, json={"object": {"sha": "a" * 40}})
        return httpx.Response(201, json={"ref": "refs/heads/ox/task"})

    broker = _broker(config, handler)
    broker.create_branch("hermes-proposal/task", "main")
    assert calls == [
        (
            "GET",
            "/repos/private-staging/ox-proposals/git/ref/heads/main",
            None,
        ),
        (
            "POST",
            "/repos/private-staging/ox-proposals/git/refs",
            {"ref": "refs/heads/hermes-proposal/task", "sha": "a" * 40},
        ),
    ]


def test_issue_pull_request_and_review_are_typed(config: BrokerConfig) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(201, json={"number": 7})

    broker = _broker(config, handler)
    broker.create_issue("audit", "body")
    broker.create_pull_request("proposal", "body", "hermes-proposal/task", "main")
    broker.create_review(7, "APPROVE", "reviewed")
    assert calls == [
        "/repos/private-staging/ox-proposals/issues",
        "/repos/private-staging/ox-proposals/pulls",
        "/repos/private-staging/ox-proposals/pulls/7/reviews",
    ]
    with pytest.raises(BrokerError, match="review event"):
        broker.create_review(7, "EXECUTE", "no")


def test_audit_receipt_contains_no_model_content_or_token(config: BrokerConfig) -> None:
    broker = _broker(config, lambda request: httpx.Response(201, json={"number": 1}))
    secret_body = "private proposal text that must not enter the receipt"
    broker.create_issue("sensitive title", secret_body)
    receipt = config.audit_path.read_text()
    assert secret_body not in receipt
    assert "sensitive title" not in receipt
    assert config.token not in receipt
    row = json.loads(receipt)
    assert row["operation"] == "create_issue"
    assert row["repository_id"]
    assert set(row) == {"schema", "timestamp", "operation", "status", "repository_id"}


def test_mcp_surface_has_no_arbitrary_http_graphql_or_shell_tool(config: BrokerConfig) -> None:
    assert ALLOWED_TOOL_NAMES == {
        "staging_branch_get",
        "staging_branch_create",
        "staging_file_get",
        "staging_file_put",
        "staging_issue_create",
        "staging_pull_request_create",
        "staging_review_create",
    }
    assert not any(
        forbidden in name
        for name in ALLOWED_TOOL_NAMES
        for forbidden in ("http", "url", "graphql", "shell", "api")
    )
    server = create_mcp_server(_broker(config, lambda request: httpx.Response(200, json={})))
    assert server is not None
