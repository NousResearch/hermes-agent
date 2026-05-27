"""Tests for GitHub PR label-command webhook routing."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, extra=extra or {"secret": "test-secret"})
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)  # noqa: SLF001
    return app


def _payload(label="hermes-review", action="labeled"):
    return {
        "action": action,
        "number": 42,
        "label": {"name": label},
        "repository": {"full_name": "owner/repo", "private": True},
        "pull_request": {
            "title": "Add feature",
            "body": "Please review this change.",
            "html_url": "https://github.com/owner/repo/pull/42",
            "user": {"login": "alice"},
            "head": {"ref": "feat/example", "sha": "abc123"},
            "base": {"ref": "main"},
        },
    }


def _prepare(adapter, payload, route=None, event_type="pull_request"):
    return adapter._prepare_github_pr_label_command(  # noqa: SLF001 - focused adapter unit test
        route or {"github_pr_label_commands": True}, payload, event_type
    )


class TestGitHubPrLabelCommands:
    def test_ignores_unlabeled_pull_request_actions(self):
        adapter = _make_adapter()

        prepared = _prepare(adapter, _payload(action="opened"))

        assert prepared is None

    def test_ignores_unconfigured_label(self):
        adapter = _make_adapter()

        prepared = _prepare(adapter, _payload(label="needs-review"))

        assert prepared is None

    def test_ignores_non_pull_request_events_even_with_matching_shape(self):
        adapter = _make_adapter()

        prepared = _prepare(adapter, _payload(label="hermes-review"), event_type="issues")

        assert prepared is None

    def test_ignores_payloads_without_pull_request_object(self):
        adapter = _make_adapter()
        payload = _payload(label="hermes-review")
        payload.pop("pull_request")

        prepared = _prepare(adapter, payload)

        assert prepared is None

    def test_invalid_custom_label_definition_is_ignored(self):
        adapter = _make_adapter()
        route = {"github_pr_label_commands": {"labels": {"ship-it": "deploy"}}}

        prepared = _prepare(adapter, _payload(label="ship-it"), route=route)

        assert prepared is None

    def test_malformed_repository_or_label_payload_is_ignored(self):
        adapter = _make_adapter()
        payload = _payload(label="hermes-review")
        payload["repository"] = "owner/repo"
        payload["label"] = "hermes-review"

        prepared = _prepare(adapter, payload)

        assert prepared is None

    def test_malformed_label_payload_is_ignored_without_error(self):
        adapter = _make_adapter()
        payload = _payload(label="hermes-review")
        payload["label"] = "hermes-review"

        prepared = _prepare(adapter, payload)

        assert prepared is None

    def test_review_label_builds_read_only_review_prompt_and_github_delivery(self):
        adapter = _make_adapter()

        prepared = _prepare(adapter, _payload(label="hermes-review"))

        assert prepared is not None
        route, prompt = prepared
        assert route["deliver"] == "github_comment"
        assert route["deliver_extra"] == {"repo": "owner/repo", "pr_number": "42"}
        assert "hermes-review" in prompt
        assert "READ-ONLY" in prompt
        assert "gh pr diff 42 --repo owner/repo" in prompt
        assert "Do not commit" in prompt
        assert "https://github.com/owner/repo/pull/42" in prompt

    def test_autofix_label_builds_guarded_fix_prompt(self):
        adapter = _make_adapter()

        route = {"github_pr_label_commands": {"allowed_modes": ["autofix"]}}
        _, prompt = _prepare(adapter, _payload(label="hermes-autofix"), route=route)

        assert "hermes-autofix" in prompt
        assert "commit" in prompt.lower()
        assert "push" in prompt.lower()
        assert "Do not merge" in prompt
        assert "Run targeted tests" in prompt

    def test_automerge_label_requires_green_ci_and_no_code_changes(self):
        adapter = _make_adapter()

        route = {"github_pr_label_commands": {"allowed_modes": ["automerge"]}}
        _, prompt = _prepare(adapter, _payload(label="hermes-automerge"), route=route)

        assert "hermes-automerge" in prompt
        assert "CI" in prompt
        assert "merge" in prompt.lower()
        assert "Do not modify files" in prompt
        assert "squash" in prompt.lower()

    def test_deploy_label_uses_existing_hermes_deploy_conventions(self):
        adapter = _make_adapter()

        route = {"github_pr_label_commands": {"allowed_modes": ["deploy"]}}
        _, prompt = _prepare(adapter, _payload(label="hermes-deploy"), route=route)

        assert "hermes-deploy" in prompt
        assert "repo-specific deploy procedure" in prompt
        assert "scripts/deploy-prod.sh" in prompt
        assert "Docker" in prompt
        assert "smoke" in prompt.lower()

    def test_custom_labels_can_override_defaults(self):
        adapter = _make_adapter()
        route = {
            "github_pr_label_commands": {
                "allowed_modes": ["deploy"],
                "labels": {
                    "ship-it": {
                        "mode": "deploy",
                        "prompt": "Deploy PR {number} from {repository.full_name}",
                    }
                }
            }
        }

        prepared = _prepare(adapter, _payload(label="ship-it"), route=route)

        assert prepared is not None
        _, prompt = prepared
        assert prompt == "Deploy PR 42 from owner/repo"

    def test_custom_label_allowlist_disables_default_labels(self):
        adapter = _make_adapter()
        route = {"github_pr_label_commands": {"labels": {"ship-it": {"mode": "deploy"}}}}

        prepared = _prepare(adapter, _payload(label="hermes-review"), route=route)

        assert prepared is None

    def test_public_repositories_are_ignored_by_default(self):
        adapter = _make_adapter()
        route = {"github_pr_label_commands": {"allowed_modes": ["deploy"]}}
        payload = _payload(label="hermes-deploy")
        payload["repository"]["private"] = False

        prepared = _prepare(adapter, payload, route=route)

        assert prepared is None

    def test_public_repositories_require_explicit_opt_in(self):
        adapter = _make_adapter()
        payload = _payload(label="hermes-review")
        payload["repository"]["private"] = False
        route = {"github_pr_label_commands": {"allow_public_repositories": True, "allowed_modes": ["review"]}}

        prepared = _prepare(adapter, payload, route=route)

        assert prepared is not None

    def test_sensitive_modes_require_explicit_opt_in(self):
        adapter = _make_adapter()

        assert _prepare(adapter, _payload(label="hermes-autofix")) is None
        assert _prepare(adapter, _payload(label="hermes-automerge")) is None
        assert _prepare(adapter, _payload(label="hermes-deploy")) is None

    def test_allowed_sender_allowlist_blocks_other_labelers(self):
        adapter = _make_adapter()
        payload = _payload(label="hermes-review")
        payload["sender"] = {"login": "mallory"}
        route = {"github_pr_label_commands": {"allowed_senders": ["theo"]}}

        prepared = _prepare(adapter, payload, route=route)

        assert prepared is None

    def test_allowed_sender_allowlist_accepts_authorized_labeler(self):
        adapter = _make_adapter()
        payload = _payload(label="hermes-review")
        payload["sender"] = {"login": "theo"}
        route = {"github_pr_label_commands": {"allowed_senders": ["theo"]}}

        prepared = _prepare(adapter, payload, route=route)

        assert prepared is not None

    def test_partial_github_comment_delivery_extra_gets_payload_defaults(self):
        adapter = _make_adapter()
        route = {
            "github_pr_label_commands": True,
            "deliver": "github_comment",
            "deliver_extra": {"repo": "owner/repo"},
        }

        prepared = _prepare(adapter, _payload(label="hermes-review"), route=route)

        assert prepared is not None
        prepared_route, _ = prepared
        assert prepared_route["deliver_extra"] == {"repo": "owner/repo", "pr_number": "42"}

    @pytest.mark.asyncio
    async def test_non_pull_request_webhook_is_ignored_even_without_event_filter(self):
        adapter = _make_adapter(
            {
                "routes": {
                    "github-pr-labels": {
                        "secret": _INSECURE_NO_AUTH,
                        "github_pr_label_commands": True,
                    }
                }
            }
        )
        adapter.handle_message = AsyncMock()
        issue_payload = {
            "action": "labeled",
            "number": 42,
            "label": {"name": "hermes-review"},
            "repository": {"full_name": "owner/repo", "private": True},
            "issue": {"title": "not a PR"},
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/github-pr-labels",
                json=issue_payload,
                headers={"X-GitHub-Event": "issues", "X-GitHub-Delivery": "d0"},
            )
            data = await resp.json()

        assert resp.status == 200
        assert data["status"] == "ignored"
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_labeled_webhook_dispatches_agent_with_label_prompt(self):
        adapter = _make_adapter(
            {
                "routes": {
                    "github-pr-labels": {
                        "secret": _INSECURE_NO_AUTH,
                        "events": ["pull_request"],
                        "github_pr_label_commands": True,
                    }
                }
            }
        )
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/github-pr-labels",
                json=_payload(label="hermes-review"),
                headers={"X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "d1"},
            )

        assert resp.status == 202
        adapter.handle_message.assert_awaited_once()
        await_args = adapter.handle_message.await_args
        assert await_args is not None
        event = await_args.args[0]
        assert "Mode: READ-ONLY review" in event.text
        assert adapter._delivery_info[event.source.chat_id]["deliver"] == "github_comment"  # noqa: SLF001
        assert adapter._delivery_info[event.source.chat_id]["deliver_extra"] == {  # noqa: SLF001
            "repo": "owner/repo",
            "pr_number": "42",
        }

    @pytest.mark.asyncio
    async def test_unconfigured_label_is_ignored_without_agent_dispatch(self):
        adapter = _make_adapter(
            {
                "routes": {
                    "github-pr-labels": {
                        "secret": _INSECURE_NO_AUTH,
                        "events": ["pull_request"],
                        "github_pr_label_commands": True,
                    }
                }
            }
        )
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhooks/github-pr-labels",
                json=_payload(label="random-label"),
                headers={"X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "d2"},
            )
            data = await resp.json()

        assert resp.status == 200
        assert data["status"] == "ignored"
        assert data["label"] == "random-label"
        adapter.handle_message.assert_not_awaited()
