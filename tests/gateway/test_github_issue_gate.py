import hashlib
import hmac
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.github_issue_gate import evaluate_github_issue_gate
from gateway.platforms.webhook import WebhookAdapter


SECRET = "github-secret"


def _signature(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _adapter(route):
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 0, "routes": {"github-issue-worker": route}},
        )
    )


def _app(adapter):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _issue_payload(**overrides):
    payload = {
        "action": "opened",
        "repository": {"full_name": "ozwicked/trading_bot"},
        "sender": {"login": "human-user"},
        "issue": {
            "number": 47,
            "title": "Fix trade trail exits",
            "body": "Please fix this.",
            "labels": [{"name": "bot-ready"}, {"name": "trading-bot"}],
        },
    }
    payload.update(overrides)
    return payload


def _route(**gate_overrides):
    gate = {
        "repositories": ["ozwicked/trading_bot", "ozwicked/grow-journal-pro"],
        "labels_all": ["bot-ready"],
        "labels_any": ["trading-bot", "growpro"],
        "self_users": ["ozwicked", "github-actions[bot]"],
        "self_markers": ["<!-- hermes:automation -->"],
    }
    gate.update(gate_overrides)
    return {
        "secret": SECRET,
        "events": ["issues", "issue_comment", "pull_request", "push"],
        "github_issue_gate": gate,
        "prompt": "Process {repository.full_name} #{issue.number}: {issue.title}",
        "deliver": "log",
    }


class TestGitHubIssueGate:
    def test_accepts_relevant_labeled_issue(self):
        decision = evaluate_github_issue_gate(_route()["github_issue_gate"], _issue_payload(), "issues")
        assert decision.keep is True
        assert decision.reason == "matched"
        assert decision.repo == "ozwicked/trading_bot"
        assert decision.number == 47
        assert "issue opened" in decision.llm_reason

    def test_rejects_issue_without_required_labels_before_llm(self):
        payload = _issue_payload(issue={"number": 48, "labels": [{"name": "bug"}]})
        decision = evaluate_github_issue_gate(_route()["github_issue_gate"], payload, "issues")
        assert decision.keep is False
        assert decision.reason == "missing_required_labels"

    def test_rejects_self_sender(self):
        decision = evaluate_github_issue_gate(
            _route()["github_issue_gate"],
            _issue_payload(sender={"login": "ozwicked"}),
            "issues",
        )
        assert decision.keep is False
        assert decision.reason == "self_sender"

    def test_rejects_internal_marker_to_prevent_self_trigger_loops(self):
        payload = _issue_payload(
            issue={
                "number": 47,
                "body": "<!-- hermes:automation -->\nPR opened by automation.",
                "labels": [{"name": "bot-ready"}, {"name": "trading-bot"}],
            }
        )
        decision = evaluate_github_issue_gate(_route()["github_issue_gate"], payload, "issues")
        assert decision.keep is False
        assert decision.reason == "self_marker"

    @pytest.mark.asyncio
    async def test_signature_gate_accepts_valid_github_signature_and_dispatches_once(self):
        adapter = _adapter(_route())
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        body = json.dumps(_issue_payload()).encode()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(
                "/webhooks/github-issue-worker",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "issues",
                    "X-GitHub-Delivery": "delivery-1",
                    "X-Hub-Signature-256": _signature(body),
                },
            )
            assert resp.status == 202
            assert (await resp.json())["status"] == "accepted"

        assert len(captured) == 1
        event = captured[0]
        assert event.message_id == "delivery-1"
        assert event.raw_message["__hermes_webhook"]["delivery_id"] == "delivery-1"
        assert event.raw_message["__hermes_github_gate"]["llm_reason"]

    @pytest.mark.asyncio
    async def test_irrelevant_event_is_filtered_before_agent_dispatch(self):
        adapter = _adapter(_route())
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        payload = _issue_payload(issue={"number": 49, "labels": [{"name": "bug"}]})
        body = json.dumps(payload).encode()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(
                "/webhooks/github-issue-worker",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "issues",
                    "X-GitHub-Delivery": "delivery-filtered",
                    "X-Hub-Signature-256": _signature(body),
                },
            )
            assert resp.status == 200
            assert await resp.json() == {"status": "ignored", "reason": "missing_required_labels"}

        assert captured == []

    @pytest.mark.asyncio
    async def test_duplicate_delivery_does_not_dispatch_second_agent_run(self):
        adapter = _adapter(_route())
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        body = json.dumps(_issue_payload()).encode()
        headers = {
            "Content-Type": "application/json",
            "X-GitHub-Event": "issues",
            "X-GitHub-Delivery": "same-delivery",
            "X-Hub-Signature-256": _signature(body),
        }
        async with TestClient(TestServer(_app(adapter))) as cli:
            first = await cli.post("/webhooks/github-issue-worker", data=body, headers=headers)
            second = await cli.post("/webhooks/github-issue-worker", data=body, headers=headers)
            assert first.status == 202
            assert second.status == 200
            assert (await second.json())["status"] == "duplicate"

        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_bad_signature_rejected_before_filter_or_agent(self):
        adapter = _adapter(_route())
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        body = json.dumps(_issue_payload()).encode()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(
                "/webhooks/github-issue-worker",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "issues",
                    "X-GitHub-Delivery": "bad-sig",
                    "X-Hub-Signature-256": "sha256=bad",
                },
            )
            assert resp.status == 401

        assert captured == []
