"""Durable Kavera SEO PR approval store and slash-surface tests."""

from __future__ import annotations

import importlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _load_module(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.seo_pr_approvals as spa

    return importlib.reload(spa)


@pytest.fixture()
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _sample_payload(**overrides):
    payload = {
        "site": "Kavera SEO Demo",
        "route": "/services/test-page",
        "target_keyword": "test page agency",
        "repo": "KaveraAI/kavera-seo-pages",
        "pr_url": "https://github.com/KaveraAI/kavera-seo-pages/pull/42",
        "pr_number": 42,
        "branch": "seo/test-page",
        "preview_url": "https://kavera-seo-pages-git-seo-test-page.vercel.app/services/test-page",
        "checks_summary": "3 passed",
        "checks_status": "passed",
        "source_platform": "telegram",
        "source_chat_id": "12345",
        "source_thread_id": "777",
        "merge_payload": {"dry_run": True, "merge_method": "squash"},
    }
    payload.update(overrides)
    return payload


def test_store_persists_pending_approval_across_store_instances(hermes_home, monkeypatch):
    spa = _load_module(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()

    approval_id = store.create(_sample_payload())

    restarted_store = spa.SEOApprovalStore()
    pending = restarted_store.list(status="pending")
    assert [item.approval_id for item in pending] == [approval_id]
    assert pending[0].site == "Kavera SEO Demo"
    assert pending[0].source_chat_id == "12345"
    assert pending[0].merge_payload["dry_run"] is True


def test_stale_approval_is_marked_expired_without_merging(hermes_home, monkeypatch):
    spa = _load_module(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()
    expired_at = datetime.now(timezone.utc) - timedelta(hours=1)
    approval_id = store.create(_sample_payload(expires_at=expired_at.isoformat()))

    result = store.approve(approval_id, actor="arthur")

    assert result.ok is False
    assert "expired" in result.message.lower()
    assert store.get(approval_id).status == "expired"


def test_duplicate_click_does_not_reapprove_consumed_item(hermes_home, monkeypatch):
    spa = _load_module(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()
    approval_id = store.create(_sample_payload())

    first = store.approve(approval_id, actor="arthur")
    second = store.approve(approval_id, actor="arthur")

    assert first.ok is True
    assert store.get(approval_id).status == "consumed"
    assert second.ok is False
    assert "already" in second.message.lower()


def test_validation_rejects_wrong_pr_branch_before_merge(hermes_home, monkeypatch):
    spa = _load_module(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()
    approval_id = store.create(_sample_payload(branch="seo/original-branch"))
    item = store.get(approval_id)

    validation = spa.validate_pr_snapshot(
        item,
        {
            "number": 42,
            "url": item.pr_url,
            "state": "OPEN",
            "isDraft": False,
            "headRefName": "seo/different-branch",
            "baseRefName": "main",
            "statusCheckRollup": [],
        },
        files=[],
    )

    assert validation.ok is False
    assert "branch" in validation.message.lower()


def test_run_slash_demo_create_hold_and_reload_persists(hermes_home, monkeypatch):
    spa = _load_module(hermes_home, monkeypatch)

    created = spa.run_slash("demo create")
    match = re.search(r"(apr_[A-Za-z0-9_-]+)", created)
    assert match, created
    approval_id = match.group(1)

    # Simulate a process/module restart before the user acts hours later.
    spa = importlib.reload(spa)
    listed = spa.run_slash("list")
    assert approval_id in listed
    assert "Kavera SEO Demo" in listed

    held = spa.run_slash(f"hold {approval_id} Needs stronger local intent")
    assert "held" in held.lower()
    assert spa.SEOApprovalStore().get(approval_id).status == "held"


def test_run_slash_batch_approve_all_consumes_pending_safe_dry_runs(hermes_home, monkeypatch):
    spa = _load_module(hermes_home, monkeypatch)
    store = spa.SEOApprovalStore()
    first = store.create(_sample_payload(route="/services/one", branch="seo/one", pr_number=101, pr_url="https://github.com/KaveraAI/kavera-seo-pages/pull/101"))
    second = store.create(_sample_payload(route="/services/two", branch="seo/two", pr_number=102, pr_url="https://github.com/KaveraAI/kavera-seo-pages/pull/102"))

    output = spa.run_slash("approve all")

    assert "Approved 2" in output
    assert store.get(first).status == "consumed"
    assert store.get(second).status == "consumed"
    assert spa.SEOApprovalStore().list(status="pending") == []


def test_approvals_slash_command_is_registered_for_cli_and_gateway():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    cmd = resolve_command("approvals")
    assert cmd is not None
    assert cmd.name == "approvals"
    assert cmd.cli_only is False
    assert resolve_command("seo-approve").name == "approvals"
    assert "approvals" in GATEWAY_KNOWN_COMMANDS
    assert "seo-approve" in GATEWAY_KNOWN_COMMANDS
