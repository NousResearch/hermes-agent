"""``hermes approvals test`` does not write config.yaml — byte identity, real command, real home.

The dry-run tester is documented and advertised for automation as "no execution, prompt or
persistence". Since the approval-required rules gained a revocation side effect (a review-policy
transition drops the superseded grant from the permanent allowlist and rewrites config.yaml),
composing the runtime evaluators naively made the diagnostic a *mutating* runtime event: it removed
the operator's allowlist entry, added ``_config_version`` and reserialized the file (PR #106779
review, F-003).

Contract: running the real command with a superseded persisted grant present reports the effective
post-reconciliation verdict AND leaves config.yaml byte-identical. Driven through a real temp
``HERMES_HOME`` and the real argparse entry point — no patched config loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pytest
import yaml

import hermes_cli.config as hc
import tools.approval as A
from hermes_cli import approvals_test as at
from tools import approval_floors

PATTERN = "kubectl *--context[= ]admin*"
KUBECTL = "kubectl --context admin -n vault exec vault-0 -- du -sh /vault/data"


@pytest.fixture
def home_with_superseded_grant(tmp_path, monkeypatch):
    """Temp ``HERMES_HOME`` whose config carries a ``human`` rule plus the Always grant a previous
    ``smart`` policy persisted — exactly the state the loader reconciles at runtime."""
    home = tmp_path / "hermes"
    home.mkdir()
    path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(A, "_permanent_approved", set())
    monkeypatch.setattr(A, "_permanent_approved_by_home", {})
    approval_floors._observed_review.clear()

    smart_key = approval_floors._RequiredRule(PATTERN, "kubectl on admin", "smart").key
    path.write_text(yaml.safe_dump({
        "approvals": {"mode": "manual", "command_approval_required": [
            {"pattern": PATTERN, "description": "kubectl on admin", "review": "human"}]},
        "security": {"tirith_enabled": False},
        "command_allowlist": [smart_key, "podman *"],
    }), encoding="utf-8")
    hc._LOAD_CONFIG_CACHE.clear()
    yield path, smart_key
    hc._LOAD_CONFIG_CACHE.clear()


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_approvals_test_leaves_config_byte_identical(home_with_superseded_grant, capsys):
    path, _ = home_with_superseded_grant
    before, before_sha = path.read_bytes(), _sha(path)

    rc = at.approvals_test_command(argparse.Namespace(
        command_words=[KUBECTL], env_type="local", json=True))

    assert rc == at.EXIT_ASK
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verdict"] == "ask-approval"
    assert verdict["rule"] == PATTERN
    assert path.read_bytes() == before, "the read-only tester rewrote config.yaml"
    assert _sha(path) == before_sha


def test_superseded_grant_is_ignored_without_being_revoked(home_with_superseded_grant, capsys):
    """The effective view the tester reports is the reconciled one — the superseded key does not
    silence the prompt — but the reconciliation stays a computation: the entry is still on disk and
    the process's permanent set is untouched, so the diagnostic never revokes an operator's grant."""
    path, smart_key = home_with_superseded_grant
    plain_command = "podman ps"

    rc = at.approvals_test_command(argparse.Namespace(
        command_words=[KUBECTL], env_type="local", json=True))
    assert rc == at.EXIT_ASK, "the smart-policy Always does not pre-approve the human rule"

    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))["command_allowlist"]
    assert smart_key in on_disk, "the diagnostic did not revoke the persisted grant"
    assert A._permanent_approved == set(), "and did not load the allowlist into this process"
    assert approval_floors._observed_review == {}, "and recorded no policy observation"

    capsys.readouterr()
    # Entries the rules do NOT supersede still allow, so the effective view is a real allowlist read.
    rc = at.approvals_test_command(argparse.Namespace(
        command_words=[plain_command], env_type="local", json=True))
    assert rc == at.EXIT_ALLOW
    assert "command_allowlist" in json.loads(capsys.readouterr().out)["detail"]
