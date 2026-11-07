"""Gate tolerance for FORBIDDEN on baseRef.branchProtectionRule (t_08d0f5d9).

The aiwl-reviewer App's installation token lacks Administration:read, so the
PR query comes back as data + an errors array on exactly that path and gh
exits non-zero. The gate must classify by the collected check evidence, not
degrade to infra; any other error shape must still fail closed.
"""
import json
import subprocess

from hermes_cli import kanban_pr_acceptance as gate

SHA = "5f28b50f00000000000000000000000000000000"
PR_URL = "https://github.com/ydengGitHub/aiwl-workstation/pull/1467"

FORBIDDEN = {"type": "FORBIDDEN",
             "path": ["repository", "pullRequest", "baseRef", "branchProtectionRule"],
             "message": "Resource not accessible by integration"}
EXTRA = {"type": "NOT_FOUND", "path": ["repository", "pullRequest"],
         "message": "Could not resolve to repository"}


def _fake_gh(calls, payload_box):
    """Stand in for `gh api`: mirrors the measured App-token shape — GraphQL
    partial data on stdout, non-zero exit, gh's own stderr line."""
    def run(command, **kwargs):
        endpoint = command[2]
        calls.append(endpoint)
        if endpoint == "graphql":
            payload = {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "MERGED",
                "baseRef": {"branchProtectionRule": None}}}},
                "errors": payload_box[0]}
            return subprocess.CompletedProcess(command, 1, json.dumps(payload),
                                               "gh: Resource not accessible by integration\n")
        if endpoint.startswith("repos/ydengGitHub/aiwl-workstation/rules/branches/main"):
            out = [[{"type": "required_status_checks",
                     "parameters": {"required_status_checks": [{"context": "local-verify"}]}}]]
        elif endpoint.startswith("repos/ydengGitHub/aiwl-workstation/commits/") and endpoint.endswith("check-runs?per_page=100&filter=latest"):
            out = [{"total_count": 1, "page_count": 1, "check_runs": [{"id": 9, "name": "local-verify",
                    "head_sha": SHA, "status": "completed", "conclusion": "success",
                    "html_url": "https://example.invalid/verify"}]}]
        elif endpoint.startswith("repos/ydengGitHub/aiwl-workstation/commits/") and endpoint.endswith("statuses?per_page=100"):
            out = [[]]  # no legacy statuses: local-verify is satisfied by the check-run only
        elif endpoint == "repos/ydengGitHub/aiwl-workstation/pulls/1467":
            out = {"head": {"sha": SHA}, "base": {"ref": "main"}, "state": "closed", "merged": True}
        else:  # unknown endpoint: let the gate fall through to infra and fail loudly here
            raise AssertionError(f"unexpected gh endpoint {endpoint!r}")
        return subprocess.CompletedProcess(command, 0, json.dumps(out), "")
    return run


def test_forbidden_protection_classifies_by_checks(monkeypatch):
    monkeypatch.setattr(gate, "_gh_env", lambda profile_home: None)
    calls = []
    payload_box = [[FORBIDDEN]]
    monkeypatch.setattr(subprocess, "run", _fake_gh(calls, payload_box))

    # The measured payload: only the branchProtectionRule read is FORBIDDEN.
    receipt = gate.collect_acceptance(PR_URL, PR_URL)
    assert receipt["ok"] is True
    assert receipt["classification"] == "success"
    assert receipt["head_sha"] == SHA
    assert receipt["checks"] == [{"name": "local-verify", "id": 9,
                                  "url": "https://example.invalid/verify",
                                  "head_sha": SHA, "classification": "success",
                                  "conclusion": "success"}]

    # Any other error in the same response still fails closed (infra, no receipts).
    payload_box[0] = [FORBIDDEN, EXTRA]
    calls.clear()
    receipt = gate.collect_acceptance(PR_URL, PR_URL)
    assert receipt["ok"] is False
    assert receipt["classification"] == "infra"
    assert receipt["head_sha"] is None and receipt["checks"] == []
    assert calls == ["graphql"]  # refused before consulting any other endpoint
