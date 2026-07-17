"""Regression tests for runtime admission policy (Gate 1 reconciliation).

These tests pin the fail-closed contract after the prior correction removed
the over-broad `private` path-part token. The correction must NOT open a hole:
arbitrary /private/tmp paths stay denied, while the single approved isolated
recovery root is permitted.

Run:  python tests/agent/test_admission_policy.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path("/Users/jon/.hermes/hermes-agent")
RUNTIME_POLICY = Path("/Users/jon/.hermes/policies/agent-policies.json")
ISOLATED_RESTORE_ROOT = "/private/tmp/hermes-isolated-restore-20260716T"

# Import the production module via the repo on sys.path (proper module context
# so dataclasses resolve correctly).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import agent.admission_policy as ap  # noqa: E402


def _make_agent(contract):
    class _Agent:
        _admission_required = True
        _admission_policy = ap.RuntimeAgentPolicy(
            canonical_id="saint",
            version="test",
            policy_hash="abc",
            contract=ap._frozen(dict(contract)),
        )
        provider = "nous"
        model = "tencent/hy3:free"

    return _Agent()


_BASE = {
    "allowed_tools": ["read_file", "write_file", "delegate_task", "send_email"],
    "write_tools": ["write_file"],
    "allowed_read_roots": [
        "/Users/jon/HermesVault/novaaaaa",
        "/Users/jon/.hermes",
        ISOLATED_RESTORE_ROOT,
    ],
    "allowed_write_roots": [
        "/Users/jon/HermesVault/novaaaaa",
        "/Users/jon/.hermes",
        ISOLATED_RESTORE_ROOT,
    ],
    "allowed_remote_hosts": [],
    "allowed_delegation_targets": ["nova"],
    "approval_classes": ["external_send"],
}


def _auth(tool, args, contract=None):
    return ap.authorize_tool_call(_make_agent(contract or _BASE), tool, args)


def test_approved_isolated_recovery_root_allowed():
    d = _auth("read_file", {"path": f"{ISOLATED_RESTORE_ROOT}/snapshot.db"})
    assert d.allowed, d.reason
    d2 = _auth("write_file", {"path": f"{ISOLATED_RESTORE_ROOT}/receipt.json"})
    assert d2.allowed, d2.reason


def test_arbitrary_private_tmp_denied():
    d = _auth("read_file", {"path": "/private/tmp/something-else.txt"})
    assert not d.allowed, "arbitrary /private/tmp must be denied"
    assert "outside declared scope" in d.reason, d.reason


def test_sensitive_name_denied():
    d = _auth(
        "read_file",
        {"path": "/Users/jon/HermesVault/novaaaaa/_system/control-plane/agent-policies.json"},
    )
    assert not d.allowed, "protected policy file must be denied"
    assert "secret/policy boundary" in d.reason, d.reason


def test_sensitive_part_denied():
    d = _auth("read_file", {"path": "/Users/jon/HermesVault/novaaaaa/secrets/key.txt"})
    assert not d.allowed, "secrets path part must be denied"


def test_production_state_outside_scope_denied():
    d = _auth("read_file", {"path": "/Users/jon/Documents/prod-state/db.sqlite"})
    assert not d.allowed, "production state path outside roots must be denied"


def test_undeclared_tool_denied():
    d = _auth("rm_file", {"path": "/Users/jon/HermesVault/novaaaaa/x.txt"})
    assert not d.allowed and "not declared" in d.reason, d.reason


def test_forbidden_delegation_denied():
    d = _auth("delegate_task", {"agent": "wrench"})
    assert not d.allowed and "forbidden" in d.reason, d.reason


def test_external_action_requires_approval():
    d = _auth("send_email", {"approval_class": "external_send", "approved": False})
    assert not d.allowed, "unapproved external action must be denied"
    d2 = _auth("send_email", {"approval_class": "external_send", "approved": True})
    assert d2.allowed, d2.reason


def test_arbitrary_custom_provider_denied_at_attach():
    policy_json = {
        "version": "t",
        "agents": [
            {
                "canonical_id": "saint",
                "supervisor": "root",
                "allowed_providers": ["nous"],
                "allowed_models": ["tencent/hy3:free"],
            }
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
        json.dump(policy_json, fh)
        fh.flush()
        pol = ap.load_runtime_policy(fh.name, "saint")

        class _FakeAgent:
            provider = "some-custom-provider"
            model = "tencent/hy3:free"

        try:
            ap.attach_runtime_policy(
                _FakeAgent(),
                {"admission": {"enabled": True, "policy_path": fh.name, "identity": "saint"}},
            )
        except ap.AdmissionPolicyError as exc:
            assert "not declared" in str(exc), str(exc)
            return
    raise AssertionError("expected AdmissionPolicyError for custom provider")


def test_runtime_policy_narrows_recovery_root():
    """Live deployed policy must not declare bare /private/tmp."""
    doc = json.loads(RUNTIME_POLICY.read_text())
    saint = next(a for a in doc["agents"] if a["canonical_id"] == "saint")
    for root in (
        saint.get("allowed_read_roots", []) + saint.get("allowed_write_roots", [])
    ):
        assert root != "/private/tmp", "bare /private/tmp still declared as a root"
        if "private" in root:
            assert root.startswith(
                ISOLATED_RESTORE_ROOT
            ), f"unexpected private root: {root}"


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in funcs:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL", fn.__name__, "->", repr(exc))
    sys.exit(1 if failures else 0)
