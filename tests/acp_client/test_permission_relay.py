"""Tests for acp_client.permission_relay — deny-default + credential denylist."""

import os

import pytest

from acp_client.permission_relay import PermissionRelay


@pytest.fixture()
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def relay(workspace):
    return PermissionRelay(workspace_path=workspace)


def _inside(workspace, *parts):
    return os.path.join(workspace, *parts)


class TestDenyDefault:
    def test_execute_is_denied_by_default(self, relay, workspace):
        d = relay.evaluate(kind="execute", locations=[], raw_input={"command": "ls"})
        assert d.outcome == "deny"
        assert "default" in d.reason

    def test_unknown_kind_is_denied(self, relay):
        d = relay.evaluate(kind="fetch", locations=[], raw_input=None)
        assert d.outcome == "deny"

    def test_read_outside_workspace_is_denied(self, relay):
        d = relay.evaluate(kind="read", locations=["/etc/hosts"])
        assert d.outcome == "deny"
        assert "outside workspace" in d.reason

    def test_read_without_location_is_denied(self, relay):
        d = relay.evaluate(kind="read", locations=[])
        assert d.outcome == "deny"


class TestWorkspaceAllow:
    def test_read_inside_workspace_is_allowed_once(self, relay, workspace):
        d = relay.evaluate(kind="read", locations=[_inside(workspace, "src", "a.py")])
        assert d.outcome == "allow"
        assert d.option_id == "allow_once"

    def test_edit_inside_workspace_is_allowed(self, relay, workspace):
        d = relay.evaluate(kind="edit", locations=[_inside(workspace, "b.py")])
        assert d.outcome == "allow"

    def test_edit_denied_when_edits_disabled(self, workspace):
        relay = PermissionRelay(workspace_path=workspace, allow_workspace_edits=False)
        d = relay.evaluate(kind="edit", locations=[_inside(workspace, "b.py")])
        assert d.outcome == "deny"

    def test_partial_outside_location_denied(self, relay, workspace):
        d = relay.evaluate(
            kind="read",
            locations=[_inside(workspace, "ok.py"), "/etc/passwd"],
        )
        assert d.outcome == "deny"


class TestCredentialDenylist:
    @pytest.mark.parametrize(
        "path",
        [
            "auth.json",
            "auth.openai.json",
            ".env",
            "secrets.yaml",
            "state.db",
            "kanban.db",
            "id_rsa",
            ".netrc",
        ],
    )
    def test_credential_path_denied_even_inside_workspace(self, relay, workspace, path):
        # Located inside the workspace, but still denied by the credential rule.
        d = relay.evaluate(kind="read", locations=[_inside(workspace, path)])
        assert d.outcome == "deny"
        assert "credential" in d.reason

    def test_credential_marker_in_raw_input_denied(self, relay, workspace):
        d = relay.evaluate(
            kind="read",
            locations=[_inside(workspace, "safe.txt")],
            raw_input={"path": "~/.hermes/auth.json"},
        )
        assert d.outcome == "deny"
        assert "credential" in d.reason


class TestSelectOption:
    def test_allow_prefers_allow_once(self, relay, workspace):
        from types import SimpleNamespace

        d = relay.evaluate(kind="read", locations=[_inside(workspace, "a.py")])
        options = [
            SimpleNamespace(option_id="allow_once", kind="allow_once"),
            SimpleNamespace(option_id="allow_always", kind="allow_always"),
        ]
        assert relay.select_option(d, options) == "allow_once"

    def test_allow_refuses_to_escalate_to_persistent_only(self, relay, workspace):
        from types import SimpleNamespace

        d = relay.evaluate(kind="read", locations=[_inside(workspace, "a.py")])
        # Only a persistent option offered → never escalate; return None (deny).
        options = [SimpleNamespace(option_id="allow_always", kind="allow_always")]
        assert relay.select_option(d, options) is None

    def test_deny_prefers_reject_option(self, relay):
        from types import SimpleNamespace

        d = relay.evaluate(kind="execute", locations=[])
        options = [
            SimpleNamespace(option_id="reject_once", kind="reject_once"),
            SimpleNamespace(option_id="allow_once", kind="allow_once"),
        ]
        assert relay.select_option(d, options) == "reject_once"


class TestStats:
    def test_stats_count_allow_and_deny(self, relay, workspace):
        relay.evaluate(kind="read", locations=[_inside(workspace, "a.py")])
        relay.evaluate(kind="execute", locations=[])
        assert relay.stats == {"allowed": 1, "denied": 1}

    def test_audit_log_hook_receives_decisions(self, workspace):
        seen = []
        relay = PermissionRelay(workspace_path=workspace, audit_log=seen.append)
        relay.evaluate(kind="execute", locations=[])
        assert len(seen) == 1
        assert seen[0].outcome == "deny"
