"""Contract and model tests for gateway.runtime.models.

Covers:
- RuntimeEvent serialization shape
- RuntimeStatus serialization shape
- event_id stability
- Secret redaction (recursive)
- Module import contract
"""

from gateway.runtime.models import (
    RuntimeEvent,
    RuntimeStatus,
    redact_secrets,
    RUN_STATUS_QUEUED,
    RUN_STATUS_CANCELLED,
    TERMINAL_STATUSES,
    EVENT_RUN_STARTED,
    EVENT_DONE,
    EVENT_ERROR,
    TERMINAL_EVENT_TYPES,
)


class TestRuntimeEvent:
    def test_serializes_expected_shape(self):
        event = RuntimeEvent(
            event_id="run_abc:1",
            seq=1,
            run_id="run_abc",
            session_id="sess_123",
            type=EVENT_RUN_STARTED,
            created_at=1000.0,
            terminal=False,
            payload={"key": "value"},
        )
        d = event.to_dict(redact=False)
        assert d["event_id"] == "run_abc:1"
        assert d["seq"] == 1
        assert d["run_id"] == "run_abc"
        assert d["session_id"] == "sess_123"
        assert d["type"] == EVENT_RUN_STARTED
        assert d["created_at"] == 1000.0
        assert d["terminal"] is False
        assert d["payload"] == {"key": "value"}

    def test_event_id_stable_format(self):
        event = RuntimeEvent(
            event_id="run_xyz:42",
            seq=42,
            run_id="run_xyz",
            session_id="sess_1",
            type="test",
        )
        assert event.event_id == "run_xyz:42"

    def test_defaults(self):
        event = RuntimeEvent(
            event_id="run_abc:1",
            seq=1,
            run_id="run_abc",
            session_id="sess_123",
            type=EVENT_DONE,
        )
        assert event.created_at > 0
        assert event.terminal is False
        assert event.payload == {}


class TestRuntimeStatus:
    def test_serializes_reconnect_fields(self):
        status = RuntimeStatus(
            run_id="run_abc",
            session_id="sess_123",
            status=RUN_STATUS_QUEUED,
            last_event_id="run_abc:5",
            last_seq=5,
            terminal=False,
            controls=["observe", "stop"],
            pending_approval_ids=["appr_1"],
            pending_clarify_ids=["clar_1"],
            error=None,
            result=None,
            created_at=1000.0,
            updated_at=2000.0,
        )
        d = status.to_dict(redact=False)
        assert d["run_id"] == "run_abc"
        assert d["session_id"] == "sess_123"
        assert d["status"] == "queued"
        assert d["last_event_id"] == "run_abc:5"
        assert d["last_seq"] == 5
        assert d["terminal"] is False
        assert d["controls"] == ["observe", "stop"]
        assert d["pending_approval_ids"] == ["appr_1"]
        assert d["pending_clarify_ids"] == ["clar_1"]
        assert d["error"] is None
        assert d["result"] is None
        assert d["created_at"] == 1000.0
        assert d["updated_at"] == 2000.0

    def test_terminal_statuses_known(self):
        assert "cancelled" in TERMINAL_STATUSES
        assert "failed" in TERMINAL_STATUSES
        assert "completed" in TERMINAL_STATUSES
        assert "expired" in TERMINAL_STATUSES
        assert "running" not in TERMINAL_STATUSES
        assert "queued" not in TERMINAL_STATUSES

    def test_terminal_event_types(self):
        assert EVENT_DONE in TERMINAL_EVENT_TYPES
        assert EVENT_ERROR in TERMINAL_EVENT_TYPES
        assert EVENT_RUN_STARTED not in TERMINAL_EVENT_TYPES


class TestRedaction:
    def test_redacts_known_key_flat(self):
        dirty = {"api_key": "sk-proj-1234567890abcdef", "name": "test"}
        clean = redact_secrets(dirty)
        assert clean["api_key"] == "<<redacted>>"
        assert clean["name"] == "test"

    def test_redacts_recursively(self):
        dirty = {
            "config": {
                "token": "ghp_abcdef1234567890",
                "nested": {"password": "hunter2"},
            },
        }
        clean = redact_secrets(dirty)
        assert clean["config"]["token"] == "<<redacted>>"
        assert clean["config"]["nested"]["password"] == "<<redacted>>"

    def test_redacts_in_lists(self):
        dirty = [
            {"api_key": "sk-12345"},
            {"data": "ok"},
        ]
        clean = redact_secrets(dirty)
        assert clean[0]["api_key"] == "<<redacted>>"
        assert clean[1]["data"] == "ok"

    def test_redacts_known_prefixes_via_agent_redact(self):
        dirty = {"url": "https://sk-proj-abc123@example.com/api"}
        clean = redact_secrets(dirty)
        assert "sk-proj-abc123" not in str(clean)

    def test_redacts_authorization_key(self):
        dirty = {"authorization": "Bearer sk-secret-token-12345"}
        clean = redact_secrets(dirty)
        assert clean["authorization"] == "<<redacted>>"

    def test_redacts_bearer_key(self):
        dirty = {"bearer": "some-long-credential-value"}
        clean = redact_secrets(dirty)
        assert clean["bearer"] == "<<redacted>>"

    def test_does_not_redact_non_secret_keys(self):
        dirty = {"username": "alice", "message": "hello", "token_count": 100}
        clean = redact_secrets(dirty)
        assert clean["username"] == "alice"
        assert clean["message"] == "hello"
        assert isinstance(clean["token_count"], int)

    def test_handles_empty_dict(self):
        assert redact_secrets({}) == {}

    def test_handles_empty_list(self):
        assert redact_secrets([]) == []

    def test_handles_none(self):
        assert redact_secrets(None) is None

    def test_redacts_access_token_key(self):
        dirty = {"access_token": "ya29.secret.oauth.token"}
        clean = redact_secrets(dirty)
        assert clean["access_token"] == "<<redacted>>"

    def test_redacts_refresh_token_key(self):
        dirty = {"refresh_token": "1/secret-refresh-value"}
        clean = redact_secrets(dirty)
        assert clean["refresh_token"] == "<<redacted>>"


class TestModuleImports:
    def test_contract_module_imports_cleanly(self):
        from gateway.runtime import (
            RuntimeEvent,
            RuntimeStatus,
            RunManager,
            redact_secrets,
        )
        assert RuntimeEvent is not None
        assert RuntimeStatus is not None
        assert RunManager is not None
        assert callable(redact_secrets)

    def test_constants_exported(self):
        from gateway.runtime import (
            RUN_STATUS_QUEUED,
            RUN_STATUS_CANCELLED,
            EVENT_RUN_STARTED,
            EVENT_DONE,
        )
        assert RUN_STATUS_QUEUED == "queued"
        assert RUN_STATUS_CANCELLED == "cancelled"
        assert EVENT_RUN_STARTED == "run.started"
        assert EVENT_DONE == "done"
