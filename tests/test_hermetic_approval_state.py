"""Regression coverage for approval state loaded during test collection.

Some test modules import ``tools.approval`` at collection time, before the
per-test ``HERMES_HOME`` fixture points Hermes at a temporary profile. The
module loads the permanent command allowlist on import, so a developer's real
``~/.hermes/config.yaml`` could leak into the first test and bypass approval
callbacks. The autouse hermetic fixture must scrub that pre-fixture state.
"""

from tools import approval as _approval_imported_during_collection


# Simulate collection-time approval state that predates the autouse fixture.
# Without the conftest scrub, these values survive into the test body and make
# dangerous commands auto-approve without consulting the callback.
_approval_imported_during_collection._permanent_approved.add("delete in root path")
_approval_imported_during_collection._YOLO_MODE_FROZEN = True


def test_hermetic_environment_scrubs_collection_time_approval_state(monkeypatch):
    from tools import approval

    assert "delete in root path" not in approval._permanent_approved
    assert approval._YOLO_MODE_FROZEN is False

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

    called_with = []

    def fake_cb(command, description, *, allow_permanent=True):
        called_with.append((command, description, allow_permanent))
        return "once"

    result = approval.check_all_command_guards(
        "chmod 777 /tmp/test-hermetic-approval-state",
        "local",
        approval_callback=fake_cb,
    )

    assert result["approved"] is True
    assert called_with
