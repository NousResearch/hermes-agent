"""Named Output must not reopen another profile owner's state database."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import session_hosted_output


@pytest.mark.parametrize("ownership", ["served-named", "foreign-owner-transport"])
def test_named_output_refuses_foreign_database_binding(tmp_path, monkeypatch, ownership):
    service = SimpleNamespace(check_admission=Mock(side_effect=AssertionError("must not inspect foreign task")))
    ref = SimpleNamespace(session_id="session", profile_id="")
    row = {
        "request_id": 'hosted:[{"room_id":"room"},1]',
        "target_session_id": ref.session_id,
        "principal_id": "hosted-owner",
    }

    if ownership == "served-named":
        home = tmp_path / "profiles" / "reviewer"
        # This negative control has no owner-transport consent. Actual named
        # admission and consent are exercised by test_named_output_owner_rpc.
        monkeypatch.setattr("gateway.session_managed_worker.managed_policy", lambda *_: None)
        monkeypatch.setattr(session_hosted_output, "_is_owner_transport_admission", lambda *_: False)
    else:
        home = tmp_path / "reviewer-owner"
        monkeypatch.setattr("gateway.session_managed_worker.managed_policy", lambda *_: None)
        monkeypatch.setattr(session_hosted_output, "_is_owner_transport_admission", lambda *_: True)

    ref.profile_id = str(home)
    authority = SimpleNamespace(
        profile_id=str(home),
        db=SimpleNamespace(db_path=home / "state.db"),
        hosted_room_service=service,
        runner=object(),
        sessions={ref.session_id: SimpleNamespace(source=object())},
    )
    monkeypatch.setattr(
        "gateway.session_policy.policy_for_source",
        lambda *_: SimpleNamespace(source="bot_room", toolsets=("bot_room",)),
    )

    assert session_hosted_output._binding(authority, ref, row) is None
    service.check_admission.assert_not_called()
    assert not authority.db.db_path.exists()
