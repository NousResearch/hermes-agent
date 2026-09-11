"""Local commands use restart-safe native consent without a second notification."""

from dataclasses import replace

import pytest

from gateway import wisdom_command as command
from hermes_wisdom.client import WisdomNotFound
from hermes_wisdom.consent import WisdomConsent
from tests.wisdom.test_native_install_policy import native_install as native_install


def context():
    return command.WisdomCommandContext(
        user_id="local-user", chat_id="local:command-session", profile="test",
        organization_id="org-1",
    )


def review(service, *, operation):
    controller = command.WisdomCommandController()
    ctx = context()
    if operation == "update":
        service.install_apply(service.install_plan("skill-1@v1", update_mode="MANUAL")["receipt"])
        service.client.records.clear()
        service.client.latest = 2
        return controller.execute("update skill-1", service, ctx)
    modes = command.bind_view_callbacks(controller.execute("install skill-1@v1", service, ctx), ctx)
    choice = next(a for a in modes.actions if a.arguments.get("update_mode") == "MANUAL")
    return controller.execute_token(choice.callback_data.removeprefix("wi:cmd:"), service, ctx)


@pytest.mark.parametrize("operation", ["install", "update"])
def test_local_review_survives_callback_memory_loss_without_queued_duplicate(native_install, monkeypatch, operation):
    from tests.wisdom.local_auth import authorize_local
    authorize_local(monkeypatch)
    service, _, _ = native_install
    shown = review(service, operation=operation)
    approve = next(a.callback_data for a in shown.actions if (a.callback_data or "").startswith("wi:agent:confirm:"))
    identity = approve.rsplit(":", 1)[1]
    text = command.render_local_view(shown, context())
    assert f"/wisdom consent {identity} confirm" in text
    assert "wip_" not in text and "wup_" not in text
    assert service.client.records == []
    consent = WisdomConsent(service)
    with service.store.transaction() as db:
        row = db.execute("SELECT state FROM wisdom_assessment WHERE id=(SELECT assessment_id FROM wisdom_consent WHERE id=?)", (identity,)).fetchone()
        assert row[0] == "passive"
        assert db.execute("SELECT COUNT(*) FROM wisdom_delivery_receipt").fetchone()[0] == 0
    consent.queue.register_session(
        "org-1", session_key="command-session", session_id="command-session", platform="local",
        actor_id="local-user", private=True, available=True, user_activity=True,
        address={"chat_id": context().chat_id, "thread_id": "", "scope_id": ""},
    )
    assert consent.queue.claim("org-1", "command-session") == []
    monkeypatch.setattr(command, "CALLBACK_TOKENS", command._CallbackTokens())
    controller = command.WisdomCommandController()
    with pytest.raises(WisdomNotFound):
        controller.execute(f"consent {identity} confirm", service, replace(context(), chat_id="local:another-session"))
    checked = controller.execute(f"consent {identity} checks.show", service, context())
    assert any(a.callback_data == approve for a in checked.actions)
    assert service.client.records == []
    completed = controller.execute(f"consent {identity} confirm", service, context())
    assert completed.summary == ("Files installed" if operation == "install" else "Files updated")
    controller.execute(f"consent {identity} confirm", service, context())
    assert len(service.client.records) == 1
    assert service.store.installation("skill-1")["version"] == (1 if operation == "install" else 2)


@pytest.mark.parametrize("status", [None, "blocked", "unavailable"])
@pytest.mark.parametrize("changed_after_review", [False, True])
def test_local_native_review_requires_current_security_clearance(native_install, status, changed_after_review, monkeypatch):
    from tests.wisdom.local_auth import authorize_local
    authorize_local(monkeypatch)
    service, _, _ = native_install
    if not changed_after_review:
        service.client.security_status = status
    shown = review(service, operation="install")
    approvals = [a for a in shown.actions if (a.callback_data or "").startswith("wi:agent:confirm:")]
    if changed_after_review:
        assert len(approvals) == 1
        service.client.security_status = status
        identity = approvals[0].callback_data.rsplit(":", 1)[1]
        result = command.WisdomCommandController().execute(f"consent {identity} confirm", service, context())
        assert result.summary == "Stale"
    else:
        assert not approvals
    assert service.client.records == []
    assert service.store.installation("skill-1") is None
