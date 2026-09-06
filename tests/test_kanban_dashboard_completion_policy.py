import pytest
from fastapi import HTTPException

from plugins.kanban.dashboard import plugin_api


def test_dashboard_review_patch_preserves_completion_policy_reason(monkeypatch):
    monkeypatch.setattr(
        plugin_api.kanban_db,
        "request_review",
        lambda *args, **kwargs: (False, "acknowledge the pending feedback contract first"),
    )
    payload = plugin_api.UpdateTaskBody(status="review", summary="Ready for review")

    with pytest.raises(HTTPException) as caught:
        plugin_api._patch_status(None, "task-1", payload, review_assignee_deferred=False)

    assert caught.value.status_code == 409
    assert caught.value.detail == "acknowledge the pending feedback contract first"