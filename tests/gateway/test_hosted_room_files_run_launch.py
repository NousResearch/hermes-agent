"""Files persistence through main's split run launcher, without a model or socket."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway.hosted_room_execution_policy import (
    current_room_execution_policy,
    execution_policy_mapping,
)
from gateway.platforms.api_server_runs import _RunLaunch, _run_agent_sync
from tools.approval_context import get_current_session_key


@pytest.mark.parametrize("persisted", [None, "Review the brief.\n\n[Group Chat files: brief.txt]"])
def test_run_launch_keeps_file_transcript_override_and_current_ownership(persisted):
    calls = []
    policy = execution_policy_mapping(target_profile="reviewer", config={})
    before_policy = current_room_execution_policy()
    before_session = get_current_session_key()

    @contextmanager
    def profile_scope(profile):
        calls.append(("profile-enter", profile))
        yield
        calls.append(("profile-exit", profile))

    def converse(**kwargs):
        calls.append(("conversation", kwargs))
        assert get_current_session_key() == "run-files"
        assert current_room_execution_policy().as_mapping() == policy
        return {"final_response": "Reviewed"}

    owner = SimpleNamespace(
        _profile_scope=profile_scope,
        _bind_api_server_session=lambda **scope: calls.append(("session", scope)),
    )
    api = SimpleNamespace(
        _publish_turn_process_ownership=lambda agent, task: calls.append(("owner", task)),
        _clear_turn_process_ownership=lambda agent: calls.append(("owner-cleared",)),
    )
    run = _RunLaunch(
        owner=owner,
        run_id="run-files",
        queue=asyncio.Queue(),
        session_id="room-files",
        gateway_session_key="gateway-room-files",
        declared_selected=False,
        user_message="Review the brief at /private/staging/brief.txt",
        conversation_history=[{"role": "assistant", "content": "Ready"}],
        agent_kwargs={"room_dispatch": {"task_id": "task-files"}, "room_execution_policy": policy},
        request_profile="reviewer",
        browser_control_principal=None,
        browser_control_transport_family=None,
        room_persist_user_message=persisted,
    )
    result, _usage = _run_agent_sync(
        owner, run, SimpleNamespace(run_conversation=converse), lambda event: None, _api_server=api,
    )

    assert result == {"final_response": "Reviewed"}
    kwargs = next(value for kind, value in calls if kind == "conversation")
    assert kwargs == {
        "user_message": run.user_message,
        "conversation_history": run.conversation_history,
        "task_id": run.session_id,
        **({"persist_user_message": persisted} if persisted is not None else {}),
    }
    assert calls.index(("owner", run.session_id)) < calls.index(("conversation", kwargs))
    assert calls[-2:] == [("owner-cleared",), ("profile-exit", "reviewer")]
    assert get_current_session_key() == before_session
    assert current_room_execution_policy() is before_policy
