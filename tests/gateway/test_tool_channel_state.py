"""Concurrency contracts for the split-runtime tool broker."""

import threading
import time

from gateway.tool_channel_state import (
    cancel_tool_request,
    clear_tool_channel_state,
    close_tool_channel,
    register_tool_notify,
    resolve_tool_result,
    submit_tool_request,
    tool_result_authorized,
    unregister_tool_notify,
    wait_for_attached_client,
)


def teardown_function():
    clear_tool_channel_state()


def test_transport_request_ids_are_unique_when_model_call_ids_repeat():
    session_key = "run-repeated-call-id"
    assert register_tool_notify(session_key, lambda request: None, "") is True

    first = submit_tool_request(session_key, {"tool_call_id": "call_same"})
    assert first is not None
    first_request_id = first.request["request_id"]
    assert resolve_tool_result(session_key, first_request_id, "first") == "resolved"

    second = submit_tool_request(session_key, {"tool_call_id": "call_same"})
    assert second is not None
    second_request_id = second.request["request_id"]

    assert second_request_id != first_request_id
    assert resolve_tool_result(session_key, second_request_id, "second") == "resolved"
    assert first.result == "first"
    assert second.result == "second"


def test_late_result_cannot_resolve_retried_model_call_id():
    session_key = "run-late-result"
    assert register_tool_notify(session_key, lambda request: None, "") is True

    expired = submit_tool_request(session_key, {"tool_call_id": "call_retry"})
    assert expired is not None
    expired_request_id = expired.request["request_id"]
    assert cancel_tool_request(session_key, expired_request_id) is True

    retry = submit_tool_request(session_key, {"tool_call_id": "call_retry"})
    assert retry is not None
    retry_request_id = retry.request["request_id"]

    assert resolve_tool_result(session_key, expired_request_id, "late") == "unknown"
    assert retry.result is None
    assert resolve_tool_result(session_key, retry_request_id, "fresh") == "resolved"
    assert retry.result == "fresh"


def test_wait_for_attached_client_observes_late_registration():
    session_key = "run-attach-wait"

    def attach_later():
        time.sleep(0.02)
        register_tool_notify(session_key, lambda request: None, "")

    thread = threading.Thread(target=attach_later)
    thread.start()
    try:
        assert wait_for_attached_client(session_key, timeout=1.0) is True
    finally:
        thread.join(timeout=1.0)
        unregister_tool_notify(session_key)


def test_wait_for_attached_client_times_out_without_executor():
    started = time.monotonic()
    assert wait_for_attached_client("run-never-attached", timeout=0.02) is False
    assert time.monotonic() - started < 0.5


def test_terminal_close_wakes_pre_attachment_waiter():
    session_key = "run-close-before-attach"
    observed = {}

    def wait_for_executor():
        observed["attached"] = wait_for_attached_client(session_key, timeout=5.0)

    thread = threading.Thread(target=wait_for_executor)
    thread.start()
    time.sleep(0.02)
    close_tool_channel(session_key)
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert observed == {"attached": False}
    assert register_tool_notify(session_key, lambda request: None, "") is False


def test_completed_request_id_remains_idempotent_across_executor_reconnect():
    session_key = "run-reconnect-idempotency"
    assert register_tool_notify(session_key, lambda request: None, "token-a") is True
    entry = submit_tool_request(session_key, {"tool_call_id": "call_once"})
    assert entry is not None
    request_id = entry.request["request_id"]
    assert resolve_tool_result(session_key, request_id, "done") == "resolved"

    assert unregister_tool_notify(session_key, client_token="token-a") is True
    assert register_tool_notify(session_key, lambda request: None, "token-b") is True
    assert resolve_tool_result(session_key, request_id, "retry") == "duplicate"


def test_completed_request_id_remains_idempotent_after_terminal_close():
    session_key = "run-terminal-idempotency"
    assert register_tool_notify(session_key, lambda request: None, "token-a") is True
    entry = submit_tool_request(session_key, {"tool_call_id": "call_once"})
    assert entry is not None
    request_id = entry.request["request_id"]
    assert resolve_tool_result(session_key, request_id, "done") == "resolved"

    close_tool_channel(session_key)

    assert resolve_tool_result(session_key, request_id, "retry") == "duplicate"


def test_unicode_executor_tokens_compare_without_type_error():
    session_key = "run-unicode-token"
    token = "executor-🔐"
    assert register_tool_notify(session_key, lambda request: None, token) is True
    assert tool_result_authorized(session_key, token) is True
    assert tool_result_authorized(session_key, "executor-other") is False
    assert unregister_tool_notify(session_key, client_token=token) is True
