"""Tests for kanban worker non-retryable model-client error hardening.

When a dispatcher-spawned kanban worker hits a model-client error that will
not be fixed by retrying (403 auth termination, model_not_found, policy blocks,
format errors, etc.), the worker should block the card cleanly instead of
burning the card's failure budget and exiting with a generic error code.
"""

from types import SimpleNamespace

import pytest

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


class _FakeKanbanDB:
    """In-memory kanban DB stub recording block_task calls."""

    def __init__(self):
        self.connections = []
        self.blocks = []

    def connect(self):
        conn = SimpleNamespace(closed=False)
        conn.close = lambda: setattr(conn, "closed", True)
        self.connections.append(conn)
        return conn

    def block_task(self, conn, task_id, *, reason=None):
        self.blocks.append({"task_id": task_id, "reason": reason})


@pytest.fixture
def fake_kanban_db(monkeypatch):
    """Replace ``hermes_cli.kanban_db`` with a recording stub for cli tests."""
    import sys

    fake = _FakeKanbanDB()
    fake_mod = SimpleNamespace(
        connect=fake.connect,
        block_task=fake.block_task,
        KANBAN_RATE_LIMIT_EXIT_CODE=75,
    )
    # The cli functions do ``from hermes_cli import kanban_db as _kb`` at call
    # time. Python first checks the parent package attribute, so patch both the
    # package attribute and ``sys.modules`` to be certain the fake is used.
    import hermes_cli

    monkeypatch.setattr(hermes_cli, "kanban_db", fake_mod)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db", fake_mod)
    return fake


def _make_result(failure_reason, *, failed=True, status_code=403, error="nope", provider=None):
    return {
        "final_response": "",
        "error": error,
        "failed": failed,
        "failure_reason": failure_reason,
        "status_code": status_code,
        "error_type": "SomeError",
        "provider": provider,
    }


@pytest.mark.parametrize(
    "reason",
    [
        "auth",
        "auth_permanent",
        "model_not_found",
        "provider_policy_blocked",
        "content_policy_blocked",
        "format_error",
    ],
)
def test_is_nonretryable_model_client_error_true_for_nonretryable_reasons(reason):
    assert cli._is_nonretryable_model_client_error(_make_result(reason)) is True


@pytest.mark.parametrize("reason", ["rate_limit", "billing", "transient", "unknown"])
def test_is_nonretryable_model_client_error_false_for_retryable_or_unknown_reasons(reason):
    assert cli._is_nonretryable_model_client_error(_make_result(reason)) is False


def test_is_nonretryable_model_client_error_false_when_not_failed():
    assert cli._is_nonretryable_model_client_error(_make_result("auth", failed=False)) is False


def test_is_nonretryable_model_client_error_false_for_non_dict_result():
    assert cli._is_nonretryable_model_client_error("not a dict") is False
    assert cli._is_nonretryable_model_client_error(None) is False


def test_content_policy_blocked_result_includes_classified_fields():
    from agent.conversation_loop import _content_policy_blocked_result

    result = _content_policy_blocked_result(
        [],
        0,
        final_response="blocked",
        error_detail="bad content",
        failure_reason="content_policy_blocked",
        status_code=400,
        error_type="BadRequestError",
    )
    assert result["failed"] is True
    assert result["failure_reason"] == "content_policy_blocked"
    assert result["status_code"] == 400
    assert result["error_type"] == "BadRequestError"


def test_kanban_model_error_block_reason_uses_cli_provider_and_truncates_summary():
    fake_cli = SimpleNamespace(agent=SimpleNamespace(provider="test-provider"))
    result = _make_result("auth", status_code=401, error="short error")
    reason = cli._kanban_model_error_block_reason(result, fake_cli)
    assert reason == "model-error: test-provider 401 auth — short error"


def test_kanban_model_error_block_reason_falls_back_to_result_provider():
    fake_cli = SimpleNamespace(agent=SimpleNamespace(provider=None))
    result = _make_result("model_not_found", status_code=404, error="missing", provider="fallback")
    reason = cli._kanban_model_error_block_reason(result, fake_cli)
    assert reason == "model-error: fallback 404 model_not_found — missing"


def test_kanban_model_error_block_reason_falls_back_to_unknown_and_default_summary():
    fake_cli = SimpleNamespace(agent=SimpleNamespace(provider=None))
    result = {"failed": True, "failure_reason": "format_error"}
    reason = cli._kanban_model_error_block_reason(result, fake_cli)
    assert reason == "model-error: unknown ? format_error — model provider returned a non-retryable client error"


def test_kanban_model_error_block_reason_truncates_long_summary():
    fake_cli = SimpleNamespace(agent=SimpleNamespace(provider="p"))
    result = _make_result("auth", status_code=403, error="x" * 300)
    reason = cli._kanban_model_error_block_reason(result, fake_cli)
    assert reason.endswith("...")
    assert len(reason) < 250


def test_block_kanban_task_for_model_error_blocks_when_env_task_set(
    monkeypatch, fake_kanban_db
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_12345")
    fake_cli = SimpleNamespace(agent=SimpleNamespace(provider="p"))
    result = _make_result("auth", status_code=403, error="access denied")

    cli._block_kanban_task_for_model_error(fake_cli, result)

    assert len(fake_kanban_db.blocks) == 1
    assert fake_kanban_db.blocks[0]["task_id"] == "t_12345"
    assert fake_kanban_db.blocks[0]["reason"].startswith("model-error: p 403 auth")
    assert all(conn.closed for conn in fake_kanban_db.connections)


def test_block_kanban_task_for_model_error_no_op_without_env_task(
    monkeypatch, fake_kanban_db
):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    fake_cli = SimpleNamespace(agent=SimpleNamespace(provider="p"))

    cli._block_kanban_task_for_model_error(fake_cli, _make_result("auth"))

    assert fake_kanban_db.blocks == []


def _make_fake_cli_for_main(result):
    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="quiet-session",
                platform="cli",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=lambda **kw: result,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, effective_query):
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **kwargs):
            return True

    return FakeCLI


def test_main_with_kanban_task_nonretryable_model_error_blocks_and_exits_cleanly(
    monkeypatch, fake_kanban_db
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abcde")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    result = _make_result("auth", status_code=403, error="access terminated")
    monkeypatch.setattr(cli, "HermesCLI", _make_fake_cli_for_main(result))
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 0
    assert len(fake_kanban_db.blocks) == 1
    assert fake_kanban_db.blocks[0]["task_id"] == "t_abcde"


def test_main_without_kanban_task_nonretryable_model_error_exits_with_error_code(
    monkeypatch, fake_kanban_db
):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    result = _make_result("auth", status_code=403, error="access terminated")
    monkeypatch.setattr(cli, "HermesCLI", _make_fake_cli_for_main(result))
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
    assert fake_kanban_db.blocks == []


def test_main_with_kanban_task_rate_limit_still_uses_rate_limit_exit_code(
    monkeypatch, fake_kanban_db
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abcde")
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    result = _make_result("rate_limit", status_code=429, error="rate limited")
    monkeypatch.setattr(cli, "HermesCLI", _make_fake_cli_for_main(result))
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert fake_kanban_db.blocks == []
