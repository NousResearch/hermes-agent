"""Kanban workers must report preflight failures to their supervisor."""

from types import SimpleNamespace

import cli


def test_kanban_chat_records_credential_preflight_failure(monkeypatch):
    """A missing provider key must not look like a clean protocol exit.

    The worker supervisor uses ``_last_turn_result`` to distinguish a failed
    one-shot turn from a process that exited without a terminal Kanban call.
    Early credential failure used to return ``None`` without setting that
    result, so the supervisor classified the process as a protocol violation
    and retried it as if the task itself were broken.
    """
    shell = SimpleNamespace(
        _last_turn_result=None,
        _last_turn_interrupted=False,
        _secret_capture_callback=None,
        _ensure_runtime_credentials=lambda: False,
        _ensure_conversation_worktree_binding=lambda: None,
    )
    monkeypatch.setattr(cli, "set_secret_capture_callback", lambda *_args: None)

    result = cli.HermesCLI.chat(shell, "work kanban task t_provider")

    assert result is None
    assert shell._last_turn_result == {
        "failed": True,
        "failure_reason": "credentials",
        "error": "runtime credentials unavailable",
    }
