"""Fail-closed tests for generic Codex approval dispatch."""

from unittest.mock import MagicMock

import pytest

from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
    _ServerRequestRouting,
    _approval_choice_to_codex_decision,
)


@pytest.mark.parametrize(
    "choice",
    [None, "", "unknown", [], ["once"], {}, {"approved": 1}],
)
def test_malformed_approval_decisions_map_to_decline(choice):
    assert _approval_choice_to_codex_decision(choice) == "decline"


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("once", "accept"), ("session", "acceptForSession"),
     ("always", "acceptForSession"), ("deny", "decline"),
     ("timeout", "decline"), ("cancelled", "decline")],
)
def test_valid_approval_decisions_keep_their_wire_behavior(choice, expected):
    assert _approval_choice_to_codex_decision(choice) == expected


@pytest.mark.parametrize(
    "callback",
    [
        lambda *_args, **_kwargs: None,
        lambda *_args, **_kwargs: {"approved": 1},
        lambda *_args, **_kwargs: ["once"],
        lambda *_args, **_kwargs: "malformed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    ],
)
def test_generic_approval_dispatch_never_accepts_malformed_or_raises(callback):
    session = CodexAppServerSession(
        cwd="/tmp", client_factory=lambda **_kwargs: MagicMock(), approval_callback=callback,
    )
    assert session._decide_exec_approval({"command": "secret-command"}) == "decline"


def test_generic_dispatch_missing_seam_fails_closed_without_prompt():
    prompt = MagicMock(side_effect=AssertionError("approval prompt must not run"))
    session = CodexAppServerSession(cwd="/tmp", client_factory=lambda **_kwargs: MagicMock())
    assert session._run_approval_callback(False, prompt, "missing seam") == "decline"
    prompt.assert_not_called()


def test_generic_dispatch_timeout_fails_closed():
    session = CodexAppServerSession(
        cwd="/tmp", client_factory=lambda **_kwargs: MagicMock(),
        approval_callback=lambda *_args, **_kwargs: "timeout",
    )
    assert session._decide_exec_approval({"command": "secret-command"}) == "decline"


def test_auto_approval_preserves_valid_bypass_without_prompt():
    prompt = MagicMock(side_effect=AssertionError("bypass must not prompt"))
    session = CodexAppServerSession(cwd="/tmp", client_factory=lambda **_kwargs: MagicMock())
    assert session._run_approval_callback(True, prompt, "auto") == "accept"
    prompt.assert_not_called()
