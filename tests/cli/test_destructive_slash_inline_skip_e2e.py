"""End-to-end integration test for the destructive-slash inline-skip path.

Drives ``HermesCLI.process_command("/reset now")`` against a minimal stand-in
and verifies:

1. ``new_session`` was invoked (the command actually ran)
2. ``_prompt_text_input_modal`` was NOT invoked (modal bypassed)
3. The skip token did not leak into the session title

This is the regression test for issue #30768 — the inline-skip escape hatch
must work without ever touching the modal, on every platform.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _make_cli_stub():
    """Build a minimal HermesCLI-shaped object that can run ``process_command``
    for the destructive-slash branches without spinning up a real TUI."""
    from cli import HermesCLI

    new_session_calls = []

    def _capture_new_session(self_, title=None, silent=False):
        new_session_calls.append({"title": title, "silent": silent})

    self_ = SimpleNamespace(
        _app=None,
        _prompt_text_input_modal=lambda **_kw: (_ for _ in ()).throw(
            AssertionError("modal must not be invoked when inline-skip token present")
        ),
        new_session=lambda **kw: _capture_new_session(self_, **kw),
        # Stub out side-effects the destructive-slash branches reach for.
        console=SimpleNamespace(clear=lambda: None),
        compact=False,
        model="stub-model",
        session_id="stub-session",
        enabled_toolsets=[],
        _pending_title=None,
        _session_db=None,
    )
    # Bind the methods we need under test.
    self_._split_destructive_skip = HermesCLI._split_destructive_skip
    self_._confirm_destructive_slash = HermesCLI._confirm_destructive_slash.__get__(
        self_, type(self_)
    )
    self_.process_command = HermesCLI.process_command.__get__(self_, type(self_))
    return self_, new_session_calls


def test_reset_now_invokes_new_session_without_modal():
    """``/reset now`` runs ``new_session`` and never touches the modal."""
    self_, calls = _make_cli_stub()

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        self_.process_command("/reset now")

    assert calls, "new_session was never invoked"
    # The /new branch passes title=None when there's no non-skip remainder.
    assert calls[0]["title"] is None


def test_new_yes_with_title_preserves_title():
    """``/new --yes My Session`` runs ``new_session(title='My Session')``."""
    self_, calls = _make_cli_stub()

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        self_.process_command("/new --yes My Session")

    assert calls, "new_session was never invoked"
    assert calls[0]["title"] == "My Session"


def test_new_without_skip_token_still_consults_modal():
    """``/new My Session`` (no skip token) must reach the modal.

    Sanity check that we haven't accidentally short-circuited the normal path.
    """
    from cli import HermesCLI

    new_session_calls = []
    modal_calls = []

    def _capture_new_session(self_, title=None, silent=False):
        new_session_calls.append({"title": title, "silent": silent})

    def _record_modal(**kw):
        modal_calls.append(kw)
        # Simulate user cancelling so new_session is not called.
        return "3"

    self_ = SimpleNamespace(
        _app=None,
        _prompt_text_input_modal=_record_modal,
        new_session=lambda **kw: _capture_new_session(self_, **kw),
        console=SimpleNamespace(clear=lambda: None),
        compact=False,
        model="stub-model",
        session_id="stub-session",
        enabled_toolsets=[],
        _pending_title=None,
        _session_db=None,
    )
    self_._split_destructive_skip = HermesCLI._split_destructive_skip
    self_._normalize_slash_confirm_choice = HermesCLI._normalize_slash_confirm_choice.__get__(
        self_, type(self_)
    )
    self_._confirm_destructive_slash = HermesCLI._confirm_destructive_slash.__get__(
        self_, type(self_)
    )
    self_.process_command = HermesCLI.process_command.__get__(self_, type(self_))

    with patch(
        "cli.load_cli_config",
        return_value={"approvals": {"destructive_slash_confirm": True}},
    ):
        self_.process_command("/new My Session")

    assert modal_calls, "modal must be reached when no skip token is present"
    assert not new_session_calls, "user cancelled — new_session must not run"
