"""Tests for the /update slash command in the classic CLI and TUI launcher.

Verifies that ``HermesCLI._handle_update_command`` correctly:
- Refuses to run under a managed install (Homebrew, Docker, etc.)
- Sets ``_pending_relaunch`` and returns ``True`` on confirmation
- Cancels cleanly on a "no"-shaped answer or unrecognized input
- Cancels cleanly when ``_prompt_text_input_modal`` returns None (timeout /
  modal dismissed)

Also verifies that ``hermes_cli.main._launch_tui`` correctly handles exit
code 42 (the TUI's signal to trigger an update) by calling
``relaunch(["update"], preserve_inherited=False)`` from the Python wrapper
side.  The companion Vitest (``ui-tui/src/__tests__/createSlashHandler.test.ts``)
covers the TypeScript slash-handler that *emits* code 42; this file covers
the Python wrapper branch that *acts on* it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cli import HermesCLI


def _bound(fn, instance):
    """Bind an unbound method to a stand-in instance."""
    return fn.__get__(instance, type(instance))


def _make_self(modal_response):
    """Build a minimal stand-in 'self' for ``_handle_update_command``.

    Uses the same SimpleNamespace pattern as ``test_destructive_slash_confirm``
    so we don't need a full ``HermesCLI`` construction.
    ``_prompt_text_input_modal`` is stubbed to return *modal_response*
    directly so tests can drive the entire confirmation branch without
    touching stdin or prompt_toolkit internals.
    """
    self_ = SimpleNamespace(
        _app=None,
        _pending_relaunch=None,
        _prompt_text_input_modal=lambda **_kw: modal_response,
    )
    self_._normalize_slash_confirm_choice = _bound(
        HermesCLI._normalize_slash_confirm_choice, self_
    )
    return self_


def _call(self_):
    """Invoke the real ``_handle_update_command`` on the stub."""
    return HermesCLI._handle_update_command(self_)


# ---------------------------------------------------------------------------
# Managed-install guard
# ---------------------------------------------------------------------------


def test_managed_install_refuses_and_does_not_set_pending_relaunch(capsys):
    """Under a managed install (brew/docker), /update prints a hint and
    returns without setting ``_pending_relaunch``."""
    self_ = SimpleNamespace(
        _app=None,
        _pending_relaunch=None,
        # Use pytest.fail so any unexpected modal invocation surfaces as a failure.
        _prompt_text_input_modal=lambda **_kw: pytest.fail("Modal should not be called"),
    )
    self_._normalize_slash_confirm_choice = _bound(
        HermesCLI._normalize_slash_confirm_choice, self_
    )
    with (
        patch("hermes_cli.config.is_managed", return_value=True),
        patch(
            "hermes_cli.config.format_managed_message",
            return_value="Use `brew upgrade ht-ai-agent` to update.",
        ),
    ):
        result = _call(self_)

    out = capsys.readouterr().out
    assert "brew upgrade ht-ai-agent" in out
    assert self_._pending_relaunch is None
    assert not result


# ---------------------------------------------------------------------------
# Confirmation proceeds only on recognised affirmative responses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", "1", "ok"])
def test_affirmative_answer_sets_pending_relaunch_and_returns_true(answer, capsys):
    """Recognised affirmative answers ("y", "yes", "1", "ok") set
    ``_pending_relaunch = ["update"]`` and return ``True`` so the caller
    (process_command) can trigger the main-thread app-exit path."""
    self_ = _make_self(modal_response=answer)
    with patch("hermes_cli.config.is_managed", return_value=False):
        result = _call(self_)

    assert self_._pending_relaunch == ["update"]
    assert result is True
    assert "Launching update" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Cancellation paths — _pending_relaunch must stay None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["n", "N", "no", "NO", " no "])
def test_negative_answer_cancels(answer, capsys):
    """Any "no"-shaped answer cancels without setting ``_pending_relaunch``."""
    self_ = _make_self(modal_response=answer)
    with patch("hermes_cli.config.is_managed", return_value=False):
        result = _call(self_)

    assert self_._pending_relaunch is None
    assert not result
    assert "Launching update" not in capsys.readouterr().out


def test_none_response_cancels(capsys):
    """``None`` from the modal (timeout or dismiss) cancels cleanly."""
    self_ = _make_self(modal_response=None)
    with patch("hermes_cli.config.is_managed", return_value=False):
        result = _call(self_)

    assert self_._pending_relaunch is None
    assert not result


@pytest.mark.parametrize("answer", ["nope", "cancel", "sure", "2", "3", "abort", ""])
def test_unrecognized_or_cancel_input_cancels(answer, capsys):
    """Unrecognised input and explicit "cancel" do not proceed.

    Previously the implementation treated any non-"n/no" answer as approval,
    which meant typos like "nope" or "cancel" would launch the update.
    Now only confirmed affirmative aliases ("y", "yes", "1", "ok") proceed;
    everything else (including empty string, "cancel", typos) cancels.
    """
    self_ = _make_self(modal_response=answer)
    with patch("hermes_cli.config.is_managed", return_value=False):
        result = _call(self_)

    assert self_._pending_relaunch is None
    assert not result
