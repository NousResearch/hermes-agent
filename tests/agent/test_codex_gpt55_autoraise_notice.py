"""Tests for one-time gating of the Codex gpt-5.5 autoraise notice.

The notice was previously re-emitted on every agent init (i.e. every Discord
turn-1 / new conversation / CLI invocation), even though its docstring called
it a "one-time notice." This module covers the per-install ack file that
silences subsequent emissions.
"""

import os
from unittest.mock import patch

import pytest

from agent.agent_init import (
    _build_codex_gpt55_autoraise_notice,
    _codex_gpt55_autoraise_ack_path,
    _codex_gpt55_autoraise_acked,
    _record_codex_gpt55_autoraise_acked,
)


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point ``get_hermes_home()`` at a clean tmp dir for the test.

    The module imports ``get_hermes_home`` from ``hermes_constants`` at
    load time — patch the symbol where it's actually looked up (inside
    ``agent.agent_init``) so our helpers see the redirected path.
    """
    monkeypatch.setattr(
        "agent.agent_init.get_hermes_home",
        lambda: str(tmp_path),
    )
    return tmp_path


def test_ack_path_is_under_hermes_home(isolated_hermes_home):
    assert _codex_gpt55_autoraise_ack_path() == os.path.join(
        str(isolated_hermes_home), ".codex_gpt55_autoraise_acked"
    )


def test_acked_false_when_marker_absent(isolated_hermes_home):
    assert _codex_gpt55_autoraise_acked() is False


def test_acked_true_after_record(isolated_hermes_home):
    _record_codex_gpt55_autoraise_acked()
    assert _codex_gpt55_autoraise_acked() is True
    # And persists across re-checks (file-backed, not just in-memory).
    assert os.path.exists(_codex_gpt55_autoraise_ack_path())


def test_record_is_idempotent(isolated_hermes_home):
    _record_codex_gpt55_autoraise_acked()
    _record_codex_gpt55_autoraise_acked()  # second call must not raise
    assert _codex_gpt55_autoraise_acked() is True


def test_record_silent_on_oserror(isolated_hermes_home):
    """Filesystem errors must not crash agent init — falling back to
    "notice re-emits next session" is no worse than the pre-patch baseline.
    """
    with patch("builtins.open", side_effect=OSError("disk full")):
        _record_codex_gpt55_autoraise_acked()  # must not raise

    assert _codex_gpt55_autoraise_acked() is False


def test_acked_silent_on_oserror(isolated_hermes_home, monkeypatch):
    """A racy unreadable home dir must degrade to "not acked" cleanly."""
    monkeypatch.setattr(os.path, "exists", lambda _: (_ for _ in ()).throw(OSError("eperm")))
    assert _codex_gpt55_autoraise_acked() is False


def test_notice_text_includes_opt_out_command():
    """The notice MUST include the exact command the user needs to opt out.

    A user staring at the notice should never have to grep for it — this
    is the assertion that protects against accidental message rewording
    in a refactor.
    """
    text = _build_codex_gpt55_autoraise_notice({"from": 0.50, "to": 0.85})
    assert "hermes config set compression.codex_gpt55_autoraise false" in text
    assert "85%" in text
    assert "50%" in text
