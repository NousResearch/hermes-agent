"""Tests for the adoption offer (hop 2).

Phase 2 task 2.4: the offer text is printed once per N days (snooze stamp),
with `never` it's silent, with `auto` + pristine + non-interactive it
invokes adopt. The offer NEVER raises.
"""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.adoption_offer import (
    should_offer,
    offer_adoption,
    _is_snoozed,
    _mark_shown,
    ADOPT_PROMPT_COPY,
    SNOOZE_SECONDS,
)


@pytest.fixture
def hermes_home(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    return root


class TestSnooze:
    def test_not_snoozed_initially(self, hermes_home):
        assert not _is_snoozed(hermes_home)

    def test_snoozed_after_mark(self, hermes_home):
        _mark_shown(hermes_home)
        assert _is_snoozed(hermes_home)

    def test_snooze_expires(self, hermes_home):
        # Write an old timestamp
        snooze_path = hermes_home / "state" / "adoption-snooze"
        snooze_path.parent.mkdir(parents=True, exist_ok=True)
        snooze_path.write_text(str(time.time() - SNOOZE_SECONDS - 1))
        assert not _is_snoozed(hermes_home)


class TestShouldOffer:
    def test_never_mode_returns_false(self, hermes_home, project_root):
        assert not should_offer(hermes_home, project_root, adopt_mode="never")

    def test_snoozed_returns_false(self, hermes_home, project_root):
        _mark_shown(hermes_home)
        # Even with a legacy install detected, snoozed means don't show
        with patch("hermes_cli.adoption.detect_legacy_install") as mock:
            mock.return_value = MagicMock(pristine=True)
            assert not should_offer(hermes_home, project_root, adopt_mode="prompt")

    def test_non_legacy_returns_false(self, hermes_home, project_root):
        with patch("hermes_cli.adoption.detect_legacy_install") as mock:
            mock.return_value = None  # Not a legacy install
            assert not should_offer(hermes_home, project_root, adopt_mode="prompt")

    def test_legacy_prompt_interactive_returns_true(self, hermes_home, project_root):
        with patch("hermes_cli.adoption.detect_legacy_install") as mock:
            mock.return_value = MagicMock(pristine=True)
            assert should_offer(
                hermes_home, project_root,
                adopt_mode="prompt", is_interactive=True,
            )

    def test_legacy_prompt_non_interactive_returns_false(self, hermes_home, project_root):
        with patch("hermes_cli.adoption.detect_legacy_install") as mock:
            mock.return_value = MagicMock(pristine=True)
            assert not should_offer(
                hermes_home, project_root,
                adopt_mode="prompt", is_interactive=False,
            )

    def test_auto_pristine_non_interactive_returns_true(self, hermes_home, project_root):
        with patch("hermes_cli.adoption.detect_legacy_install") as mock:
            mock.return_value = MagicMock(pristine=True)
            assert should_offer(
                hermes_home, project_root,
                adopt_mode="auto", is_interactive=False,
            )

    def test_detection_failure_returns_false(self, hermes_home, project_root):
        with patch("hermes_cli.adoption.detect_legacy_install", side_effect=Exception("boom")):
            assert not should_offer(hermes_home, project_root, adopt_mode="prompt")


class TestOfferAdoption:
    def test_never_raises(self, hermes_home, project_root):
        """The offer NEVER raises — even on errors."""
        with patch("hermes_cli.adoption.detect_legacy_install", side_effect=Exception("crash")):
            # Should not raise
            offer_adoption(hermes_home, project_root, adopt_mode="prompt")

    def test_prints_offer_for_legacy(self, hermes_home, project_root, capsys):
        with patch("hermes_cli.adoption.detect_legacy_install") as mock:
            mock.return_value = MagicMock(pristine=True)
            offer_adoption(
                hermes_home, project_root,
                adopt_mode="prompt", is_interactive=True,
            )
        captured = capsys.readouterr()
        assert "hermes adopt" in captured.out

    def test_silent_for_never_mode(self, hermes_home, project_root, capsys):
        offer_adoption(hermes_home, project_root, adopt_mode="never")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_marks_shown_after_offer(self, hermes_home, project_root):
        with patch("hermes_cli.adoption.detect_legacy_install") as mock:
            mock.return_value = MagicMock(pristine=True)
            offer_adoption(
                hermes_home, project_root,
                adopt_mode="prompt", is_interactive=True,
            )
        assert _is_snoozed(hermes_home)

    def test_auto_invokes_adopt_subprocess(self, hermes_home, project_root):
        with patch("hermes_cli.adoption.detect_legacy_install") as mock, \
             patch("subprocess.Popen") as mock_popen:
            mock.return_value = MagicMock(pristine=True)
            offer_adoption(
                hermes_home, project_root,
                adopt_mode="auto", is_interactive=False,
            )
            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            assert "adopt" in args
