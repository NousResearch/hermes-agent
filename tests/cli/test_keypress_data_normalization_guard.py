"""``install_keypress_data_normalization()`` must degrade, never raise.

cli.py installs every input patch inside one blanket ``except Exception: pass``,
so an AttributeError here is swallowed and silently takes down the installers
queued after it (``install_ignored_terminal_sequences`` today).
"""

from __future__ import annotations

import pytest

from hermes_cli import pt_input_extras


def _parser_class():
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    return Vt100Parser


def test_missing_private_method_degrades_to_a_no_op(monkeypatch):
    """A renamed/removed ``_call_handler`` returns 0 instead of raising."""
    monkeypatch.delattr(_parser_class(), "_call_handler", raising=False)
    assert pt_input_extras.install_keypress_data_normalization() == 0


def test_marker_cannot_outlive_the_wrapper(monkeypatch):
    """A class-level marker would outlive the wrapper and make the installer skip."""
    parser_cls = _parser_class()

    def _foreign_call_handler(self, key, insert_text):  # pragma: no cover - marker only
        return None

    monkeypatch.setattr(parser_cls, "_call_handler", _foreign_call_handler)
    assert pt_input_extras.install_keypress_data_normalization() == 1
    # A third party now replaces the handler, discarding our wrapper and its marker.
    monkeypatch.setattr(parser_cls, "_call_handler", _foreign_call_handler)
    assert pt_input_extras.install_keypress_data_normalization() == 1
