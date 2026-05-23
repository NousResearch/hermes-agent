"""Tests for Cursor SDK model selection (fast mode param)."""

from __future__ import annotations

import pytest

cursor_sdk = pytest.importorskip("cursor_sdk")

from agent.transports.cursor_sdk_session import build_cursor_model_selection


def test_composer_25_disables_fast_by_default():
    sel = build_cursor_model_selection("composer-2.5")
    assert isinstance(sel, cursor_sdk.ModelSelection)
    assert sel.id == "composer-2.5"
    assert len(sel.params) == 1
    assert sel.params[0].id == "fast"
    assert sel.params[0].value == "false"


def test_composer_25_fast_enables_fast():
    sel = build_cursor_model_selection("composer-2.5-fast")
    assert isinstance(sel, cursor_sdk.ModelSelection)
    assert sel.params[0].value == "true"


def test_auto_passes_through():
    assert build_cursor_model_selection("auto") == "auto"
