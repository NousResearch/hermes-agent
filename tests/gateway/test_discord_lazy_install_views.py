"""Regression: Discord UI view classes must be defined after lazy-install.

When discord.py is NOT installed at module load time, the
``if DISCORD_AVAILABLE:`` guard at the bottom of gateway/platforms/discord.py
evaluates to False and is skipped — leaving ExecApprovalView and its four
siblings undefined in the module globals.

check_discord_requirements() must call _define_discord_view_classes() after
a successful lazy install so that all view classes are available the moment
DISCORD_AVAILABLE flips to True.  Without this, the first button interaction
(exec approval, slash confirm, etc.) raises NameError even though
DISCORD_AVAILABLE=True.

Fixes: lazy-install path NameError for ExecApprovalView, SlashConfirmView,
UpdatePromptView, ModelPickerView, ClarifyChoiceView.
"""
import importlib
from unittest.mock import patch

import pytest


_VIEW_NAMES = [
    "ExecApprovalView",
    "SlashConfirmView",
    "UpdatePromptView",
    "ModelPickerView",
    "ClarifyChoiceView",
]


@pytest.fixture(autouse=True)
def _restore_discord_view_class_globals():
    """Snapshot + restore the 5 Discord view-class module globals (function scope).

    ``_define_discord_view_classes()`` (called by ``check_discord_requirements()``
    after a lazy install) does ``global ExecApprovalView, …, ClarifyChoiceView``
    and **re-defines** those classes as fresh class objects each call. The tests
    in this file deliberately exercise that path — which REBINDS the module
    globals to new class objects. ``monkeypatch`` reverts the spies/flags these
    tests set, but it does NOT restore the rebound view-class globals (the
    production call, not monkeypatch, rebound them), so a later test that captured
    the ORIGINAL class identity at its import time fails
    ``isinstance(view, ClarifyChoiceView)`` against a now-stale binding (same name,
    different class object — identical repr).

    This fixture snapshots the 5 globals at setup and restores them at teardown so
    a rebind can't escape the test that caused it. **Function-scoped** (NOT module)
    so it also reverts an intra-file rebind between two tests in this module under
    item-level shuffling. (The matching consumer-side guarantee — reading the class
    identity LIVE via the module attribute — lives in
    ``test_discord_clarify_buttons.py``; this restore + that live-ref are belt and
    suspenders.)
    """
    dp = importlib.import_module("plugins.platforms.discord.adapter")
    sentinel = object()
    snapshot = {name: getattr(dp, name, sentinel) for name in _VIEW_NAMES}
    try:
        yield
    finally:
        for name, original in snapshot.items():
            if original is sentinel:
                if hasattr(dp, name):
                    delattr(dp, name)
            else:
                setattr(dp, name, original)


class TestDefineDiscordViewClasses:
    """_define_discord_view_classes() registers all UI view classes in module globals."""

    def test_registers_all_five_view_classes(self, monkeypatch):
        """Calling _define_discord_view_classes() must (re)define all 5 view classes."""
        dp = importlib.import_module("plugins.platforms.discord.adapter")

        # Remove the classes to simulate the state where the module was loaded
        # with DISCORD_AVAILABLE=False (the lazy-install scenario).
        for name in _VIEW_NAMES:
            monkeypatch.delattr(dp, name)

        # Pre-condition: classes are gone
        for name in _VIEW_NAMES:
            assert not hasattr(dp, name), f"{name} should be absent before the call"

        dp._define_discord_view_classes()

        for name in _VIEW_NAMES:
            assert hasattr(dp, name), f"{name} must be defined after _define_discord_view_classes()"
            assert isinstance(getattr(dp, name), type), f"{name} must be a class"

    def test_check_discord_requirements_calls_define_on_lazy_install(self, monkeypatch):
        """check_discord_requirements() must call _define_discord_view_classes() on
        a successful lazy install so view classes exist when DISCORD_AVAILABLE=True."""
        dp = importlib.import_module("plugins.platforms.discord.adapter")

        # Simulate discord not yet available at module load.
        monkeypatch.setattr(dp, "DISCORD_AVAILABLE", False)

        define_called = [False]
        orig_define = dp._define_discord_view_classes

        def _spy_define():
            define_called[0] = True
            orig_define()

        monkeypatch.setattr(dp, "_define_discord_view_classes", _spy_define)

        # Patch lazy_deps.ensure to be a no-op (pretend install succeeds).
        # The discord imports inside check_discord_requirements() succeed because
        # _ensure_discord_mock() in conftest.py already registered the mock.
        with patch("tools.lazy_deps.ensure"):
            result = dp.check_discord_requirements()

        assert result is True, "check_discord_requirements() should return True after lazy install"
        assert define_called[0], (
            "check_discord_requirements() must call _define_discord_view_classes() "
            "after a successful lazy install so view classes are not undefined"
        )
