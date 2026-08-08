"""Seam tests for the CLITUIHooksMixin extraction (cli.py god-file slice R5).

Verifies that the four protected TUI extension hooks moved into
``hermes_cli/cli_tui_hooks_mixin.py`` remain reachable through ``HermesCLI``
with identical identity (MRO resolution), no back-import cycle, and the
behavioral contract wrapper CLIs depend on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.cli_tui_hooks_mixin import CLITUIHooksMixin

REPO_ROOT = Path(__file__).resolve().parents[2]

HOOK_NAMES = (
    "_apply_tui_skin_style",
    "_get_extra_tui_widgets",
    "_register_extra_tui_keybindings",
    "_build_tui_layout_children",
)


def _stub():
    """Isolate the mixin without paying HermesCLI.__init__ cost (house pattern)."""
    return object.__new__(CLITUIHooksMixin)


def _layout_kwargs(**overrides):
    kwargs = {
        "sudo_widget": "sudo",
        "secret_widget": "secret",
        "approval_widget": "approval",
        "slash_confirm_widget": "slash-confirm",
        "clarify_widget": "clarify",
        "model_picker_widget": "model-picker",
        "spinner_widget": "spinner",
        "spacer": "spacer",
        "status_bar": "status",
        "input_rule_top": "top-rule",
        "image_bar": "image-bar",
        "input_area": "input-area",
        "input_rule_bot": "bottom-rule",
        "voice_status_bar": "voice-status",
        "completions_menu": "completions-menu",
    }
    kwargs.update(overrides)
    return kwargs


class TestSeamIdentity:
    @pytest.mark.parametrize("name", HOOK_NAMES)
    def test_method_identity(self, name):
        assert getattr(HermesCLI, name) is getattr(CLITUIHooksMixin, name)

    def test_mixin_in_mro_after_house_mixins(self):
        mro = HermesCLI.__mro__
        assert CLITUIHooksMixin in mro
        house = (
            cli_mod.CLIAgentSetupMixin,
            cli_mod.CLICommandsMixin,
            cli_mod.CLIBillingMixin,
        )
        positions = [mro.index(m) for m in house] + [mro.index(CLITUIHooksMixin)]
        assert positions == sorted(positions), "house mixins precede CLITUIHooksMixin"

    def test_patch_binding_through_seam(self, monkeypatch):
        monkeypatch.setattr(
            CLITUIHooksMixin, "_build_tui_layout_children", lambda self, **kw: ["fake-child"]
        )
        inst = object.__new__(HermesCLI)
        assert inst._build_tui_layout_children() == ["fake-child"]


class TestNoBackImport:
    def test_module_imports_without_cli(self):
        code = (
            "import sys\n"
            "sys.modules['cli'] = None\n"
            "import hermes_cli.cli_tui_hooks_mixin as m\n"
            "assert m.CLITUIHooksMixin.__name__ == 'CLITUIHooksMixin'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_import_order_permutation(self):
        code = (
            "import hermes_cli.cli_tui_hooks_mixin\n"
            "import cli\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


class TestSkinStyleHook:
    def test_apply_tui_skin_style_false_without_app(self):
        assert _stub()._apply_tui_skin_style() is False

    def test_apply_tui_skin_style_true_updates_running_app(self, monkeypatch):
        inst = _stub()
        invalidate_calls = []
        inst._app = type("App", (), {"style": "old-style"})()
        inst._tui_style_base = {"accent": "#fff"}
        inst._build_tui_style_dict = lambda: {"x": 1}
        inst._invalidate = lambda min_interval=0.25: invalidate_calls.append(min_interval)

        seen = {}

        class FakePTStyle:
            @classmethod
            def from_dict(cls, style_dict):
                seen["dict"] = style_dict
                return "NEW-STYLE"

        monkeypatch.setattr("hermes_cli.cli_tui_hooks_mixin.PTStyle", FakePTStyle)
        assert inst._apply_tui_skin_style() is True
        assert seen["dict"] == {"x": 1}
        assert inst._app.style == "NEW-STYLE"
        assert invalidate_calls == [0.0]


class TestLayoutChildrenHook:
    def test_children_order_with_none_filtering(self):
        inst = _stub()
        children = inst._build_tui_layout_children(
            **_layout_kwargs(
                slash_confirm_widget=None,
                model_picker_widget=None,
                spinner_widget=None,
            )
        )
        # Stub has no _pet_widget / _stash_panel_widget attrs -> slots vanish.
        assert children[1:] == [
            "sudo", "secret", "approval", "clarify",
            "spacer", "status", "top-rule", "image-bar",
            "input-area", "bottom-rule", "voice-status", "completions-menu",
        ]

    def test_extra_widgets_interpolated_between_spacer_and_status(self):
        class _WrapperStub(CLITUIHooksMixin):
            def _get_extra_tui_widgets(self):
                return ["radio-menu", "mini-player"]

        inst = _WrapperStub()
        children = inst._build_tui_layout_children(**_layout_kwargs())
        spacer_idx = children.index("spacer")
        assert children[spacer_idx + 1 : spacer_idx + 3] == ["radio-menu", "mini-player"]
        assert children[spacer_idx + 3] == "status"

    def test_none_filter_removes_optional_slots(self):
        inst = _stub()
        children = inst._build_tui_layout_children(**_layout_kwargs())
        assert "slash-confirm" in children and "model-picker" in children and "spinner" in children
        filtered = inst._build_tui_layout_children(
            **_layout_kwargs(
                slash_confirm_widget=None,
                model_picker_widget=None,
                spinner_widget=None,
            )
        )
        assert "slash-confirm" not in filtered
        assert "model-picker" not in filtered
        assert "spinner" not in filtered


class TestKeybindingHook:
    def test_register_extra_keybindings_default_noop(self):
        from prompt_toolkit.key_binding import KeyBindings

        inst = _stub()
        kb = KeyBindings()
        assert inst._register_extra_tui_keybindings(kb, input_area=None) is None
        assert kb.bindings == []

    def test_override_is_dispatched(self):
        class _BindingStub(CLITUIHooksMixin):
            def __init__(self):
                self.seen = None

            def _register_extra_tui_keybindings(self, kb, *, input_area):
                self.seen = (kb, input_area)

        from prompt_toolkit.key_binding import KeyBindings

        inst = _BindingStub()
        kb = KeyBindings()
        inst._register_extra_tui_keybindings(kb, input_area="AREA")
        assert inst.seen == (kb, "AREA")
