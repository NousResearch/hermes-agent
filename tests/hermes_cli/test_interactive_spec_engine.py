"""Tests for the recursive interactive-spec menu engine (_run_interactive_spec).

The engine lives on HermesCLI and drives plugin-returned ``{"interactive": {...}}``
specs: arrow-key picker (curses), per-item actions, free-text prompts, and
nested child menus. These tests fake the I/O methods so the loop logic is
exercised deterministically without a real terminal.
"""

import sys
import types

import pytest


def _make_cli(monkeypatch, picker_choices, prompts):
    """Build a HermesCLI-ish object with faked I/O.

    picker_choices: list of int|None — successive _run_curses_picker returns.
    prompts: list of str|None — successive _prompt_text_input returns.
    """
    from cli import HermesCLI

    # Avoid the heavy real __init__: build a bare object of the class.
    cli = HermesCLI.__new__(HermesCLI)
    picks = list(picker_choices)
    txts = list(prompts)
    printed = []

    def fake_picker(self, title, items, default_index=0):
        if picks:
            return picks.pop(0)
        return None

    def fake_prompt(prompt_text):
        printed.append(("prompt", prompt_text))
        if txts:
            return txts.pop(0)
        return None

    captured = {"print": printed}

    def fake_cprint(text):
        printed.append(("print", text))

    monkeypatch.setattr(cli, "_run_curses_picker", fake_picker)
    monkeypatch.setattr(cli, "_prompt_text_input", fake_prompt)
    # _cprint is module-global; patch the name visible inside the method's module
    import cli as cli_mod
    monkeypatch.setattr(cli_mod, "_cprint", fake_cprint)
    return cli, captured


def test_item_detail_print_on_select():
    """Selecting an item with no actions/detail prints its detail."""
    import cli as cli_mod
    from cli import HermesCLI

    # Need module globals wired for the method; patch via importing engine fn.
    cli = HermesCLI.__new__(HermesCLI)
    picks = [0]  # pick first item

    def fake_picker(self, title, items, default_index=0):
        return picks.pop(0) if picks else None

    printed = []

    def fake_prompt(self, p):
        return None

    captured = {"print": printed}

    def fake_cprint(text):
        printed.append(text)

    import types

    cli._run_curses_picker = types.MethodType(fake_picker, cli)
    cli._prompt_text_input = types.MethodType(fake_prompt, cli)
    monkeypatch_local = None
    # Patch module-global _cprint used by the engine.
    import sys
    cli_mod._cprint = fake_cprint

    spec = {
        "title": "Peers",
        "items": [
            {"label": "peer-a", "value": "a1", "detail": "Detail for peer-a"},
        ],
    }
    cli._run_interactive_spec(spec, "peers")
    assert any("Detail for peer-a" in p for p in printed), printed


def test_action_with_prompt_collects_text_then_runs():
    """An action with `prompt` collects text, then runs handler(value, text)."""
    import cli as cli_mod
    from cli import HermesCLI
    import types

    cli = HermesCLI.__new__(HermesCLI)
    picks = [0, 0]  # item 0, then action 0 in the action sub-picker
    prompts = ["hello world"]

    printed = []

    def fake_picker(self, title, items, default_index=0):
        return picks.pop(0) if picks else None

    def fake_prompt(self, p):
        return prompts.pop(0) if prompts else None

    def fake_cprint(text):
        printed.append(text)

    cli._run_curses_picker = types.MethodType(fake_picker, cli)
    cli._prompt_text_input = types.MethodType(fake_prompt, cli)
    cli_mod._cprint = fake_cprint

    def handler(value, text=None):
        return f"Sent to {value}: {text}"

    spec = {
        "title": "Peers",
        "items": [
            {
                "label": "peer-a",
                "value": "a1",
                "actions": [
                    {"key": "s", "label": "Send", "handler": handler, "prompt": "Message:"},
                ],
            },
        ],
    }
    cli._run_interactive_spec(spec, "peers")
    assert any("Sent to a1: hello world" in p for p in printed), printed


def test_action_with_children_feed_value():
    """A `children` action recurses, then feeds chosen value to handler."""
    import cli as cli_mod
    from cli import HermesCLI
    import types

    cli = HermesCLI.__new__(HermesCLI)
    # item 0 -> action 0 (has children) -> child picker returns index 1 (hold)
    picks = [0, 0, 1]
    prompts = []

    printed = []

    def fake_picker(self, title, items, default_index=0):
        return picks.pop(0) if picks else None

    def fake_prompt(self, p):
        return prompts.pop(0) if prompts else None

    def fake_cprint(text):
        printed.append(text)

    cli._run_curses_picker = types.MethodType(fake_picker, cli)
    cli._prompt_text_input = types.MethodType(fake_prompt, cli)
    cli_mod._cprint = fake_cprint

    def handler(value, text=None):
        return f"Policy for {value}: {text}"

    spec = {
        "title": "Peers",
        "items": [
            {
                "label": "peer-a",
                "value": "a1",
                "actions": [
                    {
                        "key": "p",
                        "label": "Policy",
                        "handler": handler,
                        "children": {
                            "title": "Pick policy",
                            "items": [
                                {"label": "accept", "value": "accept"},
                                {"label": "hold", "value": "hold"},
                                {"label": "refuse", "value": "refuse"},
                            ],
                        },
                    },
                ],
            },
        ],
    }
    cli._run_interactive_spec(spec, "peers")
    assert any("Policy for a1: hold" in p for p in printed), printed


def test_esc_exits_cleanly():
    """An Esc (None) at the root level prints nothing and returns."""
    import cli as cli_mod
    from cli import HermesCLI
    import types

    cli = HermesCLI.__new__(HermesCLI)
    picks = [None]  # immediate Esc
    prompts = []

    printed = []

    def fake_picker(self, title, items, default_index=0):
        return picks.pop(0) if picks else None

    def fake_prompt(self, p):
        return prompts.pop(0) if prompts else None

    def fake_cprint(text):
        printed.append(text)

    cli._run_curses_picker = types.MethodType(fake_picker, cli)
    cli._prompt_text_input = types.MethodType(fake_prompt, cli)
    cli_mod._cprint = fake_cprint

    spec = {"title": "Peers", "items": [{"label": "x", "value": "x1"}]}
    cli._run_interactive_spec(spec, "peers")
    assert printed == [], printed


def test_prompt_actions_route_through_free_text_modal():
    """Prompt actions use _prompt_free_text_modal (works from daemon thread)."""
    import cli as cli_mod
    from cli import HermesCLI
    import types

    cli = HermesCLI.__new__(HermesCLI)
    picks = [0, 0]  # item 0, action 0
    prompts = ["modal text"]

    printed = []
    called_with = {}

    def fake_picker(self, title, items, default_index=0):
        return picks.pop(0) if picks else None

    def fake_free_text(self, title, prompt):
        called_with["title"] = title
        called_with["prompt"] = prompt
        return prompts.pop(0) if prompts else None

    def fake_cprint(text):
        printed.append(text)

    cli._run_curses_picker = types.MethodType(fake_picker, cli)
    cli._prompt_free_text_modal = types.MethodType(fake_free_text, cli)
    cli_mod._cprint = fake_cprint

    def handler(value, text=None):
        return f"Created {value} with {text}"

    spec = {
        "title": "Groups",
        "items": [
            {
                "label": "Create group",
                "value": "create",
                "actions": [
                    {"key": "c", "label": "Create", "handler": handler,
                     "prompt": "New group name:"},
                ],
            },
        ],
    }
    cli._run_interactive_spec(spec, "groups")
    assert called_with.get("prompt") == "New group name:", called_with
    assert any("Created create with modal text" in p for p in printed), printed


@pytest.fixture(autouse=True)
def _restore_cli_cprint():
    """The tests below assign ``cli_mod._cprint`` directly (no monkeypatch), so
    the fake leaks past test teardown and silently no-ops later modules that
    assert on real ``_cprint`` recording (e.g.
    test_interrupt_output_history_regression). Snapshot and restore."""
    import cli as cli_mod

    original = cli_mod._cprint
    yield
    cli_mod._cprint = original
