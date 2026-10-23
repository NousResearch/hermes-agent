"""Regression test for #101757.

``CLITuiMixin._tui_voice_record_key_sequence()`` resolves the CLI/TUI voice
push-to-talk keybinding. It must never raise: any failure while loading
``voice.record_key`` from config (a broken/unreadable config.yaml, reported
from a Docker deployment) has to fall back to the documented Ctrl+B default
instead of crashing the whole CLI at startup with
``UnboundLocalError: cannot access local variable 'pt_key_to_sequence'``.

The resolution originally lived inline in ``HermesCLI.run()`` (cli.py) and
was later extracted into ``hermes_cli/cli_tui_mixin.py`` as part of an
unrelated god-file refactor; the bug (and this regression test) moved with
it.
"""

import sys
import types

import cli  # noqa: F401 - ensures cli.py (and its transitive imports) load before any test
from hermes_cli.cli_tui_mixin import CLITuiMixin


class _FakeCLI(CLITuiMixin):
    def __init__(self):
        self.cached_raw_key = None

    def set_voice_record_key_cache(self, raw_key):
        self.cached_raw_key = raw_key


def test_falls_back_to_ctrl_b_when_config_import_fails(monkeypatch):
    """Simulates the reported crash: importing ``hermes_cli.config`` (or
    anything else in the try block before the real ``pt_key_to_sequence``
    is imported) fails. The old inline code left ``pt_key_to_sequence``
    unbound in that case and crashed on the very next line; the fixed
    method must instead return the Ctrl+B default without raising."""
    fake_config_module = types.ModuleType("hermes_cli.config")  # no load_config attribute
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_config_module)

    cli = _FakeCLI()
    sequence = cli._tui_voice_record_key_sequence()

    assert sequence == ("c-b",)
    assert cli.cached_raw_key == "ctrl+b"


def test_resolves_configured_key_on_the_happy_path(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"voice": {"record_key": "alt+v"}},
    )

    cli = _FakeCLI()
    sequence = cli._tui_voice_record_key_sequence()

    assert sequence == ("escape", "v")
    assert cli.cached_raw_key == "alt+v"
