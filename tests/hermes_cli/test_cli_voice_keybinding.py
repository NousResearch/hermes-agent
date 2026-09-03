"""Regression test for #101757.

``HermesCLI.run()`` registers the voice push-to-talk keybinding via
``_resolve_voice_keybinding()``. That helper must never raise: any failure
while loading ``voice.record_key`` from config (a broken/unreadable
config.yaml, reported from a Docker deployment) has to fall back to the
documented Ctrl+B default instead of crashing the whole CLI at startup with
``UnboundLocalError: cannot access local variable 'pt_key_to_sequence'``.
"""

import sys
import types

import cli


def test_falls_back_to_ctrl_b_when_config_import_fails(monkeypatch):
    """Simulates the reported crash: importing ``hermes_cli.config`` (or
    anything else in the try block before the real ``pt_key_to_sequence``
    is imported) fails. The old inline code left ``pt_key_to_sequence``
    unbound in that case and crashed on the very next line; the extracted
    helper must instead return the Ctrl+B default without raising."""
    fake_config_module = types.ModuleType("hermes_cli.config")  # no load_config attribute
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_config_module)

    raw_key, sequence = cli._resolve_voice_keybinding()

    assert raw_key == "ctrl+b"
    assert sequence == ("c-b",)


def test_resolves_configured_key_on_the_happy_path(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"voice": {"record_key": "alt+v"}},
    )

    raw_key, sequence = cli._resolve_voice_keybinding()

    assert raw_key == "alt+v"
    assert sequence == ("escape", "v")
