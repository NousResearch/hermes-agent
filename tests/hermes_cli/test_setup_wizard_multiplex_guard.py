"""Wizard must not install a standalone service when the profile is served by the
running multiplexed gateway (#111958).

Setup gateway wizard installs a stray standalone launchd/systemd service for a new
profile even though the default gateway (multiplex_profiles: true) already serves
it — the same gap #97120 closed for the CLI entry points (run/start/install),
but never applied to the interactive wizard path.
"""

import hermes_cli.gateway as gw


def _patch_wizard(monkeypatch, served=True, installed=False, running=False):
    monkeypatch.setattr(
        gw, "named_profile_served_by_running_multiplexer", lambda name=None: served
    )
    monkeypatch.setattr(gw, "_is_service_installed", lambda: installed)
    monkeypatch.setattr(gw, "_is_service_running", lambda: running)


def test_wizard_skips_standalone_install_when_served_by_multiplexer(monkeypatch, capsys):
    calls = []
    _patch_wizard(monkeypatch, served=True)
    monkeypatch.setattr(gw, "_wizard_install_service", lambda backend: calls.append(backend))
    monkeypatch.setattr(gw, "_service_backend", lambda: "launchd")

    gw._wizard_post_setup()

    assert calls == [], f"wizard installed a standalone service for a multiplex-served profile: {calls}"
    out = capsys.readouterr().out
    assert "multiplex" in out.lower()


def test_wizard_still_installs_when_not_served_by_multiplexer(monkeypatch):
    calls = []
    _patch_wizard(monkeypatch, served=False)
    monkeypatch.setattr(gw, "_wizard_install_service", lambda backend: calls.append(backend))
    monkeypatch.setattr(gw, "_service_backend", lambda: "launchd")

    gw._wizard_post_setup()

    assert calls == ["launchd"]
