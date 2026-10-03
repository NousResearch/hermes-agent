"""Issue #121130: gateway setup wizard must not ask the service question twice.

On a machine where the gateway service is already installed but not
running, one ``gateway_setup()`` run asked to start it twice: once up
front in ``_wizard_service_status_block`` ("Start it now?") and again at
the end in ``_wizard_post_setup`` ("Start the gateway service?"). If the
user declined the first offer, the later step must skip the topic
instead of re-prompting.
"""

from __future__ import annotations

import types


def _drive_wizard(monkeypatch, *, installed: bool, running: bool):
    import hermes_cli.gateway as gw

    questions: list[str] = []

    monkeypatch.setattr(gw, "is_managed", lambda: False)
    monkeypatch.setattr(gw, "color", lambda s, *a: s)
    monkeypatch.setattr(gw, "Colors", types.SimpleNamespace(DIM=0, MAGENTA=0))
    monkeypatch.setattr(gw, "_is_service_installed", lambda: installed)
    monkeypatch.setattr(gw, "_is_service_running", lambda: running)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gw, "has_conflicting_systemd_units", lambda: False)
    monkeypatch.setattr(gw, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gw, "print_success", lambda *a: None)
    monkeypatch.setattr(gw, "print_warning", lambda *a: None)
    monkeypatch.setattr(gw, "print_info", lambda *a: None)
    monkeypatch.setattr(gw, "print_header", lambda *a: None)
    monkeypatch.setattr(gw, "_all_platforms", lambda: [{"emoji": "x", "label": "T"}])
    monkeypatch.setattr(gw, "_platform_status", lambda p: "configured")
    monkeypatch.setattr(gw, "_configure_platform", lambda p: None)
    monkeypatch.setattr(gw, "_served_profile_needs_no_service", lambda: False)
    monkeypatch.setattr(gw, "_setup_service_action", lambda *a, **k: None)

    def _no_backend(*a, **k):
        raise AssertionError("install flow reached for an installed service")

    monkeypatch.setattr(gw, "_service_backend", _no_backend)
    # len(platforms) == 1, so choice 1 == "Done": exit the platform loop at once.
    monkeypatch.setattr(gw, "prompt_choice", lambda *a, **k: 1)
    monkeypatch.setattr(
        gw, "prompt_yes_no", lambda q, *a, **k: questions.append(q) or False
    )

    gw.gateway_setup()
    return questions


class TestGatewaySetupAsksOnce:
    def test_start_offered_once_when_installed_but_stopped(self, monkeypatch):
        questions = _drive_wizard(monkeypatch, installed=True, running=False)
        starts = [q for q in questions if "Start" in q]
        assert len(starts) == 1, f"service start asked twice in one run: {questions}"

    def test_restart_still_offered_once_when_running(self, monkeypatch):
        questions = _drive_wizard(monkeypatch, installed=True, running=True)
        restarts = [q for q in questions if "Restart" in q]
        assert len(restarts) == 1, f"expected one restart offer: {questions}"
