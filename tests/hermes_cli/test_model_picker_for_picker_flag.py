"""The GUI model picker must be built with the picker credential posture.

``build_model_options_payload`` builds the payload the API server, the dashboard and the TUI
pickers consume — and nothing else (all three call sites are ``GET /api/model/options``
variants). It used to call ``build_models_payload`` without ``for_picker``, so the shared lister
ran with the non-picker posture and dropped rows the CLI picker showed.

The visible case was an OAuth-subscription provider whose credential lives in an external store
(``~/.codex/auth.json``): the picker-specific credential branch in the lister is the one that
sees external stores, so the desktop/dashboard picker lost the whole "ChatGPT or Codex
Subscription" row while ``hermes model`` listed it with all its models (#124510).

These tests pin the contract at the payload seam: whatever else the payload build changes, the
picker credential posture must be passed through.
"""

from hermes_cli.inventory import build_model_options_payload, load_picker_context


def _capture(monkeypatch):
    """Replace the payload builder with a recorder that returns a minimal valid payload."""
    seen: list[dict] = []

    def _recorder(ctx, **kwargs):
        seen.append(kwargs)
        return {"providers": []}

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", _recorder)
    return seen


def test_picker_payload_passes_the_picker_credential_posture(monkeypatch, tmp_path):
    """The GUI picker payload must build with ``for_picker=True`` — the posture that keeps a
    provider row visible when its credential pool is entirely in cooldown or its credential
    lives in an external store. The CLI picker already passed it; the GUI payload now agrees."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen = _capture(monkeypatch)

    payload = build_model_options_payload(load_picker_context())

    assert seen, "the payload build never reached build_models_payload"
    assert payload == {"providers": []}
    assert all(kwargs.get("for_picker") is True for kwargs in seen), (
        f"the GUI picker payload must build with for_picker=True, got "
        f"{[kwargs.get('for_picker') for kwargs in seen]}"
    )


def test_picker_payload_keeps_the_posture_on_explicit_refresh(monkeypatch, tmp_path):
    """``refresh=True`` ("Refresh Models") is the blocking probe path — it must not silently
    drop the picker posture while it is at it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen = _capture(monkeypatch)

    build_model_options_payload(load_picker_context(), refresh=True)

    assert seen, "the refreshed payload build never reached build_models_payload"
    assert all(kwargs.get("for_picker") is True for kwargs in seen), (
        "an explicit refresh must keep the picker credential posture"
    )
