"""A silenced session-info field must not leave a blank row in the reset notice.

The reset notice is assembled from one localized line per field (model / provider /
context), joined with ``\\n``. A locale overlay may deliberately silence a field by
setting it to an empty string (``gateway.session.info_provider: ""``), and the
endpoint line is skipped when the base URL is public. Joining verbatim then leaves
an empty row behind — a stray blank line in the middle of the notice.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def turn_mixin(tmp_path, monkeypatch):
    """A GatewayTurnMixin bound to a stub resolver; no gateway runtime is started.

    ``HERMES_HOME`` is redirected so a user locale overlay on the host running these
    tests cannot leak into them.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent import i18n

    i18n.reset_language_cache()

    from gateway.run_turn import GatewayTurnMixin
    import gateway.run_turn as mod

    stub = SimpleNamespace(
        model="test-model",
        provider="test-provider",
        context_length=200_000,
        context_source="config",
        base_url="https://api.example.com/v1",
    )
    import gateway.run

    monkeypatch.setattr(gateway.run, "_resolve_gateway_model_context",
                        lambda *a, **k: stub, raising=False)
    return GatewayTurnMixin.__new__(GatewayTurnMixin)


def test_silenced_field_leaves_no_blank_line(turn_mixin, monkeypatch):
    """``info_provider`` silenced by a locale overlay must not produce an empty row."""
    import gateway.run_turn as mod

    real_t = mod.t

    def fake_t(key, **kwargs):
        if key == "gateway.session.info_provider":
            return ""          # overlay silenced this field
        return real_t(key, **kwargs)

    monkeypatch.setattr(mod, "t", fake_t)

    out = turn_mixin._format_session_info()

    assert "\n\n" not in out, f"blank row left in notice: {out!r}"
    assert not out.startswith("\n") and not out.endswith("\n"), repr(out)
    assert out.count("\n") >= 1, "other fields must still be rendered"


def test_no_silencing_keeps_every_field(turn_mixin, monkeypatch):
    """Baseline: with nothing silenced, all three labelled fields are present."""
    out = turn_mixin._format_session_info()

    assert "\n\n" not in out, repr(out)
    for needle in ("test-model", "test-provider"):
        assert needle in out, f"{needle} missing from {out!r}"
