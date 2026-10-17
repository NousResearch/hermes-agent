"""`/api/model/info` must not hand clients a deleted MoA preset (#82613).

When ``model.default`` names a MoA preset that was deleted (or renamed) from
``moa.presets``, the endpoint used to return the dead name verbatim — the
Desktop seeded its composer with a model absent from the picker catalog and the
first message crashed with ``MoAPresetNotFoundError``. The manual-pick path is
already guarded client-side (#90244); this is the default-derived counterpart,
fixed at the source so every client (desktop, TUI, web dashboard) benefits.
"""

from __future__ import annotations


def _config(model: str, moa: dict) -> dict:
    return {"model": {"default": model, "provider": "moa"}, "moa": moa}


def _get_info(monkeypatch, config: dict) -> dict:
    from hermes_cli.web_routers import models as router

    monkeypatch.setattr(router, "_load_config_scoped", lambda profile: config)
    import agent.model_metadata as metadata

    monkeypatch.setattr(
        metadata, "get_model_context_length",
        lambda *a, **k: 0,  # the probe is irrelevant to this contract
    )
    return router.get_model_info()


def test_model_info_keeps_a_valid_moa_default_verbatim(monkeypatch):
    config = _config("TOP", {
        "default_preset": "TOP",
        "presets": {
            "TOP": {"reference_models": [], "aggregator": {}},
            "日常": {"reference_models": [], "aggregator": {}},
        },
    })

    info = _get_info(monkeypatch, config)

    assert info["model"] == "TOP"
    assert info["provider"] == "moa"
    assert info["stale_default"] is False


def test_model_info_falls_back_when_default_preset_was_deleted(monkeypatch):
    # ``model.default: TOP`` but only ``日常`` remains in ``moa.presets``.
    config = _config("TOP", {
        "default_preset": "TOP",
        "presets": {"日常": {"reference_models": [], "aggregator": {}}},
    })

    info = _get_info(monkeypatch, config)

    # The same fallback ``normalize_moa_config`` applies to ``moa.default_preset``:
    # the first valid existing preset, never the deleted name.
    assert info["model"] == "日常"
    assert info["provider"] == "moa"
    assert info["stale_default"] is True


def test_model_info_degrades_to_builtin_preset_on_empty_moa_section(monkeypatch):
    # A hand-nuked ``moa`` section still yields the built-in default preset
    # (``normalize_moa_config`` seeds one); the deleted name never leaks.
    config = _config("TOP", {"default_preset": "TOP", "presets": {}})

    info = _get_info(monkeypatch, config)

    assert info["model"] != "TOP"
    assert info["model"]  # non-empty: the built-in "default" preset
    assert info["provider"] == "moa"
    assert info["stale_default"] is True
