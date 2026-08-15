"""Regression tests for image_gen use_gateway persistence (managed FAL clobber).

Bug: ``_select_plugin_image_gen_provider`` hardcoded
``image_gen.use_gateway = False``. When a user picked FAL through the
Nous-subscription managed flow, ``_write_provider_config`` first set
``use_gateway = True`` — then the image selector ran and clobbered it back
to False, silently routing every generation through the user's personal
FAL_KEY instead of the Nous Tool Gateway (real incident: personal key
drained to zero while the subscription sat unused).

The video twin (``_select_plugin_video_gen_provider``) already accepted a
``use_gateway`` kwarg; these tests pin the image path to the same contract.
"""

from hermes_cli.tools_config import (
    _select_plugin_image_gen_provider,
    _select_plugin_video_gen_provider,
    _write_provider_config,
)


def _quiet(monkeypatch):
    import hermes_cli.tools_config as tc

    monkeypatch.setattr(tc, "_print_success", lambda *a, **k: None)
    monkeypatch.setattr(tc, "_print_info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(tc, "_configure_imagegen_model_for_plugin", lambda *a, **k: None)
    monkeypatch.setattr(tc, "_configure_videogen_model_for_plugin", lambda *a, **k: None)


def test_image_gen_selector_preserves_managed_gateway_flag(monkeypatch):
    """Managed pick: use_gateway=True must survive the selector."""
    _quiet(monkeypatch)
    config = {}

    # The managed flow first writes the managed flag...
    _write_provider_config(
        {"image_gen_plugin_name": "fal"}, config, managed_feature="image_gen"
    )
    assert config["image_gen"]["use_gateway"] is True

    # ...then the selector runs; passing the managed flag must NOT clobber it.
    _select_plugin_image_gen_provider("fal", config, use_gateway=True)
    assert config["image_gen"]["provider"] == "fal"
    assert config["image_gen"]["use_gateway"] is True


def test_image_gen_selector_direct_key_pick_clears_gateway(monkeypatch):
    """Non-managed pick keeps the historical default: direct key, no gateway."""
    _quiet(monkeypatch)
    config = {"image_gen": {"use_gateway": True}}

    _select_plugin_image_gen_provider("fal", config)
    assert config["image_gen"]["provider"] == "fal"
    assert config["image_gen"]["use_gateway"] is False


def test_image_and_video_selectors_share_the_gateway_contract(monkeypatch):
    """The two selectors are twins: same kwarg, same persistence behavior."""
    _quiet(monkeypatch)

    for use_gateway in (True, False):
        config = {}
        _select_plugin_image_gen_provider("fal", config, use_gateway=use_gateway)
        _select_plugin_video_gen_provider("fal", config, use_gateway=use_gateway)
        assert config["image_gen"]["use_gateway"] is use_gateway
        assert config["video_gen"]["use_gateway"] is use_gateway
