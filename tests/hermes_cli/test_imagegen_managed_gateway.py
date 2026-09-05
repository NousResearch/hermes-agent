"""Regression tests for image_gen provider persistence (managed FAL clobber).

Historical bug: ``_select_plugin_image_gen_provider`` hardcoded the direct
(non-managed) routing. When a user picked FAL through the Nous-subscription
managed flow, the managed write landed first — then the image selector ran
and clobbered it back to direct, silently routing every generation through
the user's personal FAL_KEY instead of the Nous Tool Gateway (real incident:
personal key drained to zero while the subscription sat unused).

Current contract (strict provider-string selection): each picker row writes
exactly ONE provider string per category — ``image_gen.provider: nous`` for
the managed "Nous Subscription" row, ``image_gen.provider: fal`` for the
BYOK FAL row — and any legacy ``use_gateway`` key is popped so the
read-time shim (use_gateway: true ⇒ nous) cannot override the fresh pick.
The video twin (``_select_plugin_video_gen_provider``) shares the contract.
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


def test_image_gen_selector_preserves_managed_selection(monkeypatch):
    """Managed pick: the 'nous' provider string must survive the selector."""
    _quiet(monkeypatch)
    config = {}

    # The managed flow first persists the managed selection...
    _write_provider_config(
        {"image_gen_plugin_name": "fal"}, config, managed_feature="image_gen"
    )
    assert config["image_gen"]["provider"] == "nous"
    assert "use_gateway" not in config["image_gen"]

    # ...then the selector runs; the managed kwarg must NOT clobber it
    # back onto the vendor name (direct-key routing).
    _select_plugin_image_gen_provider("fal", config, use_gateway=True)
    assert config["image_gen"]["provider"] == "nous"
    assert "use_gateway" not in config["image_gen"]


def test_image_gen_selector_direct_key_pick_writes_vendor(monkeypatch):
    """Non-managed pick writes the vendor name and pops the legacy flag."""
    _quiet(monkeypatch)
    config = {"image_gen": {"use_gateway": True}}

    _select_plugin_image_gen_provider("fal", config)
    assert config["image_gen"]["provider"] == "fal"
    assert "use_gateway" not in config["image_gen"]


def test_image_and_video_selectors_share_the_selection_contract(monkeypatch):
    """The two selectors are twins: same kwarg, same persistence behavior."""
    _quiet(monkeypatch)

    for use_gateway, expected in ((True, "nous"), (False, "fal")):
        config = {
            "image_gen": {"use_gateway": not use_gateway},
            "video_gen": {"use_gateway": not use_gateway},
        }
        _select_plugin_image_gen_provider("fal", config, use_gateway=use_gateway)
        _select_plugin_video_gen_provider("fal", config, use_gateway=use_gateway)
        assert config["image_gen"]["provider"] == expected
        assert config["video_gen"]["provider"] == expected
        assert "use_gateway" not in config["image_gen"]
        assert "use_gateway" not in config["video_gen"]


def _quiet_reconfigure(monkeypatch):
    """Silence prints + model pickers for _reconfigure_provider paths."""
    import hermes_cli.tools_config as tc

    monkeypatch.setattr(tc, "_print_success", lambda *a, **k: None)
    monkeypatch.setattr(tc, "_print_info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(tc, "_print_warning", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(tc, "_configure_imagegen_model", lambda *a, **k: None)
    monkeypatch.setattr(tc, "_run_post_setup", lambda *a, **k: None, raising=False)
    # Managed rows gate on live Portal auth — stub it green.
    import hermes_cli.nous_subscription as ns

    monkeypatch.setattr(ns, "ensure_nous_portal_access", lambda **k: True)


def test_reconfigure_managed_fal_row_keeps_managed_selection(monkeypatch):
    """The sibling bug of fe63353cb: the legacy-backend model-pick step in
    _reconfigure_provider hardcoded the direct selection AFTER the managed
    branch wrote the managed one — a Nous Subscription user re-entering the
    picker to change models was silently flipped onto their personal
    FAL_KEY."""
    _quiet_reconfigure(monkeypatch)
    import hermes_cli.tools_config as tc

    managed_row = {
        "name": "Nous Subscription",
        "env_vars": [],
        "requires_nous_auth": True,
        "managed_nous_feature": "image_gen",
        "override_env_vars": ["FAL_KEY"],
        "imagegen_backend": "fal",
    }
    config = {"image_gen": {"model": "fal-ai/gpt-image-2", "use_gateway": True}}

    tc._reconfigure_provider(managed_row, config)

    assert config["image_gen"]["provider"] == "nous"
    assert "use_gateway" not in config["image_gen"]


def test_reconfigure_direct_fal_row_writes_vendor_selection(monkeypatch):
    """Direct-key FAL reconfig writes the vendor name and pops any stale
    legacy use_gateway key so the read-time shim can't resurrect 'nous'."""
    _quiet_reconfigure(monkeypatch)
    import hermes_cli.tools_config as tc

    direct_row = {
        "name": "FAL.ai",
        "env_vars": [],
        "imagegen_backend": "fal",
    }
    config = {"image_gen": {"use_gateway": True}}

    tc._reconfigure_provider(direct_row, config)

    assert config["image_gen"]["provider"] == "fal"
    assert "use_gateway" not in config["image_gen"]
