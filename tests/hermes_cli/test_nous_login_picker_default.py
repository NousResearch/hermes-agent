"""The post-login Nous model picker must not switch an existing setup on a bare Enter (#102943).

Before the fix the login caller passed no ``current_model`` and the picker's cursor started on
row 0, the first curated (flagship) model, so one stray newline after the device-code approval
rewrote ``model.provider``/``model.default``.
"""

from __future__ import annotations

import pytest

from hermes_cli import auth_model_picker as picker
from hermes_cli import auth_nous

MODELS = ["vendor/flagship", "vendor/mid", "vendor/small"]


@pytest.fixture
def bare_enter(monkeypatch):
    """Stand-in for curses_radiolist where the user presses Enter on the initial cursor row."""
    seen = {}

    def _radiolist(title, choices, selected=0, **kwargs):
        seen["choices"], seen["selected"] = list(choices), selected
        return selected

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", _radiolist)
    monkeypatch.setattr(picker, "_confirm_selection_guards", lambda *a, **kw: True)
    return seen


def test_skip_by_default_starts_on_skip_when_current_model_is_not_listed(bare_enter):
    assert picker._prompt_model_selection(MODELS, current_model="other/model", skip_by_default=True) is None
    assert bare_enter["choices"][bare_enter["selected"]] == picker._SKIP_LABEL


def test_skip_by_default_starts_on_listed_current_model(bare_enter):
    selected = picker._prompt_model_selection(MODELS, current_model="vendor/small", skip_by_default=True)
    assert selected == "vendor/small"
    assert bare_enter["selected"] == 0


def test_default_picker_still_starts_on_first_row(bare_enter):
    """`hermes model` flows are explicit model choices and keep their row-0 default."""
    assert picker._prompt_model_selection(MODELS) == "vendor/flagship"
    assert bare_enter["selected"] == 0


def _write_model_config(monkeypatch, model_cfg):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"model": model_cfg})


@pytest.mark.parametrize(
    ("model_cfg", "expected"),
    [
        # Another provider is configured: nothing on the Nous list to keep, so Skip.
        ({"provider": "openrouter", "default": "deepseek/deepseek-v4.1-flash"}, ("", True)),
        # Already on Nous: keep that model (Skip if it is no longer listed).
        ({"provider": "nous", "default": "vendor/mid"}, ("vendor/mid", True)),
        ({"provider": "NOUS ", "default": " vendor/mid "}, ("vendor/mid", True)),
        # Legacy bare-string model with no provider: still an existing setup.
        ("some/model", ("", True)),
        # Fresh install: nothing configured, first curated row stays the default.
        ("", ("", False)),
        ({"provider": "auto", "default": ""}, ("", False)),
        (None, ("", False)),
    ],
)
def test_login_picker_defaults(monkeypatch, model_cfg, expected):
    _write_model_config(monkeypatch, model_cfg)
    assert auth_nous._login_picker_defaults() == expected


def _run_login_picker(monkeypatch, model_cfg) -> dict:
    _write_model_config(monkeypatch, model_cfg)
    monkeypatch.setattr("hermes_cli.models.get_curated_nous_model_ids", lambda *a, **kw: list(MODELS))
    monkeypatch.setattr("hermes_cli.models.check_nous_free_tier", lambda **kw: False)
    monkeypatch.setattr("hermes_cli.models.union_with_portal_paid_recommendations",
                        lambda ids, pricing, portal: (ids, pricing))
    monkeypatch.setattr("hermes_cli.models.union_with_nous_on_sale_models", lambda ids, pricing: ids)
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *a, **kw: {})
    monkeypatch.setattr("hermes_cli.models_pricing.nous_policy_allowed_ids", lambda **kw: None)
    monkeypatch.setattr("hermes_cli.nous_account.nous_policy_notice", lambda **kw: "")
    calls = {}

    def _prompt(model_ids, **kwargs):
        calls.update(kwargs, model_ids=list(model_ids))
        return None

    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", _prompt)
    auth_nous._pick_nous_model_after_login({"agent_key": "k", "portal_base_url": ""}, "https://x.test/v1")
    return calls


def test_login_from_another_provider_defaults_to_skip(monkeypatch):
    calls = _run_login_picker(monkeypatch, {"provider": "openrouter", "default": "deepseek/deepseek-v4.1-flash"})
    assert calls["model_ids"] == MODELS
    assert calls["current_model"] == ""
    assert calls["skip_by_default"] is True


def test_relogin_on_nous_keeps_the_configured_model(monkeypatch):
    calls = _run_login_picker(monkeypatch, {"provider": "nous", "default": "vendor/mid"})
    assert calls["current_model"] == "vendor/mid"
    assert calls["skip_by_default"] is True


def test_first_login_on_a_fresh_install_keeps_row_zero_default(monkeypatch):
    calls = _run_login_picker(monkeypatch, "")
    assert calls["skip_by_default"] is False
