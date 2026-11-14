"""alibaba-coding-plan and alibaba-coding-plan-cn must not both appear in the
/model picker off a single shared key (#101122).

The CN profile now has its own ALIBABA_CODING_PLAN_CN_API_KEY (checked first),
keeping the shared ALIBABA_CODING_PLAN_API_KEY / DASHSCOPE_API_KEY as ordered
fallbacks so existing CN users are not broken.  The picker hides a ``-cn`` row
whose only lit vars are shared with a lit non-CN sibling row.
"""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers

_CLEAR = {k: "" for k in ("ALIBABA_CODING_PLAN_API_KEY", "ALIBABA_CODING_PLAN_CN_API_KEY", "DASHSCOPE_API_KEY")}


def _alibaba_slugs(current_provider=""):
    return [p["slug"] for p in list_authenticated_providers(current_provider=current_provider) if "coding-plan" in p["slug"]]


def _available_slugs():
    """``list_available_providers`` rows the WebUI model list treats as usable."""
    from hermes_cli.models import list_available_providers

    return [p["id"] for p in list_available_providers() if p["authenticated"] and "coding-plan" in p["id"]]


@patch.dict(os.environ, {**_CLEAR, "ALIBABA_CODING_PLAN_CN_API_KEY": "sk-cn-fake"}, clear=False)
def test_alibaba_cn_appears_when_only_cn_key_set():
    assert _alibaba_slugs() == ["alibaba-coding-plan-cn"]


@patch.dict(os.environ, {**_CLEAR, "ALIBABA_CODING_PLAN_API_KEY": "sk-intl-fake"}, clear=False)
def test_alibaba_cn_does_not_appear_when_only_intl_key_set():
    """#101122: the shared intl key alone must light only the intl row."""
    assert _alibaba_slugs() == ["alibaba-coding-plan"]


@patch.dict(os.environ, {**_CLEAR, "ALIBABA_CODING_PLAN_API_KEY": "sk-intl-fake"}, clear=False)
def test_available_providers_agrees_with_the_picker_on_a_shared_intl_key():
    """#148137: the WebUI listing marked the CN twin authenticated off the sibling's key.

    ``list_available_providers`` reports one ``authenticated`` flag per canonical provider,
    so the same phantom-twin rule the picker applies has to gate that flag too.
    """
    assert _available_slugs() == _alibaba_slugs() == ["alibaba-coding-plan"]


@patch.dict(os.environ, {**_CLEAR, "ALIBABA_CODING_PLAN_CN_API_KEY": "sk-cn-fake"}, clear=False)
def test_available_providers_keeps_the_cn_row_for_its_own_key():
    """The dedicated CN key is real evidence: both listings keep the twin."""
    assert _available_slugs() == _alibaba_slugs() == ["alibaba-coding-plan-cn"]


@patch.dict(os.environ, {**_CLEAR, "ALIBABA_CODING_PLAN_API_KEY": "sk-intl-fake"}, clear=False)
def test_available_providers_keeps_the_current_cn_twin(monkeypatch):
    """A user who is on the CN endpoint keeps its row even when only the shared key is set."""
    from hermes_cli import auth

    monkeypatch.setattr(auth, "get_active_provider", lambda: "alibaba-coding-plan-cn")
    assert "alibaba-coding-plan-cn" in _available_slugs()
    assert "alibaba-coding-plan-cn" in _alibaba_slugs(current_provider="alibaba-coding-plan-cn")


@patch.dict(os.environ, {**_CLEAR, "ALIBABA_CODING_PLAN_API_KEY": "sk-intl-fake"}, clear=False)
def test_available_providers_keeps_a_cn_twin_pinned_in_config(monkeypatch):
    """``model.provider`` in config.yaml is the current provider, with no auth-store pick at all.

    ``resolve_provider`` reads config.yaml before auth.json, so a home pinned to the twin keeps the
    row the picker shows for it — reading only the auth store hid it on the WebUI listing.
    """
    from hermes_cli import auth

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **k: {"model": {"provider": "alibaba-coding-plan-cn"}},
    )
    monkeypatch.setattr(auth, "get_active_provider", lambda: None)
    assert "alibaba-coding-plan-cn" in _available_slugs()
    assert sorted(_available_slugs()) == sorted(_alibaba_slugs(current_provider="alibaba-coding-plan-cn"))
