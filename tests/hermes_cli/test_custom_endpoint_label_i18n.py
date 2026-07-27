"""The bare ``custom`` provider's display label is i18n-aware.

"Custom endpoint" is UI prose (unlike provider ids / model ids, which are
identifiers and stay untranslated), so it resolves through the locale
catalog at call time: ``locales/<lang>.yaml`` key ``provider.custom_endpoint``.

Guards the read-time resolution contract: the label must NOT be frozen into
module-level dicts at import time, because the active language comes from
config/env and can differ per process.
"""

from __future__ import annotations

from agent.i18n import t
from hermes_cli.model_switch import _bare_custom_provider_def
from hermes_cli.models import provider_label
from hermes_cli.providers import custom_endpoint_label, get_label


def test_catalog_key_exists_in_english():
    """A missing key would make t() return the key path itself — the label
    functions must always resolve to a human string."""
    assert t("provider.custom_endpoint", lang="en") == "Custom endpoint"


def test_labels_default_to_english():
    assert provider_label("custom") == "Custom endpoint"
    assert get_label("custom") == "Custom endpoint"
    assert custom_endpoint_label() == "Custom endpoint"


def test_labels_localize_with_language_env(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    zh_label = t("provider.custom_endpoint", lang="zh")
    assert zh_label != "Custom endpoint"  # zh catalog actually translates it
    assert provider_label("custom") == zh_label
    assert get_label("custom") == zh_label
    assert custom_endpoint_label() == zh_label


def test_bare_custom_provider_def_uses_localized_name(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    pdef = _bare_custom_provider_def("https://relay.example.com/v1")
    assert pdef is not None
    assert pdef.name == t("provider.custom_endpoint", lang="zh")


def test_other_provider_labels_unaffected():
    assert get_label("moa") == "Mixture of Agents"
    assert provider_label("moa") == "Mixture of Agents"
