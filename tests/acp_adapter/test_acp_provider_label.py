"""``_provider_label`` — a NAMED custom provider prints its name, not the runtime class ``custom``."""

from types import SimpleNamespace

from acp_adapter.commands import _provider_label


def test_named_custom_provider_uses_requested_name():
    agent = SimpleNamespace(provider="custom", requested_provider="engy")
    assert _provider_label(agent) == "engy"


def test_plain_provider_unchanged():
    assert _provider_label(SimpleNamespace(provider="nous")) == "nous"


def test_falls_back_to_class_then_default():
    assert _provider_label(SimpleNamespace(provider="custom")) == "custom"
    assert _provider_label(SimpleNamespace()) == "auto"
    assert _provider_label(SimpleNamespace(provider="custom"), default="") == "custom"