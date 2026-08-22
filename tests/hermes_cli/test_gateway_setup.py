"""Gateway setup menu regression coverage."""

from hermes_cli import gateway as gateway_cli
from hermes_cli import setup as setup_mod


def test_webhooks_are_listed_and_dispatch_to_setup():
    platform = next(item for item in gateway_cli._all_platforms() if item["key"] == "webhook")
    assert gateway_cli._builtin_setup_fn(platform["key"]) is setup_mod._setup_webhooks
    assert gateway_cli._builtin_setup_fn("webhooks") is setup_mod._setup_webhooks
