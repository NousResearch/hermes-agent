"""Tests for hermes_cli/webhook.py — webhook subscription helpers."""


def test_hermes_home_returns_path():
    from hermes_cli.webhook import _hermes_home
    result = _hermes_home()
    assert result is not None


def test_load_subscriptions_returns_dict():
    from hermes_cli.webhook import _load_subscriptions
    assert isinstance(_load_subscriptions(), dict)


def test_get_webhook_config_returns_dict():
    from hermes_cli.webhook import _get_webhook_config
    assert isinstance(_get_webhook_config(), dict)
