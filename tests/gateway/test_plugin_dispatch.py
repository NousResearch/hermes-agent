"""Signature-tolerant plugin setup/connect dispatch (#97065)."""

import inspect

from gateway.plugin_dispatch import connect_adapter, invoke_setup_fn


def test_zero_arg_setup_fn_is_called_without_config():
    calls = []

    def setup():
        calls.append("zero")

    invoke_setup_fn(setup, config={"key": "keet"})
    assert calls == ["zero"]


def test_config_arg_setup_fn_receives_platform_dict():
    seen = []

    def setup(config):
        seen.append(config)

    platform = {"key": "keet", "label": "Keet"}
    invoke_setup_fn(setup, config=platform)
    assert seen == [platform]


def test_keyword_only_config_setup_fn():
    seen = []

    def setup(*, config):
        seen.append(config)

    invoke_setup_fn(setup, config="cfg")
    assert seen == ["cfg"]


class _ReconnectAdapter:
    def __init__(self):
        self.calls = []

    async def connect(self, *, is_reconnect: bool = False):
        self.calls.append(is_reconnect)
        return True


class _LegacyAdapter:
    def __init__(self):
        self.calls = []

    async def connect(self):
        self.calls.append("bare")
        return True


def test_connect_forwards_is_reconnect_when_accepted():
    adapter = _ReconnectAdapter()
    result = connect_adapter(adapter, is_reconnect=True)
    assert inspect.iscoroutine(result)

    async def _run():
        return await result

    import asyncio

    assert asyncio.run(_run()) is True
    assert adapter.calls == [True]


def test_connect_omits_is_reconnect_for_legacy_signature():
    adapter = _LegacyAdapter()
    result = connect_adapter(adapter, is_reconnect=True)
    import asyncio

    assert asyncio.run(result) is True
    assert adapter.calls == ["bare"]
