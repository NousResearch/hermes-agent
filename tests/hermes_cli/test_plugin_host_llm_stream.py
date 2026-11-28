"""The request/response plugin IPC must never impersonate a live iterator."""

import pytest

from hermes_cli.plugin_host import PluginHost, PluginHostUnsupported


@pytest.mark.parametrize('method', ['astream', 'stream'])
def test_streaming_is_refused_before_facade_dispatch(method):
    host = PluginHost.__new__(PluginHost)
    host._plugin_ctx = lambda params: pytest.fail('must refuse before creating or calling a facade')
    with pytest.raises(PluginHostUnsupported, match='streaming.*IPC'):
        host._serve_facade({'facade': 'llm', 'method': method})
