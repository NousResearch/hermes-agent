"""Slack topical methods preserve package-local SDK and helper bindings."""

import importlib
import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize('scoped', [False, True])
async def test_document_method_reads_its_own_live_facade(scoped, monkeypatch):
    package = 'plugins.platforms.slack'
    if scoped:
        package = 'slack_owner_scope_test'
        directory = Path(__file__).resolve().parents[2] / 'plugins/platforms/slack'
        spec = importlib.util.spec_from_file_location(
            package, directory / '__init__.py', submodule_search_locations=[str(directory)]
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, package, module)
        spec.loader.exec_module(module)
    try:
        facade = importlib.import_module(package + '.adapter')
        adapter = facade.SlackAdapter(PlatformConfig(enabled=True, token='test-token'))
        adapter._app = None
        expected = object()
        factory = Mock(return_value=expected)
        monkeypatch.setattr(facade, 'SendResult', factory)
        assert await adapter.send_document('D1', '/unused') is expected
        factory.assert_called_once_with(success=False, error='Not connected')
    finally:
        if scoped:
            for name in list(sys.modules):
                if name.startswith(package + '.'):
                    sys.modules.pop(name)
