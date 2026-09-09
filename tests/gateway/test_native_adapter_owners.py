"""Native sibling owners keep SDK binding and package-scoped facade reads live."""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import get_type_hints
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["telegram", "discord", "matrix"])
@pytest.mark.parametrize("scoped", [False, True])
async def test_delivery_uses_its_own_live_facade(platform, scoped, monkeypatch):
    package = f"plugins.platforms.{platform}"
    if scoped:
        directory = Path(__file__).resolve().parents[2] / "plugins/platforms" / platform
        package = f"native_owner_scope_{platform}"
        spec = importlib.util.spec_from_file_location(
            package, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, package, module)
        spec.loader.exec_module(module)
    try:
        facade = importlib.import_module(f"{package}.adapter")
        cls = getattr(facade, f"{platform.title()}Adapter")
        adapter = cls(PlatformConfig(enabled=True, token="test-token", extra={}))
        if platform == "telegram":
            adapter._bot = None
        else:
            adapter._client = None
        # Patch AFTER construction: siblings must not freeze the facade's SDK/helpers.
        result = object()
        factory = Mock(return_value=result)
        monkeypatch.setattr(facade, "SendResult", factory)
        assert await adapter.send("123", "test") is result
        factory.assert_called_once()
        assert factory.call_args.kwargs["success"] is False
        assert get_type_hints(adapter.send)["return"].__name__ == "SendResult"
    finally:
        if scoped:
            for name in list(sys.modules):
                if name.startswith(package + "."):
                    sys.modules.pop(name)


def test_telegram_formatter_uses_live_facade_helper(monkeypatch):
    from plugins.platforms.telegram import adapter as facade

    adapter = facade.TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr(facade, "_escape_mdv2", lambda _text: "formatted")
    assert adapter.format_message("plain") == "formatted"
