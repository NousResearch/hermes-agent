"""Regression for #121938: lazy-installed Discord media uses its loaded plugin namespace."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send_document", "send_voice"])
async def test_lazy_installed_media_uses_active_adapter_module(tmp_path, monkeypatch, method):
    # Directory plugins are loaded as hermes_plugins.<slug>, not as
    # plugins.platforms.discord. Simulate the lazy installer rebinding only
    # the running adapter's discord global after its initial import failed.
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins/platforms/discord"
    name = "hermes_plugins.discord_media_regression"
    spec = importlib.util.spec_from_file_location(
        name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    active = sys.modules[f"{name}.adapter"]
    import plugins.platforms.discord.adapter as canonical
    monkeypatch.setattr(canonical, "discord", None)

    class File:
        def __init__(self, fp, filename):
            self.fp = fp
            self.filename = filename

    monkeypatch.setattr(active, "discord", SimpleNamespace(
        File=File, http=SimpleNamespace(Route=lambda *a, **kw: (a, kw)),
    ))
    channel = SimpleNamespace(
        id=222, send=AsyncMock(return_value=SimpleNamespace(id=444, attachments=[object()])),
    )
    request = AsyncMock(return_value={"id": "555"})
    adapter = active.DiscordAdapter(PlatformConfig())
    adapter._client = SimpleNamespace(
        get_channel=lambda channel_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=request),
    )
    path = tmp_path / "delivery.txt"
    path.write_bytes(b"attachment bytes")

    result = await getattr(adapter, method)("222", str(path))

    assert result.success
    if method == "send_document":
        files = channel.send.await_args.kwargs["files"]
        assert len(files) == 1 and isinstance(files[0], File)
        assert files[0].filename == path.name
    else:
        request.assert_awaited_once()
        assert result.message_id == "555"
