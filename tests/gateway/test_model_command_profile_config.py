"""Regression coverage for profile-scoped gateway ``/model`` reads."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _CapturingPickerAdapter:
    def __init__(self):
        self.kwargs = None

    async def send_model_picker(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(success=True)


def _make_event():
    return MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="secondary-chat",
            chat_type="group",
        ),
    )


@pytest.mark.asyncio
async def test_model_picker_reads_routed_profile_config(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    default_home = tmp_path / "default"
    secondary_home = tmp_path / "profiles" / "secondary"
    default_home.mkdir()
    secondary_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "model:\n  default: default-model\n  provider: default-provider\n",
        encoding="utf-8",
    )
    (secondary_home / "config.yaml").write_text(
        "model:\n"
        "  default: 2070\n"
        "  provider: 2071\n"
        "  base_url: 2072\n"
        "providers:\n"
        "  2071:\n"
        "    base_url: https://example.invalid/v1\n"
        "model_catalog:\n"
        "  excluded_models:\n"
        "    2071: [blocked/*]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    listing_kwargs = {}

    def _list_picker_providers(**kwargs):
        listing_kwargs.update(kwargs)
        return [
            {
                "slug": "2071",
                "name": "Secondary Provider",
                "is_current": True,
                "models": ["2070"],
                "total_models": 1,
            }
        ]

    monkeypatch.setattr(
        "hermes_cli.model_switch_providers.list_picker_providers",
        _list_picker_providers,
    )

    runner = object.__new__(GatewayRunner)
    adapter = _CapturingPickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._resolve_profile_home_for_source = lambda _source: secondary_home
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
    runner._reply_anchor_for_event = lambda *_args, **_kwargs: None

    result = await runner._handle_model_command(_make_event())

    assert result is None
    assert adapter.kwargs is not None
    assert adapter.kwargs["current_model"] == "2070"
    assert adapter.kwargs["current_provider"] == "2071"
    assert listing_kwargs["current_model"] == "2070"
    assert listing_kwargs["current_provider"] == "2071"
    assert listing_kwargs["current_base_url"] == "2072"
    assert listing_kwargs["user_providers"] == {
        "2071": {"base_url": "https://example.invalid/v1"},
    }
    assert listing_kwargs["excluded_models"] == {2071: ["blocked/*"]}
