"""One profile setting reaches every Discord thread creation consumer."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from gateway.config import Platform, load_gateway_config
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["top-level", "nested-with-root", "nested-and-root-conflict"])
async def test_profile_duration_reaches_gateway_handoff_auto_and_model_tool(tmp_path, monkeypatch, shape):
    from tools import discord_tool
    homes=[tmp_path/"a",tmp_path/"b"]
    for home,value in zip(homes,[10080,"invalid"]):
        home.mkdir()
        config = {"platforms": {"discord": {"enabled": True}},
                  "discord": {"thread_auto_archive_duration": value}}
        if shape != "top-level":
            config["platforms"]["discord"]["extra"] = {"thread_auto_archive_duration": value}
            config["discord"] = {"require_mention": False}
            if shape == "nested-and-root-conflict":
                config["discord"]["thread_auto_archive_duration"] = 60
        (home/"config.yaml").write_text(json.dumps(config), encoding="utf-8")
    requests=[]
    def request(method,path,token,**kwargs):
        requests.append(kwargs["body"])
        return {"id":"999","name":"Created"}
    monkeypatch.setattr(discord_tool,"_discord_request",request)
    monkeypatch.setattr(discord_tool,"_get_bot_token",lambda:"test-token")
    thread=SimpleNamespace(id=999,name="Created",send=AsyncMock())
    for home,duration in [(homes[0],10080),(homes[1],1440),(homes[0],10080)]:
        scope=set_hermes_home_override(home)
        try:
            adapter=DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
            parent=SimpleNamespace(id=123,create_thread=AsyncMock(return_value=thread))
            adapter._client=SimpleNamespace(get_channel=lambda cid:parent)
            interaction=SimpleNamespace(channel=parent,user=SimpleNamespace(display_name="Example"))
            result=await adapter._create_thread(interaction,name="Manual")
            assert result["success"]
            assert parent.create_thread.call_args.kwargs["auto_archive_duration"] == duration
            assert parent.create_thread.call_args.kwargs["type"].value == 11
            assert await adapter.create_handoff_thread("123","Handoff") == "999"
            assert parent.create_thread.call_args.kwargs["auto_archive_duration"] == duration
            assert parent.create_thread.call_args.kwargs["type"].value == 11
            message=SimpleNamespace(content="Hello",author=SimpleNamespace(display_name="Example"),
                                    create_thread=AsyncMock(return_value=thread))
            assert await adapter._auto_create_thread(message) is thread
            assert message.create_thread.call_args.kwargs["auto_archive_duration"] == duration
            discord_tool.discord_core(action="create_thread",channel_id="123",name="From model")
            assert requests[-1]["auto_archive_duration"] == duration and requests[-1]["type"] == 11
            assert type(requests[-1]["auto_archive_duration"]) is int
            discord_tool.discord_core(action="create_thread",channel_id="123",name="Override",auto_archive_duration=60)
            assert requests[-1]["auto_archive_duration"] == 60
            count=len(requests)
            calls=parent.create_thread.await_count
            for invalid in [17,60.0,True,[],{},"60"]:
                result=json.loads(discord_tool.discord_core(action="create_thread",channel_id="123",name="Invalid",auto_archive_duration=invalid))
                assert result.get("error") and len(requests)==count
                rejected=await adapter._create_thread(interaction,name="Invalid",auto_archive_duration=invalid)
                assert rejected.get("error") and parent.create_thread.await_count==calls
        finally:
            reset_hermes_home_override(scope)
