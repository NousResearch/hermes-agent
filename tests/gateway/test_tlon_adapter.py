from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType, SendResult
from gateway.platforms.tlon import (
    TlonAdapter,
    TlonSSEClient,
    _extract_message_text,
    _text_to_story,
)
from gateway.platforms.tlon_approval import (
    create_pending_approval,
    format_approval_request,
    format_pending_list,
    normalize_notification_id,
)
from gateway.platforms.tlon_discovery import parse_groups_ui_init
from gateway.platforms.tlon_media import (
    TlonDownloadedAttachment,
    format_blob_annotations,
    parse_blob_data,
)
from gateway.platforms.tlon_settings import parse_settings_response


def test_text_to_story_handles_mentions_links_code_and_images():
    story = _text_to_story(
        "hi ~zod\nlink https://example.com\n\n"
        "```python\nprint(1)\n```\n"
        "![alt](https://example.com/a.png)"
    )

    assert story[0] == {
        "inline": [
            "hi ",
            {"ship": "~zod"},
            {"break": None},
            "link ",
            {
                "link": {
                    "href": "https://example.com",
                    "content": "https://example.com",
                }
            },
        ]
    }
    assert story[1] == {
        "block": {
            "code": {
                "code": "print(1)",
                "lang": "python",
            }
        }
    }
    assert story[2] == {
        "block": {
            "image": {
                "src": "https://example.com/a.png",
                "alt": "alt",
                "width": 0,
                "height": 0,
            }
        }
    }


def test_extract_message_text_preserves_inline_spacing_and_blocks():
    content = [
        {
            "inline": [
                "hi ",
                {"bold": ["there"]},
                {"break": None},
                {"link": {"href": "https://example.com", "content": "site"}},
                " ",
                {"ship": "~zod"},
            ]
        },
        {"block": {"code": {"lang": "python", "code": "print(1)"}}},
    ]

    assert _extract_message_text(content) == (
        "hi there\nsite ~zod\n```python\nprint(1)\n```"
    )


def test_parse_blob_data_formats_supported_entries():
    entries = parse_blob_data(
        """
        [
          {"type":"file","fileUri":"https://example.com/report.pdf","mimeType":"application/pdf","name":"report.pdf","size":2048},
          {"type":"voicememo","fileUri":"https://example.com/memo.m4a","duration":4.2,"transcription":"hello"},
          {"type":"video","fileUri":"https://example.com/clip.mp4","mimeType":"video/mp4","name":"clip.mp4"},
          {"type":"unknown","fileUri":"https://example.com/nope"}
        ]
        """
    )

    assert [entry.type for entry in entries] == ["file", "voicememo", "video"]
    annotation = format_blob_annotations(entries)
    assert "[file: report.pdf (application/pdf, 2KB)] https://example.com/report.pdf" in annotation
    assert "[voice memo: 4s] https://example.com/memo.m4a" in annotation
    assert 'Transcription: "hello"' in annotation
    assert "[video: clip.mp4 (video/mp4, unknown size)] https://example.com/clip.mp4" in annotation


def test_parse_groups_ui_init_discovers_channels_and_names():
    parsed = parse_groups_ui_init({
        "groups": {
            "~host/test": {
                "meta": {"title": "Test Group"},
                "channels": {
                    "chat/~host/general": {},
                    "heap/~host/images": {},
                    "diary/~host/blog": {},
                    "bad-channel": {},
                },
            }
        },
        "foreigns": {"~else/group": {"invites": [{"valid": True}]}},
    })

    assert parsed.channels == {
        "chat/~host/general",
        "heap/~host/images",
        "diary/~host/blog",
    }
    assert parsed.channel_to_group["chat/~host/general"] == "~host/test"
    assert parsed.group_names["~host/test"] == "Test Group"
    assert "~else/group" in parsed.foreigns


def test_parse_settings_response_reads_tlon_bucket():
    settings = parse_settings_response({
        "all": {
            "moltbot": {
                "tlon": {
                    "groupChannels": ["chat/~host/general"],
                    "dmAllowlist": ["~zod"],
                    "autoDiscover": True,
                    "channelRules": '{"chat/~host/general":{"mode":"restricted","allowedShips":["~nec"]}}',
                    "defaultAuthorizedShips": ["~bus"],
                    "ownerShip": "~ten",
                    "ownerListenEnabled": False,
                    "ownerListenDisabledChannels": ["chat/~host/noisy"],
                }
            }
        }
    })

    assert settings.group_channels == ["chat/~host/general"]
    assert settings.dm_allowlist == ["~zod"]
    assert settings.auto_discover is True
    assert settings.channel_rules["chat/~host/general"]["allowedShips"] == ["~nec"]
    assert settings.default_authorized_ships == ["~bus"]
    assert settings.owner_ship == "~ten"
    assert settings.owner_listen_enabled is False
    assert settings.owner_listen_disabled_channels == ["chat/~host/noisy"]


@pytest.mark.asyncio
async def test_sse_broadcasts_unknown_subscription_id_like_openclaw():
    client = TlonSSEClient("http://ship.test", "code", "~bot-palnet")
    seen = []

    async def channel_handler(event):
        seen.append(("channels", event))

    async def dm_handler(event):
        seen.append(("chat", event))

    await client.subscribe(
        app="channels",
        path="/v2",
        on_event=channel_handler,
        on_error=None,
        on_quit=None,
    )
    await client.subscribe(
        app="chat",
        path="/v3",
        on_event=dm_handler,
        on_error=None,
        on_quit=None,
    )

    await client._process_event(
        'id: 1\n'
        'data: {"json":{"nest":"chat/~host/general","response":{"post":{}}}}\n'
    )

    assert [name for name, _event in seen] == ["channels", "chat"]
    assert seen[0][1]["nest"] == "chat/~host/general"


def test_approval_formatting_lists_pending_request():
    approval = create_pending_approval(
        approval_type="dm",
        requesting_ship="~zod",
        existing_ids=[],
        message_preview="hello",
    )

    pending = format_pending_list([approval])
    assert approval.id in pending
    assert "~zod" in pending


@pytest.mark.asyncio
async def test_connect_subscribes_to_openclaw_group_and_channel_read_paths(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_URL", "http://ship.test")
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_SHIP_CODE", "code")

    class FakeTlonSSE:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.subscriptions = []
            self._connected = False
            FakeTlonSSE.instances.append(self)

        async def authenticate(self):
            return None

        async def scry(self, path):
            if path == "/contacts/v1/self.json":
                return {}
            if path == "/settings/all.json":
                return {}
            return {}

        async def subscribe(self, *, app, path, on_event, on_error, on_quit):
            self.subscriptions.append((app, path))

        async def connect(self):
            self._connected = True

        async def close(self):
            self._connected = False

    monkeypatch.setattr("gateway.platforms.tlon.TlonSSEClient", FakeTlonSSE)

    adapter = TlonAdapter(PlatformConfig())
    assert await adapter.connect() is True

    subscriptions = FakeTlonSSE.instances[0].subscriptions
    assert ("channels", "/v2") in subscriptions
    assert ("chat", "/v3") in subscriptions
    assert ("groups", "/groups/ui") in subscriptions
    assert ("groups", "/v1/foreigns") in subscriptions

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_channel_event_auto_watches_chat_and_heap_like_openclaw(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("TLON_AUTO_DISCOVER", "false")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = set()
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/new",
        "response": {
            "post": {
                "id": "auto-watch-post",
                "r-post": {
                    "set": {
                        "seal": {"id": "auto-watch-post"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [{"inline": [{"ship": "~bot-palnet"}, " hello"]}],
                        },
                    }
                },
            }
        },
    })

    assert "chat/~host/new" in adapter.monitored_channels
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_groups_ui_event_watches_joined_chat_and_heap_channels(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    adapter = TlonAdapter(PlatformConfig())
    adapter.auto_accept_group_invites = True
    adapter._sse = AsyncMock()

    await adapter._handle_groups_ui_event({
        "flag": "~host/group",
        "channels": {
            "chat/~host/general": {},
            "heap/~host/gallery": {},
            "diary/~host/blog": {},
        },
        "join": {
            "group": "~host/group",
            "channels": ["chat/~host/joined", "diary/~host/notes"],
        },
    })

    assert adapter.monitored_channels == {
        "chat/~host/general",
        "heap/~host/gallery",
        "chat/~host/joined",
    }
    assert adapter._channel_to_group["chat/~host/general"] == "~host/group"
    assert adapter._channel_to_group["heap/~host/gallery"] == "~host/group"
    assert adapter._channel_to_group["chat/~host/joined"] == "~host/group"
    values = [
        call.kwargs["json_data"]["put-entry"]["value"]
        for call in adapter._sse.poke.await_args_list
    ]
    assert values[-1] == [
        "chat/~host/general",
        "heap/~host/gallery",
        "chat/~host/joined",
    ]


@pytest.mark.asyncio
async def test_foreigns_event_auto_accepts_group_invites(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    adapter = TlonAdapter(PlatformConfig())
    adapter.auto_accept_group_invites = True
    adapter._settings.group_invite_allowlist = ["~zod"]
    adapter._sse = AsyncMock()

    await adapter._handle_group_foreigns_event({
        "~host/group": {
            "invites": [{"valid": True, "ship": "~zod"}],
        }
    })

    adapter._sse.poke.assert_awaited_once()
    call = adapter._sse.poke.await_args.kwargs
    assert call["app"] == "groups"
    assert call["mark"] == "group-join"
    assert call["json_data"] == {"flag": "~host/group", "join-all": True}


@pytest.mark.asyncio
async def test_foreigns_event_empty_allowlist_is_fail_closed(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    adapter = TlonAdapter(PlatformConfig())
    adapter.auto_accept_group_invites = True
    adapter._settings.group_invite_allowlist = []
    adapter.owner_ship = "~malmur-halmex"
    adapter._sse = AsyncMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="~bot/170.141"))
    adapter._put_settings_entry = AsyncMock()

    await adapter._handle_group_foreigns_event({
        "~host/group": {
            "invites": [{"valid": True, "ship": "~zod"}],
        }
    })

    adapter._sse.poke.assert_not_awaited()
    assert len(adapter.pending_approvals) == 1
    assert adapter.pending_approvals[0].type == "group"
    assert adapter.pending_approvals[0].requesting_ship == "~zod"


@pytest.mark.asyncio
async def test_foreigns_event_queues_approval_when_auto_accept_disabled(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    adapter = TlonAdapter(PlatformConfig())
    adapter.auto_accept_group_invites = False
    adapter._settings.group_invite_allowlist = ["~zod"]
    adapter.owner_ship = "~malmur-halmex"
    adapter._sse = AsyncMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="~bot/170.141"))
    adapter._put_settings_entry = AsyncMock()

    await adapter._handle_group_foreigns_event({
        "~host/group": {
            "preview": {"meta": {"title": "Test Group"}},
            "invites": [{"valid": True, "ship": "~zod"}],
        }
    })

    adapter._sse.poke.assert_not_awaited()
    assert len(adapter.pending_approvals) == 1
    assert adapter.pending_approvals[0].group_flag == "~host/group"
    assert adapter.pending_approvals[0].group_title == "Test Group"


@pytest.mark.asyncio
async def test_foreigns_event_accepts_owner_invite_even_without_auto_accept(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    adapter = TlonAdapter(PlatformConfig())
    adapter.auto_accept_group_invites = False
    adapter.owner_ship = "~malmur-halmex"
    adapter._sse = AsyncMock()

    await adapter._handle_group_foreigns_event({
        "~host/group": {
            "invites": [{"valid": True, "ship": "~malmur-halmex"}],
        }
    })

    adapter._sse.poke.assert_awaited_once()


@pytest.mark.asyncio
async def test_channel_event_routes_top_level_mentions(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_ALLOW_ALL_USERS", "true")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter._channel_to_group["chat/~host/test"] = "~host/group"
    adapter._group_names["~host/group"] = "Test Group"
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "170141184507864167403996323545639550976",
                "r-post": {
                    "set": {
                        "seal": {"id": "170141184507864167403996323545639550976"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [
                                {"inline": [{"ship": "~bot-palnet"}, " hello"]}
                            ],
                        },
                    }
                },
            }
        },
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.message_id == "170141184507864167403996323545639550976"
    assert event.reply_to_message_id is None
    assert event.source.chat_id == "chat/~host/test"
    assert event.source.parent_chat_id == "~host/group"
    assert event.source.chat_name == "Test Group / test"
    assert event.source.user_id == "~zod"
    assert isinstance(event.timestamp, datetime)


@pytest.mark.asyncio
async def test_channel_event_routes_bot_alias_mentions(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_BOT_ALIASES", "Hermes")
    monkeypatch.setenv("TLON_ALLOW_ALL_USERS", "true")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "alias-post",
                "r-post": {
                    "set": {
                        "seal": {"id": "alias-post"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [{"inline": ["Hermes: hello"]}],
                        },
                    }
                },
            }
        },
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"


@pytest.mark.asyncio
async def test_channel_event_routes_owner_without_mention(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    monkeypatch.setenv("TLON_OWNER_LISTEN_ENABLED", "true")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "owner-listen-post",
                "r-post": {
                    "set": {
                        "seal": {"id": "owner-listen-post"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [{"inline": ["hello without mention"]}],
                        },
                    }
                },
            }
        },
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello without mention"


@pytest.mark.asyncio
async def test_channel_event_retries_delayed_top_level_blob(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    monkeypatch.setattr("gateway.platforms.tlon.asyncio.sleep", AsyncMock())
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()
    adapter._fetch_post_blob = AsyncMock(
        return_value='[{"type":"file","fileUri":"https://example.com/a.pdf","name":"a.pdf","mimeType":"application/pdf"}]'
    )

    async def fake_prepare(*, story_content, blob, text):
        if blob:
            return (
                '[file: a.pdf (application/pdf, unknown size)] https://example.com/a.pdf',
                ["/tmp/a.pdf"],
                ["application/pdf"],
                MessageType.DOCUMENT,
            )
        return text, [], [], MessageType.TEXT

    adapter._prepare_media_context = fake_prepare

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "170141184507864167403996323545639550976",
                "r-post": {
                    "set": {
                        "seal": {"id": "170141184507864167403996323545639550976"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [],
                        },
                    }
                },
            }
        },
    })

    adapter._fetch_post_blob.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.DOCUMENT
    assert event.media_urls == ["/tmp/a.pdf"]
    assert "a.pdf" in event.text


@pytest.mark.asyncio
async def test_channel_event_fetches_thread_reply_blob(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()
    adapter._fetch_reply_blob = AsyncMock(
        return_value='[{"type":"file","fileUri":"https://example.com/thread.pdf","name":"thread.pdf","mimeType":"application/pdf"}]'
    )

    async def fake_prepare(*, story_content, blob, text):
        if blob:
            return (
                '[file: thread.pdf (application/pdf, unknown size)] https://example.com/thread.pdf',
                ["/tmp/thread.pdf"],
                ["application/pdf"],
                MessageType.DOCUMENT,
            )
        return text, [], [], MessageType.TEXT

    adapter._prepare_media_context = fake_prepare

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "parent-post",
                "r-post": {
                    "reply": {
                        "id": "reply-post",
                        "r-reply": {
                            "set": {
                                "seal": {"parent-id": "parent-post"},
                                "memo": {
                                    "author": "~zod",
                                    "sent": 1_700_000_000_000,
                                    "content": [],
                                },
                            }
                        },
                    }
                },
            }
        },
    })

    adapter._fetch_reply_blob.assert_awaited_once_with(
        "chat/~host/test",
        "parent-post",
        "reply-post",
    )
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "parent-post"
    assert event.media_urls == ["/tmp/thread.pdf"]


@pytest.mark.asyncio
async def test_channel_event_ignores_owner_when_owner_listen_disabled_for_channel(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    monkeypatch.setenv("TLON_OWNER_LISTEN_ENABLED", "true")
    monkeypatch.setenv("TLON_OWNER_LISTEN_DISABLED_CHANNELS", "chat/~host/test")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "owner-listen-disabled-post",
                "r-post": {
                    "set": {
                        "seal": {"id": "owner-listen-disabled-post"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [{"inline": ["hello without mention"]}],
                        },
                    }
                },
            }
        },
    })

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_channel_event_routes_thread_reply_to_parent(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_ALLOW_ALL_USERS", "true")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "parent-post",
                "r-post": {
                    "reply": {
                        "id": "reply-post",
                        "r-reply": {
                            "set": {
                                "seal": {"parent-id": "parent-post"},
                                "memo": {
                                    "author": "~zod",
                                    "sent": 1_700_000_000_000,
                                    "content": [
                                        {"inline": ["hey ", {"ship": "~bot-palnet"}]}
                                    ],
                                },
                            }
                        },
                    }
                },
            }
        },
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hey"
    assert event.message_id == "reply-post"
    assert event.reply_to_message_id == "parent-post"
    assert event.source.thread_id == "parent-post"


@pytest.mark.asyncio
async def test_channel_event_routes_openclaw_thread_reply_essay(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_ALLOW_ALL_USERS", "true")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "parent-post",
                "r-post": {
                    "reply": {
                        "id": "reply-post",
                        "r-reply": {
                            "set": {
                                "seal": {"parent-id": "parent-post"},
                                "reply-essay": {
                                    "author": "~zod",
                                    "sent": 1_700_000_000_000,
                                    "content": [
                                        {"inline": ["got it ", {"ship": "~bot-palnet"}]}
                                    ],
                                },
                            }
                        },
                    }
                },
            }
        },
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "got it"
    assert event.message_id == "reply-post"
    assert event.reply_to_message_id == "parent-post"
    assert event.source.thread_id == "parent-post"


@pytest.mark.asyncio
async def test_channel_event_refreshes_group_mapping_for_context(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("TLON_AUTO_DISCOVER", "true")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter._sse = AsyncMock()
    adapter._sse.scry.return_value = {
        "groups": {
            "~host/group": {
                "meta": {"title": "Test Group"},
                "channels": {"chat/~host/test": {}},
            }
        }
    }
    adapter.handle_message = AsyncMock()

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "post-id",
                "r-post": {
                    "set": {
                        "seal": {"id": "post-id"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [
                                {"inline": [{"ship": "~bot-palnet"}, " hello"]}
                            ],
                        },
                    }
                },
            }
        },
    })

    event = adapter.handle_message.await_args.args[0]
    assert event.source.parent_chat_id == "~host/group"
    assert event.source.chat_name == "Test Group / test"


@pytest.mark.asyncio
async def test_channel_event_routes_blob_only_owner_message(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    adapter = TlonAdapter(PlatformConfig())
    adapter.monitored_channels = {"chat/~host/test"}
    adapter.handle_message = AsyncMock()

    async def fake_download_blob_attachments(entries):
        return [TlonDownloadedAttachment("/tmp/report.pdf", "application/pdf")], []

    monkeypatch.setattr(
        "gateway.platforms.tlon.download_blob_attachments",
        fake_download_blob_attachments,
    )

    await adapter._handle_channel_event({
        "nest": "chat/~host/test",
        "response": {
            "post": {
                "id": "blob-post",
                "r-post": {
                    "set": {
                        "seal": {"id": "blob-post"},
                        "essay": {
                            "author": "~zod",
                            "sent": 1_700_000_000_000,
                            "content": [],
                            "blob": '[{"type":"file","fileUri":"https://example.com/report.pdf","mimeType":"application/pdf","name":"report.pdf","size":2048}]',
                        },
                    }
                },
            }
        },
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "[file: report.pdf" in event.text
    assert event.media_urls == ["/tmp/report.pdf"]
    assert event.media_types == ["application/pdf"]
    assert event.message_type.value == "document"


@pytest.mark.asyncio
async def test_dm_event_uses_partner_for_routing_and_skips_own_messages(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.delenv("TLON_OWNER_SHIP", raising=False)
    monkeypatch.delenv("TLON_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("TLON_DM_ALLOWLIST", raising=False)
    monkeypatch.delenv("TLON_DEFAULT_AUTHORIZED_SHIPS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = TlonAdapter(PlatformConfig())
    adapter.handle_message = AsyncMock()

    await adapter._handle_dm_event({
        "whom": "~zod",
        "id": "~zod/170.141",
        "response": {
            "add": {
                "essay": {
                    "author": "~zod",
                    "sent": 1_700_000_000_000,
                    "content": [{"inline": ["hello"]}],
                }
            }
        },
    })

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.source.chat_id == "~zod"
    assert event.source.chat_type == "dm"

    adapter.handle_message.reset_mock()
    await adapter._handle_dm_event({
        "whom": "~zod",
        "id": "~bot-palnet/170.142",
        "response": {
            "add": {
                "essay": {
                    "author": "~bot-palnet",
                    "sent": 1_700_000_000_001,
                    "content": [{"inline": ["own message"]}],
                }
            }
        },
    })

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_invite_list_accepts_string_and_object_entries(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    monkeypatch.setenv("TLON_ALLOWED_USERS", "~nec")
    adapter = TlonAdapter(PlatformConfig())
    adapter._sse = AsyncMock()

    await adapter._handle_dm_event(["~zod", {"ship": "~nec"}, 42, {"other": "~bud"}])

    assert adapter._sse.poke.await_count == 2
    ships = [call.kwargs["json_data"]["ship"] for call in adapter._sse.poke.await_args_list]
    assert ships == ["zod", "nec"]


@pytest.mark.asyncio
async def test_unauthorized_dm_queues_owner_approval(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~ten")
    adapter = TlonAdapter(PlatformConfig())
    adapter.handle_message = AsyncMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="~bot-palnet/170.141"))

    await adapter._handle_dm_event({
        "whom": "~zod",
        "id": "~zod/170.141",
        "response": {
            "add": {
                "essay": {
                    "author": "~zod",
                    "sent": 1_700_000_000_000,
                    "content": [{"inline": ["hello"]}],
                }
            }
        },
    })

    adapter.handle_message.assert_not_awaited()
    assert len(adapter.pending_approvals) == 1
    assert adapter.pending_approvals[0].requesting_ship == "~zod"
    assert adapter.pending_approvals[0].notification_message_id == "170141"
    adapter.send.assert_awaited_once()


def test_tlon_approval_request_mentions_reactions():
    approval = create_pending_approval(
        approval_type="dm",
        requesting_ship="~zod",
        existing_ids=[],
        message_preview="hello",
    )

    text = format_approval_request(approval)

    assert "React to this message: 👍 approve · 👎 deny · 🛑 block" in text


@pytest.mark.asyncio
async def test_owner_reaction_approves_pending_tlon_dm(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~ten")
    adapter = TlonAdapter(PlatformConfig())
    adapter._put_settings_entry = AsyncMock()
    adapter._dispatch_pending_message = AsyncMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="~bot-palnet/2"))
    approval = create_pending_approval(
        approval_type="dm",
        requesting_ship="~zod",
        existing_ids=[],
    )
    approval.notification_message_id = normalize_notification_id("~bot-palnet/170.141")
    adapter.pending_approvals = [approval]

    await adapter._handle_dm_event({
        "whom": "~ten",
        "id": "~bot-palnet/170.141",
        "response": {
            "add-react": {
                "author": "~ten",
                "react": "👍",
            }
        },
    })

    assert "~zod" in adapter.dm_allowlist
    assert adapter.pending_approvals == []
    adapter._dispatch_pending_message.assert_awaited_once_with(approval)
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_exec_approval_registers_owner_dm_reaction_prompt(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~ten")
    adapter = TlonAdapter(PlatformConfig())
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="~bot-palnet/170.141"))

    result = await adapter.send_exec_approval(
        "chat/~host/general",
        "rm -rf /tmp/example",
        "tlon:chat/~host/general:~ten",
        "dangerous command",
    )

    assert result.success
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[0] == "~ten"
    normalized = normalize_notification_id("~bot-palnet/170.141")
    assert adapter._exec_approval_prompts[normalized]["session_key"] == "tlon:chat/~host/general:~ten"


@pytest.mark.asyncio
async def test_owner_reaction_resolves_exec_approval(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~ten")
    adapter = TlonAdapter(PlatformConfig())
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="~bot-palnet/2"))
    normalized = normalize_notification_id("~bot-palnet/170.141")
    adapter._exec_approval_prompts[normalized] = {
        "session_key": "tlon:~ten:~ten",
        "chat_id": "~ten",
    }
    adapter._exec_approval_prompt_by_session["tlon:~ten:~ten"] = normalized
    calls = []

    def fake_resolve(session_key, choice, resolve_all=False):
        calls.append((session_key, choice, resolve_all))
        return 1

    monkeypatch.setattr("tools.approval.resolve_gateway_approval", fake_resolve)

    await adapter._handle_dm_event({
        "whom": "~ten",
        "id": "~bot-palnet/170.141",
        "response": {
            "add-react": {
                "author": "~ten",
                "react": "🛑",
            }
        },
    })

    assert calls == [("tlon:~ten:~ten", "deny", False)]
    assert normalized not in adapter._exec_approval_prompts
    assert "tlon:~ten:~ten" not in adapter._exec_approval_prompt_by_session


def _dm_history_post(author: str, sent: int, text: str, post_id: str):
    return {
        "seal": {"id": post_id},
        "essay": {
            "author": author,
            "sent": sent,
            "kind": "/chat",
            "blob": None,
            "content": [{"inline": [text]}],
            "meta": None,
        },
        "type": "post",
    }


@pytest.mark.asyncio
async def test_dm_history_initial_catchup_only_routes_newest_unanswered(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    adapter = TlonAdapter(PlatformConfig())
    adapter.handle_message = AsyncMock()
    adapter.dm_poll_initial_catchup_seconds = 600

    now_ms = 1_700_001_000_000
    monkeypatch.setattr("gateway.platforms.tlon.time.time", lambda: now_ms / 1000)

    await adapter._initialize_dm_history(
        "~zod",
        [
            _dm_history_post("~bot-palnet", now_ms - 500_000, "old reply", "~bot-palnet/1"),
            _dm_history_post("~zod", now_ms - 300_000, "make a group", "~zod/2"),
            _dm_history_post("~zod", now_ms - 100_000, "hello", "~zod/3"),
            _dm_history_post(
                "~bot-palnet",
                now_ms - 50_000,
                "Gateway shutting down - Your current task will be interrupted.",
                "~bot-palnet/status",
            ),
        ],
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.message_id == "~zod/3"
    assert "~zod/2" in adapter._processed_ids


@pytest.mark.asyncio
async def test_dm_history_poll_routes_new_unprocessed_messages(monkeypatch):
    monkeypatch.setenv("TLON_SHIP_NAME", "~bot-palnet")
    monkeypatch.setenv("TLON_OWNER_SHIP", "~zod")
    adapter = TlonAdapter(PlatformConfig())
    adapter.handle_message = AsyncMock()
    adapter._sse = AsyncMock()
    adapter._sse.scry.return_value = {
        "writs": {
            "1": _dm_history_post("~zod", 1_700_000_000_000, "one", "~zod/1"),
            "2": _dm_history_post("~zod", 1_700_000_001_000, "two", "~zod/2"),
        }
    }
    adapter._dm_poll_initialized.add("~zod")

    await adapter._poll_dm_history("~zod")

    assert adapter.handle_message.await_count == 2
    assert [call.args[0].text for call in adapter.handle_message.await_args_list] == [
        "one",
        "two",
    ]
