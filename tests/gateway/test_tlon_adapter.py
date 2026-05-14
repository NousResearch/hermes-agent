from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.tlon import (
    TlonAdapter,
    _extract_message_text,
    _text_to_story,
)
from gateway.platforms.tlon_approval import create_pending_approval, format_pending_list
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
    adapter.send = AsyncMock()

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
    adapter.send.assert_awaited_once()
