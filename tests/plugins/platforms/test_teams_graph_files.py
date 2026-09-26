"""Unit tests for Teams Graph/SharePoint channel file upload helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.teams.graph_files import (
    GRAPH_CHANNEL_FILE_PERMISSION,
    GRAPH_CHANNEL_MESSAGE_PERMISSION,
    GRAPH_CHANNEL_MESSAGE_RSC,
    SIMPLE_UPLOAD_MAX_BYTES,
    GraphFileTarget,
    GraphFileUploadError,
    encode_graph_share_id,
    extract_graph_file_target,
    extract_graph_message_file_refs,
    graph_message_path,
    list_graph_message_file_refs,
    markdown_file_link,
    resolve_graph_credentials,
    resolve_graph_file_target,
    resolve_inbound_file_download_url,
    safe_graph_filename,
    upload_conversation_file,
)
from tools.microsoft_graph_auth import GraphCredentials
from tools.microsoft_graph_client import MicrosoftGraphAPIError


class _FakeGraph:
    def __init__(
        self,
        *,
        folder=None,
        item=None,
        link=None,
        session=None,
        get_error=None,
        put_error=None,
    ):
        self.folder = folder or {
            "id": "folder-1",
            "parentReference": {"driveId": "drive-1"},
            "webUrl": "https://contoso.sharepoint.com/sites/team/Shared Documents/General",
        }
        self.item = item or {
            "id": "item-1",
            "name": "report.pdf",
            "webUrl": "https://contoso.sharepoint.com/sites/team/Shared%20Documents/General/report.pdf",
            "size": 12,
        }
        self.link = link if link is not None else {
            "link": {"webUrl": "https://contoso.sharepoint.com/:b:/s/team/share-token"},
        }
        self.session = session or {
            "uploadUrl": "https://contoso.sharepoint.com/upload-session",
        }
        self.get_error = get_error
        self.put_error = put_error
        self.gets: list[str] = []
        self.puts: list[tuple] = []
        self.posts: list[tuple] = []

    async def get_json(self, path, **kwargs):
        self.gets.append(path)
        if self.get_error is not None:
            raise self.get_error
        return self.folder

    async def put_bytes(self, path, *, content, content_type=None, params=None, **kwargs):
        self.puts.append((path, content, content_type, params))
        if self.put_error is not None:
            raise self.put_error
        return self.item

    async def post_json(self, path, *, json_body=None, **kwargs):
        self.posts.append((path, json_body))
        if "createUploadSession" in path:
            return self.session
        if "createLink" in path:
            return self.link
        return {}


class TestExtractAndResolveTarget:
    def test_extracts_channel_aad_group_id_from_channel_data_dict(self):
        activity = SimpleNamespace(
            conversation=SimpleNamespace(
                id="19:chan@thread.tacv2", conversation_type="channel"),
            channel_data={
                "team": {"id": "19:team@thread.skype", "aadGroupId": "team-guid"},
                "channel": {"id": "19:chan@thread.tacv2"},
                "teamsChannelId": "19:chan@thread.tacv2",
            },
        )
        target = extract_graph_file_target(activity)
        assert target == GraphFileTarget(
            conversation_type="channel",
            team_id="team-guid",
            channel_id="19:chan@thread.tacv2",
        )

    def test_extracts_group_chat_id(self):
        activity = SimpleNamespace(
            conversation=SimpleNamespace(
                id="19:chat@thread.v2", conversation_type="groupChat"),
            channel_data={},
        )
        target = extract_graph_file_target(activity)
        assert target == GraphFileTarget(
            conversation_type="groupChat", chat_id="19:chat@thread.v2")

    def test_ignores_magicmock_channel_data(self):
        activity = MagicMock()
        activity.conversation = MagicMock(id="19:abc@thread.v2", conversation_type="personal")
        assert extract_graph_file_target(activity) is None

    def test_personal_chat_is_not_a_graph_target(self):
        activity = SimpleNamespace(
            conversation=SimpleNamespace(id="a:personal", conversation_type="personal"),
            channel_data={},
        )
        assert extract_graph_file_target(activity) is None

    def test_resolve_uses_extra_team_id_when_stash_missing(self):
        target = resolve_graph_file_target(
            "19:chan@thread.tacv2",
            conv_type="channel",
            extra_team_id="team-guid",
        )
        assert target is not None
        assert target.team_id == "team-guid"
        assert target.channel_id == "19:chan@thread.tacv2"

    def test_resolve_group_chat_from_conversation_type_only(self):
        target = resolve_graph_file_target(
            "19:chat@thread.v2", conv_type="groupChat")
        assert target == GraphFileTarget(
            conversation_type="groupChat", chat_id="19:chat@thread.v2")

    def test_resolve_channel_without_team_id_is_none(self):
        assert resolve_graph_file_target(
            "19:chan@thread.tacv2", conv_type="channel") is None


class TestCredentialsAndHelpers:
    def test_prefers_msgraph_env(self):
        creds = resolve_graph_credentials(
            {
                "MSGRAPH_TENANT_ID": "g-tenant",
                "MSGRAPH_CLIENT_ID": "g-client",
                "MSGRAPH_CLIENT_SECRET": "g-secret",
            },
            teams_tenant_id="t-tenant",
            teams_client_id="t-client",
            teams_client_secret="t-secret",
        )
        assert creds == GraphCredentials("g-tenant", "g-client", "g-secret")

    def test_falls_back_to_teams_bot_credentials(self):
        creds = resolve_graph_credentials(
            {},
            teams_tenant_id="t-tenant",
            teams_client_id="t-client",
            teams_client_secret="t-secret",
        )
        assert creds == GraphCredentials("t-tenant", "t-client", "t-secret")

    def test_missing_both_returns_none(self):
        assert resolve_graph_credentials({}) is None

    def test_safe_filename_strips_path_and_reserved_chars(self):
        assert safe_graph_filename(r"..\\foo/bar:<baz>.pdf") == "bar__baz_.pdf"

    def test_markdown_link_neutralizes_brackets(self):
        text = markdown_file_link("evil](http://x)", "https://sp.example/file")
        assert "](http://x)" not in text
        assert text.startswith("[")
        assert "https://sp.example/file" in text

    def test_files_folder_path_quotes_channel_id(self):
        path = GraphFileTarget(
            conversation_type="channel",
            team_id="team-guid",
            channel_id="19:chan@thread.tacv2",
        ).files_folder_path()
        assert path.startswith("/teams/team-guid/channels/")
        assert "filesFolder" in path
        assert "@" not in path.split("/channels/")[1].split("/")[0]


class TestUploadConversationFile:
    @pytest.mark.asyncio
    async def test_small_file_simple_put_and_share_link(self):
        graph = _FakeGraph()
        uploaded = await upload_conversation_file(
            graph,
            GraphFileTarget(
                conversation_type="channel",
                team_id="team-guid",
                channel_id="19:chan@thread.tacv2",
            ),
            file_name="report.pdf",
            data=b"%PDF-hello",
            content_type="application/pdf",
        )
        assert uploaded.link.startswith("https://contoso.sharepoint.com/")
        assert uploaded.name == "report.pdf"
        assert graph.gets and "filesFolder" in graph.gets[0]
        assert graph.puts
        put_path, content, content_type, params = graph.puts[0]
        assert put_path.endswith("/content")
        assert content == b"%PDF-hello"
        assert content_type == "application/pdf"
        assert params["@microsoft.graph.conflictBehavior"] == "rename"
        assert any("createLink" in path for path, _ in graph.posts)

    @pytest.mark.asyncio
    async def test_share_link_failure_falls_back_to_web_url(self):
        graph = _FakeGraph()

        async def _fail_link(path, *, json_body=None, **kwargs):
            graph.posts.append((path, json_body))
            if "createLink" in path:
                raise MicrosoftGraphAPIError(403, "POST", path, "Access denied")
            return {}

        graph.post_json = _fail_link  # type: ignore[method-assign]
        uploaded = await upload_conversation_file(
            graph,
            GraphFileTarget(conversation_type="groupChat", chat_id="19:chat@thread.v2"),
            file_name="notes.bin",
            data=b"\x00\x01",
        )
        assert uploaded.share_url == ""
        assert uploaded.web_url.endswith("report.pdf")

    @pytest.mark.asyncio
    async def test_large_file_uses_upload_session(self):
        graph = _FakeGraph(item={
            "id": "item-big",
            "name": "big.bin",
            "webUrl": "https://contoso.sharepoint.com/big.bin",
            "size": SIMPLE_UPLOAD_MAX_BYTES + 1,
        })
        session_puts: list[tuple[str, bytes]] = []

        async def _put_session(url: str, data: bytes):
            session_puts.append((url, data))
            return graph.item

        data = b"x" * (SIMPLE_UPLOAD_MAX_BYTES + 1)
        uploaded = await upload_conversation_file(
            graph,
            GraphFileTarget(conversation_type="groupChat", chat_id="19:chat@thread.v2"),
            file_name="big.bin",
            data=data,
            put_upload_session=_put_session,
        )
        assert not graph.puts
        assert any("createUploadSession" in path for path, _ in graph.posts)
        assert session_puts == [("https://contoso.sharepoint.com/upload-session", data)]
        assert uploaded.item_id == "item-big"

    @pytest.mark.asyncio
    async def test_files_folder_403_propagates(self):
        graph = _FakeGraph(
            get_error=MicrosoftGraphAPIError(403, "GET", "/teams/x/channels/y/filesFolder", "denied"),
        )
        with pytest.raises(MicrosoftGraphAPIError) as exc:
            await upload_conversation_file(
                graph,
                GraphFileTarget(
                    conversation_type="channel", team_id="t", channel_id="c"),
                file_name="a.pdf",
                data=b"abc",
            )
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_web_url_raises(self):
        graph = _FakeGraph(item={"id": "item-1", "name": "a.bin"}, link={})
        with pytest.raises(GraphFileUploadError):
            await upload_conversation_file(
                graph,
                GraphFileTarget(conversation_type="groupChat", chat_id="19:c"),
                file_name="a.bin",
                data=b"abc",
            )

    def test_permission_constant_is_application_files_readwrite_all(self):
        assert GRAPH_CHANNEL_FILE_PERMISSION == "Files.ReadWrite.All"


class TestInboundGraphDownloadUrl:
    def test_encode_graph_share_id_is_u_bang_base64url(self):
        import base64

        url = "https://contoso.sharepoint.com/sites/team/Shared%20Documents/notes.txt"
        expected = "u!" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        assert encode_graph_share_id(url) == expected

    @pytest.mark.asyncio
    async def test_unique_id_uses_files_folder_then_item(self):
        download = "https://contoso.sharepoint.com/_layouts/download.aspx?x"
        graph = _FakeGraph()

        async def get_json(path, **kwargs):
            graph.gets.append(path)
            if "filesFolder" in path:
                return graph.folder
            return {
                "id": "item-1",
                "name": "notes.txt",
                "@microsoft.graph.downloadUrl": download,
            }

        graph.get_json = get_json  # type: ignore[method-assign]
        url = await resolve_inbound_file_download_url(
            graph,
            target=GraphFileTarget(
                conversation_type="channel",
                team_id="team-guid",
                channel_id="19:chan@thread.tacv2",
            ),
            content={"uniqueId": "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}"},
            filename="notes.txt",
        )
        assert url == download
        assert any("filesFolder" in path for path in graph.gets)
        assert any("/items/" in path for path in graph.gets)

    @pytest.mark.asyncio
    async def test_sharepoint_content_url_uses_shares_api(self):
        download = "https://contoso.sharepoint.com/_layouts/download.aspx?shared"
        sharing = "https://contoso.sharepoint.com/:t:/g/team/abc"
        graph = _FakeGraph()

        async def get_json(path, **kwargs):
            graph.gets.append(path)
            if "/shares/" in path:
                assert encode_graph_share_id(sharing).split("!")[1] in path or "u!" in path
                return {"@microsoft.graph.downloadUrl": download}
            raise AssertionError(f"unexpected Graph GET {path}")

        graph.get_json = get_json  # type: ignore[method-assign]
        url = await resolve_inbound_file_download_url(
            graph,
            target=None,
            content={},
            content_url=sharing,
        )
        assert url == download
        assert any("/shares/" in path for path in graph.gets)

    @pytest.mark.asyncio
    async def test_non_sharepoint_url_is_not_sent_to_shares_api(self):
        graph = _FakeGraph()
        graph.get_json = AsyncMock(side_effect=AssertionError("must not call Graph"))
        url = await resolve_inbound_file_download_url(
            graph,
            content_url="https://smba.trafficmanager.net/emea/v3/attachments/1",
        )
        assert url == ""


class TestGraphMessageFileRefs:
    def test_channel_message_path_quotes_ids(self):
        path = graph_message_path(
            GraphFileTarget(
                conversation_type="channel",
                team_id="team-guid",
                channel_id="19:chan@thread.tacv2",
            ),
            "12345",
        )
        assert path.startswith("/teams/team-guid/channels/")
        assert path.endswith("/messages/12345")
        assert "@" not in path.split("/channels/")[1].split("/")[0]

    def test_group_chat_message_path(self):
        path = graph_message_path(
            GraphFileTarget(conversation_type="groupChat", chat_id="19:chat@thread.v2"),
            "99",
        )
        assert path == "/chats/19%3Achat%40thread.v2/messages/99"

    def test_extracts_reference_attachments_and_skips_html(self):
        refs = extract_graph_message_file_refs({
            "attachments": [
                {"contentType": "text/html", "content": "<p>hi</p>"},
                {
                    "contentType": "reference",
                    "contentUrl": "https://contoso.sharepoint.com/sites/t/notes.txt",
                    "name": "notes.txt",
                    "uniqueId": "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}",
                },
                {"contentType": "application/vnd.microsoft.card.adaptive", "name": "card"},
            ],
        })
        assert len(refs) == 1
        assert refs[0]["name"] == "notes.txt"
        assert refs[0]["contentUrl"].endswith("notes.txt")
        assert refs[0]["uniqueId"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_message_read_permission_constants(self):
        assert GRAPH_CHANNEL_MESSAGE_PERMISSION == "ChannelMessage.Read.All"
        assert GRAPH_CHANNEL_MESSAGE_RSC == "ChannelMessage.Read.Group"
        assert GRAPH_CHANNEL_FILE_PERMISSION == "Files.ReadWrite.All"

    @pytest.mark.asyncio
    async def test_list_graph_message_file_refs_gets_channel_message(self):
        graph = _FakeGraph()

        async def get_json(path, **kwargs):
            graph.gets.append(path)
            return {
                "attachments": [{
                    "contentType": "reference",
                    "contentUrl": "https://contoso.sharepoint.com/sites/t/a.pdf",
                    "name": "a.pdf",
                }],
            }

        graph.get_json = get_json  # type: ignore[method-assign]
        refs = await list_graph_message_file_refs(
            graph,
            GraphFileTarget(
                conversation_type="channel", team_id="t", channel_id="c"),
            "msg-1",
        )
        assert refs[0]["name"] == "a.pdf"
        assert graph.gets == ["/teams/t/channels/c/messages/msg-1"]
