import base64
from pathlib import Path

import pytest
from acp.schema import (
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)

from acp_adapter import content as acp_content
from acp_adapter.server import HermesACPAgent, _content_blocks_to_openai_user_content


def test_windows_file_uri_keeps_drive_path_outside_wsl(monkeypatch):
    monkeypatch.setattr(acp_content, "is_wsl", lambda: False)

    assert acp_content._path_from_file_uri("file:///C:/Users/alice/notes.md") == Path("C:/Users/alice/notes.md")
    assert acp_content._path_from_file_uri(r"C:\Users\alice\notes.md") == Path("C:/Users/alice/notes.md")
    assert acp_content._path_from_file_uri("C:relative.md") is None
    assert acp_content._path_from_file_uri("C:%5Crelative.md") is None
    assert acp_content._path_from_file_uri("file:///C:relative.md") is None
    assert acp_content._path_from_file_uri("file:///C%3Arelative.md") is None


def test_windows_file_uri_uses_mount_path_inside_wsl(monkeypatch):
    monkeypatch.setattr(acp_content, "is_wsl", lambda: True)

    assert acp_content._path_from_file_uri("file:///C:/Users/alice/notes.md") == Path("/mnt/c/Users/alice/notes.md")
    assert acp_content._path_from_file_uri(r"C:\Users\alice\notes.md") == Path("/mnt/c/Users/alice/notes.md")


@pytest.mark.windows_only
def test_acp_resource_link_inlines_native_windows_file_uri(tmp_path):
    attached = tmp_path / "native-windows.md"
    attached.write_text("Native Windows ACP resource", encoding="utf-8")

    content = _content_blocks_to_openai_user_content([
        ResourceContentBlock(
            type="resource_link",
            name=attached.name,
            uri=attached.as_uri(),
            mimeType="text/markdown",
        ),
    ])

    assert "Native Windows ACP resource" in content


def test_acp_image_blocks_convert_to_openai_multimodal_content():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="What is in this image?"),
        ImageContentBlock(type="image", data="aGVsbG8=", mimeType="image/png"),
    ])

    assert content == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        },
    ]


def test_text_only_acp_blocks_stay_string_for_legacy_prompt_path():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="/help"),
    ])

    assert content == "/help"


def test_acp_resource_link_file_is_inlined_as_text(tmp_path):
    attached = tmp_path / "notes.md"
    attached.write_bytes(b"# Notes\n\nAttached file body")

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="Please read this file"),
        ResourceContentBlock(
            type="resource_link",
            name="notes.md",
            title="Project notes",
            uri=attached.as_uri(),
            mimeType="text/markdown",
        ),
    ])

    assert content == (
        "Please read this file\n"
        "[Attached file: Project notes (notes.md)]\n"
        f"URI: {attached.as_uri()}\n\n"
        "# Notes\n\nAttached file body"
    )




@pytest.mark.asyncio
async def test_initialize_advertises_image_prompt_capability():
    response = await HermesACPAgent().initialize()

    assert response.agent_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities.image is True


# 1x1 transparent PNG — smallest valid image payload for inlining tests.
_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


