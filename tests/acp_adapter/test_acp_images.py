import base64

import pytest
from acp.schema import (
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
)

from acp_adapter.server import HermesACPAgent, _content_blocks_to_openai_user_content

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
    attached.write_text("# Notes\n\nAttached file body", encoding="utf-8", newline="\n")

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

@pytest.mark.platforms("windows")
def test_native_drive_path_and_file_uri_refer_to_same_attachment(tmp_path):
    from acp_adapter.content import _path_from_file_uri
    path = tmp_path / "notes with spaces.md"
    path.write_text("body", encoding="utf-8")
    assert _path_from_file_uri(str(path)) == _path_from_file_uri(path.as_uri()) == path

def test_truncated_utf8_attachment_is_inlined_as_a_prefix_of_its_text(tmp_path):
    """The 512 KiB cut lands inside a 3-byte character; the inlined body must still be the
    file's own text (a prefix of it), not the whole cut decoded as latin-1."""
    from acp_adapter.content import _MAX_ACP_RESOURCE_BYTES

    text = "你好世界" * (_MAX_ACP_RESOURCE_BYTES // 6)  # ~2x the cap; the cap is not a multiple of 3
    raw = text.encode("utf-8")
    assert len(raw) > _MAX_ACP_RESOURCE_BYTES and _MAX_ACP_RESOURCE_BYTES % 3
    attached = tmp_path / "notes_zh.md"
    attached.write_bytes(raw)

    linked = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="summarize"),
        ResourceContentBlock(type="resource_link", name=attached.name, uri=attached.as_uri()),
    ])
    embedded = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="summarize"),
        EmbeddedResourceContentBlock(type="resource", resource=BlobResourceContents(
            uri=attached.as_uri(), blob=base64.b64encode(raw).decode("ascii"), mime_type="text/markdown")),
    ])

    for content in (linked, embedded):
        body = content.split(f"URI: {attached.as_uri()}\n\n", 1)[1].split("\n\n[Truncated", 1)[0]
        assert text.startswith(body)
        # Only the split character is dropped: everything up to the cut survives.
        assert len(body.encode("utf-8")) > _MAX_ACP_RESOURCE_BYTES - 3

@pytest.mark.asyncio
async def test_initialize_advertises_image_prompt_capability():
    response = await HermesACPAgent().initialize()

    assert response.agent_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities.image is True
