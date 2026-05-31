import base64

import pytest
from acp.schema import (
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
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
    attached.write_text("# Notes\n\nAttached file body", encoding="utf-8")

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


def test_acp_embedded_text_resource_is_inlined_as_text():
    content = _content_blocks_to_openai_user_content([
        EmbeddedResourceContentBlock(
            type="resource",
            resource=TextResourceContents(
                uri="file:///workspace/todo.txt",
                mimeType="text/plain",
                text="first\nsecond",
            ),
        ),
    ])

    assert content == (
        "[Attached file: todo.txt]\n"
        "URI: file:///workspace/todo.txt\n\n"
        "first\nsecond"
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


def test_acp_resource_link_image_file_is_inlined_as_image_url(tmp_path):
    attached = tmp_path / "shot.png"
    attached.write_bytes(_ONE_PX_PNG)

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="Look at this screenshot"),
        ResourceContentBlock(
            type="resource_link",
            name="shot.png",
            uri=attached.as_uri(),
            mimeType="image/png",
        ),
    ])

    assert isinstance(content, list)
    # [user text, image header, image_url]
    assert content[0] == {"type": "text", "text": "Look at this screenshot"}
    assert content[1]["type"] == "text"
    assert "[Attached image: shot.png]" in content[1]["text"]
    assert content[2]["type"] == "image_url"
    expected_url = "data:image/png;base64," + base64.b64encode(_ONE_PX_PNG).decode("ascii")
    assert content[2]["image_url"]["url"] == expected_url


def test_acp_resource_link_image_mime_inferred_from_suffix(tmp_path):
    """No mimeType sent — should still be recognised as image by file suffix."""
    attached = tmp_path / "pic.jpg"
    attached.write_bytes(_ONE_PX_PNG)  # content doesn't matter for the code path

    content = _content_blocks_to_openai_user_content([
        ResourceContentBlock(
            type="resource_link",
            name="pic.jpg",
            uri=attached.as_uri(),
        ),
    ])

    assert isinstance(content, list)
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_acp_embedded_blob_image_is_inlined_as_image_url():
    b64 = base64.b64encode(_ONE_PX_PNG).decode("ascii")
    content = _content_blocks_to_openai_user_content([
        EmbeddedResourceContentBlock(
            type="resource",
            resource=BlobResourceContents(
                uri="file:///tmp/embed.png",
                mimeType="image/png",
                blob=b64,
            ),
        ),
    ])

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "[Attached image: embed.png]" in content[0]["text"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }
