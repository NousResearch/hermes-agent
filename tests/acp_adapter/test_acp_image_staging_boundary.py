"""prompt.submit attachments and ACP image staging stay inside the profile image cache (PR #106742 security lane)."""
import base64
import os
from pathlib import Path

import pytest

from hermes_state_runtime import RuntimeStoreError

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def test_hardlink_in_staging_to_outside_file_is_refused(tmp_path, monkeypatch):
    from gateway.platforms import base
    from gateway.session_ingress_media import admit_attachments
    staging = tmp_path / "home" / "cache" / "images"
    staging.mkdir(parents=True)
    monkeypatch.setattr(base, "get_image_cache_dir", lambda: staging)
    monkeypatch.setattr(base, "get_document_cache_dir", lambda: tmp_path / "home" / "cache" / "documents")
    secret = tmp_path / "id_rsa"
    secret.write_bytes(PNG + b"PRIVATE")
    (staging / "own.png").write_bytes(PNG)
    assert admit_attachments([{"path": str(staging / "own.png"), "mime": "image/png"}])  # control
    os.link(secret, staging / "planted.png")
    with pytest.raises(RuntimeStoreError):
        admit_attachments([{"path": str(staging / "planted.png"), "mime": "image/png"}])


@pytest.mark.parametrize("mime", ["image/png/../../../../escape", "image/svg+xml", "text/html", "image/gif"])
def test_acp_image_mime_never_chooses_the_staged_extension(tmp_path, monkeypatch, mime):
    """The extension comes from the sniffed bytes (png/jpeg/gif/webp), never the client's text;
    a declared type that disagrees with the bytes (PNG bytes as image/gif) is refused too."""
    from acp.schema import ImageContentBlock
    from acp_adapter.content import _content_blocks_to_openai_user_content
    from acp_adapter.gateway_server import _stage_user_content
    from gateway.platforms import base
    from hermes_cli.gateway_client import GatewayClientError
    staging = tmp_path / "home" / "cache" / "images"
    staging.mkdir(parents=True)
    monkeypatch.setattr(base, "get_image_cache_dir", lambda: staging)
    block = ImageContentBlock(type="image", data=base64.b64encode(PNG).decode(), mime_type=mime)
    with pytest.raises(GatewayClientError):
        _stage_user_content(_content_blocks_to_openai_user_content([block]))
    assert list(staging.iterdir()) == []
