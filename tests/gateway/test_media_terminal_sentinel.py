"""A leaked terminal ``<|eos|>`` sentinel must not hide a MEDIA attachment (#111046)."""

import pytest

from gateway.platforms.base import BasePlatformAdapter


@pytest.mark.parametrize("sentinel", ["<|eos|>", "<|eos|><|eos|>"])
@pytest.mark.parametrize("filename", ["chart.png", "payload.weirdext", "Caddyfile"])
def test_terminal_eos_sentinel_leaves_extraction_and_cleanup_unchanged(tmp_path, monkeypatch, filename, sentinel):
    """Known-extension, unknown-extension and extension-less tags glued to a (possibly repeated)
    ``<|eos|>`` extract and clean exactly like the same response without the sentinel."""
    root = tmp_path / "media-cache"
    root.mkdir()
    media_file = root / filename
    media_file.write_bytes(b"media")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    clean = f"Here is the file.\nMEDIA:{media_file}"

    assert BasePlatformAdapter.extract_media(clean + sentinel) == BasePlatformAdapter.extract_media(clean)
    assert BasePlatformAdapter.strip_media_directives_for_display(clean + sentinel) == "Here is the file."


@pytest.mark.parametrize(
    "text",
    [
        "MEDIA:/tmp/example.png<|EOS|>",  # not the exact token: still not a path delimiter
        "MEDIA:/tmp/example.png<|eos|> trailing prose",  # not terminal
        "```text\nMEDIA:/tmp/example.png<|eos|>\n```",  # protected code stays byte-identical
        '{"example":"MEDIA:/tmp/example.png<|eos|>"}',  # protected JSON value
    ],
)
def test_non_terminal_or_protected_sentinel_is_not_a_media_boundary(text):
    assert BasePlatformAdapter.extract_media(text) == ([], text)
    assert BasePlatformAdapter.strip_media_directives_for_display(text) == text
