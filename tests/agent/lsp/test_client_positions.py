"""Position encoding contracts used by incremental document synchronization."""
import pytest

from agent.lsp.client import LSPClient, _end_position


@pytest.mark.parametrize(("text", "expected"), [
    ("", {"line": 0, "character": 0}),
    ("ascii", {"line": 0, "character": 5}),
    ("a😀", {"line": 0, "character": 3}),
    ("a\r\n😀\n", {"line": 2, "character": 0}),
    ("a\rb", {"line": 1, "character": 1}),
    ("a\u2028b", {"line": 0, "character": 3}),
])
def test_end_position_uses_lsp_lines_and_utf16_units(text, expected):
    assert _end_position(text) == expected


@pytest.mark.asyncio
async def test_document_sync_precedes_disk_change_notification(tmp_path, monkeypatch):
    path = tmp_path / "document.qmd"
    path.write_text("old 😀 content", encoding="utf-8")
    client = LSPClient(server_id="test", workspace_root=str(tmp_path), command=["unused"])
    client._sync_kind = 2
    sent = []

    async def capture(method, params):
        sent.append((method, params))

    monkeypatch.setattr(LSPClient, "is_running", property(lambda self: True))
    monkeypatch.setattr(client, "_send_notification", capture)
    await client.open_file(str(path), language_id="quarto")
    path.write_text("new content", encoding="utf-8")
    await client.open_file(str(path), language_id="quarto")

    assert [method for method, _ in sent] == [
        "textDocument/didOpen", "workspace/didChangeWatchedFiles",
        "textDocument/didChange", "workspace/didChangeWatchedFiles",
    ]
    change = sent[2][1]["contentChanges"][0]
    assert change["range"]["end"] == {"line": 0, "character": 14}
    assert change["text"] == "new content"
