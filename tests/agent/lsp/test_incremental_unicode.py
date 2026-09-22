"""The server's document mirror must match disk after each incremental replacement."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient, file_uri


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original",
    [
        'const face = "😀";',
        '// header\nconst letter = "𐐀";',
        'const word = "中文";',
        'const face = "😀";\n',
        "",
    ],
)
async def test_incremental_replacements_preserve_server_document(tmp_path, original):
    path = tmp_path / "sample.ts"
    client = LSPClient(
        server_id="incremental",
        workspace_root=str(tmp_path),
        command=[sys.executable, str(Path(__file__).with_name("_mock_lsp_server.py"))],
        env={"MOCK_LSP_SCRIPT": "incremental"},
    )
    await client.start()
    try:
        for text in (original, 'const next = "🚀";', "const done = true;"):
            path.write_text(text, encoding="utf-8")
            await client.open_file(str(path), language_id="typescript")
            hover = await asyncio.wait_for(
                client._send_request(
                    "textDocument/hover",
                    {
                        "textDocument": {"uri": file_uri(str(path))},
                        "position": {"line": 0, "character": 0},
                    },
                ),
                timeout=5,
            )
            assert hover["contents"]["value"] == path.read_text(encoding="utf-8")
    finally:
        await client.shutdown()
