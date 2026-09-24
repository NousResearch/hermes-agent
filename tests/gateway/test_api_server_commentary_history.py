"""The API server's independent HTTP history route obeys the shared display contract.

API-server coverage adapted from PhanAnh-V's #88069, on #107386's projector.
"""

import json
from copy import deepcopy

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_http_history_follows_db_owner_a_b_a_and_preserves_canonical_rows(
    tmp_path, monkeypatch
):
    # The foreground/home must not choose the policy for a different session DB.
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "config.yaml").write_text("display:\n  show_commentary: false\n")
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    homes = [tmp_path / "a", tmp_path / "b"]
    databases = []
    try:
        for index, home in enumerate(homes):
            home.mkdir()
            (home / "config.yaml").write_text(
                f"display:\n  show_commentary: {'true' if index == 0 else 'false'}\n"
            )
            db = SessionDB(home / "state.db")
            databases.append(db)
            db.create_session("same-id", "api_server")
            db.append_message("same-id", "user", "Check.")
            db.append_message(
                "same-id",
                "assistant",
                "Final.",
                reasoning="Inspect\n\nPublic.",
                codex_reasoning_items=[
                    {
                        "type": "reasoning",
                        "id": "rs",
                        "encrypted_content": "opaque",
                        "summary": [{"type": "summary_text", "text": "Inspect"}],
                    }
                ],
                codex_message_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "Public."}],
                    }
                ],
            )
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        app = web.Application()
        app.router.add_get(
            "/api/sessions/{session_id}/messages", adapter._handle_session_messages
        )
        originals = [deepcopy(db.get_messages("same-id")) for db in databases]
        async with TestClient(TestServer(app)) as client:
            for index in [0, 1, 0]:
                adapter._session_db = databases[index]
                response = await client.get("/api/sessions/same-id/messages")
                assert response.status == 200, await response.text()
                result = await response.json()
                row = result["data"][1]
                assert row["content"] == "Final."
                assert row["reasoning"] == "Inspect\n\nPublic."
                assert row["display_commentary"] == (["Public."] if index == 0 else [])
                assert row["display_reasoning"] == "Inspect"
                assert row["display_reasoning_items"][0]["id"] == "rs"
                assert (
                    "codex_message_items" not in row
                    and "codex_reasoning_items" not in row
                )
                assert "opaque" not in json.dumps(result)
        assert [db.get_messages("same-id") for db in databases] == originals
    finally:
        for db in databases:
            db.close()
