"""Unit checks for the WhatsApp seen-chat merge. Run: python -m unittest hermes_cli.whatsapp_seen_chats_test"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hermes_cli.whatsapp_seen_chats import (
    bridge_chats_for_editor,
    bridge_subjects_for_chats,
    collect_whatsapp_seen_chats,
    whatsapp_chat_kind,
)


class WhatsAppSeenChatsTest(unittest.TestCase):
    def test_group_kind_follows_jid_suffix(self):
        self.assertEqual(whatsapp_chat_kind("120363@g.us", "dm"), "group")
        self.assertEqual(whatsapp_chat_kind("15551234567@s.whatsapp.net"), "dm")

    def test_directory_alias_and_session_name_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "channel_directory.json").write_text(
                json.dumps({
                    "platforms": {
                        "whatsapp": [
                            {"id": "120363@g.us", "name": "120363@g.us", "type": "group"},
                            {"id": "15551234567@s.whatsapp.net:topic", "name": "Ada", "type": "dm", "thread_id": "topic"},
                        ]
                    }
                }),
                encoding="utf-8",
            )
            (home / "channel_aliases.json").write_text(
                json.dumps({"whatsapp": {"120363@g.us": "Job Quest"}}),
                encoding="utf-8",
            )
            conn = sqlite3.connect(home / "state.db")
            conn.execute(
                "CREATE TABLE sessions (chat_id TEXT, chat_type TEXT, display_name TEXT, title TEXT, "
                "started_at REAL, ended_at REAL, session_key TEXT, origin_json TEXT)"
            )
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "120363@g.us",
                    "group",
                    "Job Quest",
                    None,
                    100.0,
                    250.0,
                    "agent:jbq_orchestrator:whatsapp:group:120363@g.us",
                    None,
                ),
            )
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("other", "dm", "Skip me", None, 9.0, None, "agent:main:telegram:dm:other", None),
            )
            conn.commit()
            conn.close()

            chats = collect_whatsapp_seen_chats([home])

        by_id = {chat["chat_id"]: chat for chat in chats}
        self.assertEqual(set(by_id), {"120363@g.us", "15551234567@s.whatsapp.net"})
        self.assertEqual(by_id["120363@g.us"]["name"], "Job Quest")
        self.assertEqual(by_id["120363@g.us"]["kind"], "group")
        self.assertEqual(by_id["120363@g.us"]["last_seen"], 250.0)
        self.assertEqual(by_id["15551234567@s.whatsapp.net"]["name"], "Ada")
        self.assertEqual(by_id["15551234567@s.whatsapp.net"]["kind"], "dm")
        self.assertGreater(chats[0]["last_seen"] or 0, chats[1]["last_seen"] or 0)

    def test_numeric_session_name_is_replaced_by_the_bridge_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            conn = sqlite3.connect(home / "state.db")
            conn.execute(
                "CREATE TABLE sessions (chat_id TEXT, chat_type TEXT, display_name TEXT, title TEXT, "
                "started_at REAL, ended_at REAL, session_key TEXT, origin_json TEXT)"
            )
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "120363431353336060@g.us",
                    "group",
                    "120363431353336060",
                    None,
                    10.0,
                    None,
                    "agent:fpx_market:whatsapp:group:120363431353336060@g.us",
                    None,
                ),
            )
            conn.commit()
            conn.close()
            chats = collect_whatsapp_seen_chats(
                [home],
                bridge_chats=[{
                    "chat_id": "120363431353336060@g.us",
                    "name": "FPX Marketing",
                    "kind": "group",
                }],
            )
        self.assertEqual(chats[0]["name"], "FPX Marketing")

    def test_session_title_is_not_shown_as_the_chat_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "channel_aliases.json").write_text(
                json.dumps({"whatsapp": {"120363@g.us": "Hermes?"}}),
                encoding="utf-8",
            )
            conn = sqlite3.connect(home / "state.db")
            conn.execute(
                "CREATE TABLE sessions (chat_id TEXT, chat_type TEXT, display_name TEXT, title TEXT, "
                "started_at REAL, ended_at REAL, session_key TEXT, origin_json TEXT)"
            )
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "120363@g.us",
                    "group",
                    "120363",
                    "Hermes?",
                    10.0,
                    None,
                    "agent:fpx_laureline:whatsapp:group:120363@g.us",
                    None,
                ),
            )
            conn.commit()
            conn.close()

            chats = collect_whatsapp_seen_chats([home])
            saved = json.loads((home / "channel_aliases.json").read_text(encoding="utf-8"))

        self.assertIsNone(chats[0]["name"])
        self.assertEqual(saved["whatsapp"], {})

    def test_per_chat_lookup_keeps_a_subject_and_skips_the_numeric_placeholder(self):
        group_id = "120363431353336060@g.us"

        def lookup(chat_id):
            if chat_id == group_id:
                return {"name": "FPX Marketing", "isGroup": True}
            return {"name": chat_id.split("@", 1)[0], "isGroup": True}

        subjects = bridge_subjects_for_chats(
            [group_id, "120363428138939295@g.us", "15551234567"],
            lookup,
        )
        self.assertEqual(subjects, [{
            "chat_id": group_id,
            "name": "FPX Marketing",
            "kind": "group",
        }])

    def test_missing_chats_route_falls_back_to_per_chat_subjects(self):
        group_id = "120363412396471379@g.us"
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            conn = sqlite3.connect(home / "state.db")
            conn.execute(
                "CREATE TABLE sessions (chat_id TEXT, chat_type TEXT, display_name TEXT, title TEXT, "
                "started_at REAL, ended_at REAL, session_key TEXT, origin_json TEXT)"
            )
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (group_id, "group", group_id.split("@", 1)[0], "Hermes?", 10.0, None,
                 f"agent:jbq:whatsapp:group:{group_id}", None),
            )
            conn.commit()
            conn.close()
            bridge = bridge_chats_for_editor(
                [group_id],
                None,
                lambda chat_id: {"name": "Job Quest", "isGroup": True},
            )
            chats = collect_whatsapp_seen_chats([home], bridge_chats=bridge)
        self.assertEqual(chats[0]["name"], "Job Quest")

    def test_empty_bridge_list_does_not_ask_each_chat(self):
        def lookup(_chat_id):
            raise AssertionError("disconnected bridge must not be queried per chat")

        self.assertEqual(bridge_chats_for_editor(["120363@g.us"], [], lookup), [])

    def test_alias_prune_failure_still_lists_chats(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "channel_directory.json").write_text(
                json.dumps({
                    "platforms": {
                        "whatsapp": [
                            {"id": "120363@g.us", "name": "Job Quest", "type": "group"},
                        ]
                    }
                }),
                encoding="utf-8",
            )
            with mock.patch(
                "hermes_cli.whatsapp_seen_chats.prune_message_title_aliases",
                side_effect=OSError("read-only"),
            ):
                chats = collect_whatsapp_seen_chats([home])
        self.assertEqual([chat["chat_id"] for chat in chats], ["120363@g.us"])


if __name__ == "__main__":
    unittest.main()
