"""Regression tests for Telegram DM topics config normalization."""

from __future__ import annotations

import unittest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter, _normalize_dm_topics_config


class TestTelegramDmTopicsConfigNormalization(unittest.TestCase):
    def test_numeric_dict_shape_normalizes_to_list_shape(self):
        raw = {
            "0": {
                "chat_id": "-1003637268545",
                "topics": {
                    "0": {"name": "Correo electrónico", "thread_id": 1366},
                },
            },
        }

        self.assertEqual(
            _normalize_dm_topics_config(raw),
            [
                {
                    "chat_id": "-1003637268545",
                    "topics": [
                        {"name": "Correo electrónico", "thread_id": 1366},
                    ],
                },
            ],
        )

    def test_adapter_init_accepts_numeric_dict_shape(self):
        adapter = TelegramAdapter(
            PlatformConfig(
                enabled=True,
                token="***",
                extra={
                    "dm_topics": {
                        "0": {
                            "chat_id": "-1003637268545",
                            "topics": {
                                "0": {"name": "Correo electrónico", "thread_id": 1366},
                            },
                        },
                    }
                },
            )
        )

        self.assertEqual(adapter._dm_topic_chat_ids, {"-1003637268545"})
        self.assertEqual(adapter._dm_topics_config[0]["topics"][0]["thread_id"], 1366)


if __name__ == "__main__":
    unittest.main()
