"""Tests for cron/scripts/classify_items.py."""

from __future__ import annotations

import io
import json as jsonlib
import sys
from unittest.mock import MagicMock


def test_monitor_classifier_gemma_config_preserves_explicit_max_output_tokens(
    monkeypatch, capsys
):
    import agent.auxiliary_client as aux_mod
    from cron.scripts import classify_items

    recorded = {}

    class DummyHTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            recorded["url"] = url
            recorded["json"] = json
            recorded["headers"] = headers
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": jsonlib.dumps(
                                        [
                                            {
                                                "index": 0,
                                                "score": 9,
                                                "reason": "reply today",
                                            }
                                        ]
                                    )
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            }
            return response

        def close(self):
            return None

    monkeypatch.setattr(
        "agent.gemini_native_adapter.httpx.Client",
        lambda *args, **kwargs: DummyHTTP(),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-monitor-test")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "auxiliary": {
                "monitor": {
                    "provider": "gemini",
                    "model": "gemma-4-31b-it",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta",
                }
            }
        },
    )
    with aux_mod._client_cache_lock:
        aux_mod._client_cache.clear()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_items.py",
            "--criteria",
            "Urgent if it needs a reply today.",
            "--threshold",
            "7",
            "--format",
            "json",
        ],
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            jsonlib.dumps(
                [
                    {
                        "id": "msg-1",
                        "title": "Need a reply",
                        "text": "Please get back to me today.",
                    }
                ]
            )
        ),
    )

    rc = classify_items.main()
    stdout = capsys.readouterr().out
    surfaced = jsonlib.loads(stdout)

    assert rc == 0
    assert surfaced[0]["id"] == "msg-1"
    assert recorded["url"].endswith("/models/gemma-4-31b-it:generateContent")
    assert recorded["json"]["generationConfig"]["maxOutputTokens"] == 1024


def test_parse_scores_keeps_explicit_index_behavior():
    from cron.scripts.classify_items import _parse_scores

    scores = _parse_scores(
        '[{"index": 0, "score": 9, "reason": "reply today"}]',
        1,
    )

    assert scores == {0: {"index": 0, "score": 9, "reason": "reply today"}}


def test_parse_scores_falls_back_to_array_position_when_index_is_omitted():
    from cron.scripts.classify_items import _parse_scores

    scores = _parse_scores(
        '[{"score": 9, "reason": "reply today"}]',
        1,
    )

    assert scores == {0: {"score": 9, "reason": "reply today"}}


def test_parse_scores_ignores_out_of_range_explicit_index_without_positional_fallback():
    from cron.scripts.classify_items import _parse_scores

    scores = _parse_scores(
        '[{"index": 4, "score": 9, "reason": "wrong item"}]',
        1,
    )

    assert scores == {}


def test_monitor_classifier_same_order_array_without_index_surfaces_above_threshold(
    monkeypatch, capsys
):
    from cron.scripts import classify_items

    def fake_call_llm(**kwargs):
        assert kwargs["task"] == "monitor"
        assert kwargs["max_tokens"] == 1024
        return MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='[{"score": 9, "reason": "reply today"}]'
                    )
                )
            ]
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_items.py",
            "--criteria",
            "Urgent if it needs a reply today.",
            "--threshold",
            "7",
            "--format",
            "json",
        ],
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            jsonlib.dumps(
                [
                    {
                        "id": "msg-1",
                        "title": "Need a reply",
                        "text": "Please get back to me today.",
                    }
                ]
            )
        ),
    )

    rc = classify_items.main()
    stdout = capsys.readouterr().out
    surfaced = jsonlib.loads(stdout)

    assert rc == 0
    assert surfaced[0]["id"] == "msg-1"
    assert surfaced[0]["score"] == 9
    assert surfaced[0]["reason"] == "reply today"


def test_monitor_classifier_below_threshold_reply_stays_silent(monkeypatch, capsys):
    from cron.scripts import classify_items

    def fake_call_llm(**kwargs):
        assert kwargs["task"] == "monitor"
        assert kwargs["max_tokens"] == 1024
        return MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='[{"score": 1, "reason": "ignore"}]'
                    )
                )
            ]
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_items.py",
            "--criteria",
            "Urgent if it needs a reply today.",
            "--threshold",
            "7",
            "--format",
            "json",
        ],
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            jsonlib.dumps(
                [
                    {
                        "id": "msg-1",
                        "title": "Need a reply",
                        "text": "Please get back to me today.",
                    }
                ]
            )
        ),
    )

    rc = classify_items.main()
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == ""
