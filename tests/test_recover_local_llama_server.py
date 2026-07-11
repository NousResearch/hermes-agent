from __future__ import annotations

import scripts.recover_local_llama_server as recover


def test_active_slot_probe_strips_openai_v1_base_url(monkeypatch):
    captured = {}

    def fake_request(url, *, api_key="", method="GET", timeout=5.0):
        captured["url"] = url
        captured["method"] = method
        return [{"id": 3, "is_processing": True}]

    monkeypatch.setattr(recover, "_request", fake_request)

    slot_ids = recover._active_slot_ids("http://127.0.0.1:9090/v1", "key", 1.0)

    assert slot_ids == [3]
    assert captured == {
        "url": "http://127.0.0.1:9090/slots",
        "method": "GET",
    }


def test_cancel_slots_strips_openai_v1_base_url(monkeypatch):
    captured = {}

    def fake_request(url, *, api_key="", method="GET", timeout=5.0):
        captured["url"] = url
        captured["method"] = method
        return {}

    monkeypatch.setattr(recover, "_request", fake_request)

    assert recover._cancel_slots("http://127.0.0.1:9090/v1/", "key", [7], 1.0)
    assert captured == {
        "url": "http://127.0.0.1:9090/slots/7?action=cancel",
        "method": "POST",
    }
