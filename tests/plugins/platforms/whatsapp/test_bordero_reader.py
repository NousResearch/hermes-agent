from __future__ import annotations

import pytest

from plugins.platforms.whatsapp.bordero_reader import (
    BorderoReaderConfigError,
    build_ingest_prompt,
    load_bordero_reader_config,
    route_for_message,
)


UBBO = "120363000000000001@g.us"
SALDANHA = "120363000000000002@g.us"


def _routes(**overrides):
    routes = [
        {
            "group_jid": UBBO,
            "store": "PTT",
            "location": "UBBO",
            "telegram_chat_id": "-1003743117566",
            "telegram_thread_id": "101",
        },
        {
            "group_jid": SALDANHA,
            "store": "ODI",
            "location": "Saldanha",
            "telegram_chat_id": "-1003743117566",
            "telegram_thread_id": "102",
        },
    ]
    routes[0].update(overrides.get("ubbo", {}))
    routes[1].update(overrides.get("saldanha", {}))
    return routes


def test_disabled_reader_has_no_routes():
    config = load_bordero_reader_config({"bordero_read_only": False})
    assert config.enabled is False
    assert config.routes == {}


def test_enabled_reader_requires_exactly_two_canonical_store_routes():
    config = load_bordero_reader_config({"bordero_read_only": True, "bordero_routes": _routes()})

    assert config.enabled is True
    assert set(config.routes) == {UBBO, SALDANHA}
    assert config.routes[UBBO].store == "PTT"
    assert config.routes[UBBO].location == "UBBO"
    assert config.routes[SALDANHA].store == "ODI"
    assert config.routes[SALDANHA].location == "Saldanha"
    assert config.routes[UBBO].telegram_target == "telegram:-1003743117566:101"


@pytest.mark.parametrize(
    "bad_routes",
    [
        _routes()[0:1],
        _routes() + [_routes()[0]],
        _routes(ubbo={"group_jid": "Borderô UBBO"}),
        _routes(ubbo={"group_jid": SALDANHA}),
        _routes(ubbo={"store": "ODI"}),
        _routes(ubbo={"location": "Saldanha"}),
        _routes(ubbo={"telegram_thread_id": ""}),
    ],
)
def test_invalid_routes_fail_closed(bad_routes):
    with pytest.raises(BorderoReaderConfigError):
        load_bordero_reader_config({"bordero_read_only": True, "bordero_routes": bad_routes})


def test_route_lookup_is_exact_group_only_and_never_matches_dm_or_name():
    config = load_bordero_reader_config({"bordero_read_only": True, "bordero_routes": _routes()})

    assert route_for_message({"isGroup": True, "chatId": UBBO}, config) is config.routes[UBBO]
    assert route_for_message({"isGroup": False, "chatId": UBBO}, config) is None
    assert route_for_message({"isGroup": True, "chatId": "Borderô UBBO"}, config) is None
    assert route_for_message({"isGroup": True, "chatId": "999999999999999999@g.us"}, config) is None


def test_ingest_prompt_contains_only_configured_telegram_target_and_silence_rule():
    config = load_bordero_reader_config({"bordero_read_only": True, "bordero_routes": _routes()})
    prompt = build_ingest_prompt(config.routes[UBBO])

    assert "telegram:-1003743117566:101" in prompt
    assert "não responder no WhatsApp" in prompt
    assert "braza-operations" in prompt
    assert "telegram:-1003743117566:102" not in prompt
    assert "whatsapp:" not in prompt.lower()
    assert "send_message" not in prompt
