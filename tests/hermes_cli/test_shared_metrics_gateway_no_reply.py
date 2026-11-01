"""Completed no-reply turns retain clock ownership, opt-in and the closed v4 contract."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.observability import relay_shared_metrics as rsm
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_gateway as smg
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def rows(monkeypatch):
    smg.drain()
    smg._reply_clocks.clear()
    smg._chat_homes.clear()
    got = []
    monkeypatch.setattr(rsm, "enabled", lambda: True)
    monkeypatch.setattr(rsm, "record_process_mark",
                        lambda mark, data: got.append((mark, data, str(get_hermes_home()))))
    yield got
    smg.drain()
    smg._reply_clocks.clear()
    smg._chat_homes.clear()


def source(platform="telegram", chat_id="chat"):
    return SimpleNamespace(platform=platform, chat_id=chat_id)


def test_no_reply_reasons_and_platforms_validate_against_contract_and_v4_schema():
    import jsonschema

    schema = json.loads((Path(contract.__file__).parent / "schemas/hermes.shared_metrics.v4.schema.json").read_text())
    reasons = {"silent", "interrupted", "empty", "dropped", "delivery_failed", "error"}
    assert contract.NO_REPLY_REASONS == reasons
    for reason in reasons:
        for platform in contract.ADAPTER_PLATFORMS.values():
            dims = smg.no_reply_fields(platform=platform, reason=reason)
            assert contract.counter_dimensions_are_valid(contract.NO_REPLY_METRIC, dims)
            jsonschema.validate({"name": contract.NO_REPLY_METRIC, "type": "counter", "dimensions": dims, "value": 1},
                                {"$defs": schema["$defs"], "$ref": "#/$defs/no_reply_counter"})
    assert smg.no_reply_fields(platform="telegram", reason="private text")["reason"] == "empty"


def test_end_clock_requires_its_own_token_and_is_idempotent(rows):
    src = source()
    old = smg.start_reply_clock(src)
    current = smg.start_reply_clock(src)
    assert current is not None and current != old
    assert not smg.end_reply_clock(src, src.chat_id, "empty", token=old)
    assert not smg.end_reply_clock(src, src.chat_id, "empty")
    assert smg._reply_clocks[(src.platform, src.chat_id)][0] == current
    assert smg.end_reply_clock(src, src.chat_id, "empty", token=current)
    assert not smg.end_reply_clock(src, src.chat_id, "empty", token=current)
    smg.drain()
    assert [(mark, dims) for mark, dims, _ in rows] == [
        ("hermes.gateway.no_reply", {"platform": "telegram", "reason": "empty"})]


@pytest.mark.parametrize("replied", [True, False], ids=["media", "text"])
def test_replied_or_stopped_clock_never_emits_no_reply(rows, caplog, replied):
    src = source()
    token = smg.start_reply_clock(src)
    with caplog.at_level(logging.INFO):
        if replied:
            assert smg.end_reply_clock(src, src.chat_id, "replied", token=token)
        else:
            smg.stop_reply_clock(src, src.chat_id)
        assert not smg.end_reply_clock(src, src.chat_id, "error", token=token)
    smg.drain()
    assert not any(mark == "hermes.gateway.no_reply" for mark, _, _ in rows)
    assert "turn ended without a reply" not in caplog.text


def test_no_reply_uses_owning_home_and_fronted_platform(rows, tmp_path):
    src = source("discord")
    home = tmp_path / "owner"
    scope = set_hermes_home_override(str(home))
    try:
        token = smg.start_reply_clock(src)
    finally:
        reset_hermes_home_override(scope)
    assert smg.end_reply_clock(source("relay"), src.chat_id, "silent", token=token)
    smg.drain()
    assert rows == [("hermes.gateway.no_reply", {"platform": "discord", "reason": "silent"}, str(home))]


def test_disabled_collection_still_logs_content_free_line(rows, monkeypatch, caplog):
    src = source()
    token = smg.start_reply_clock(src)
    monkeypatch.setattr(rsm, "enabled", lambda: False)
    with caplog.at_level(logging.INFO):
        assert smg.end_reply_clock(src, src.chat_id, "silent", token=token, session_key="session")
    smg.drain()
    assert rows == []
    assert [(r.levelname, r.message) for r in caplog.records if "turn ended without a reply" in r.message] == [
        ("INFO", "turn ended without a reply: platform=telegram session=session reason=silent")]


def test_end_clock_is_best_effort_even_if_adapter_resolution_raises(rows):
    class Broken:
        @property
        def platform(self):
            raise RuntimeError("broken adapter")

    assert smg.end_reply_clock(Broken(), "chat", "empty", token=1) is False


def test_catalog_platform_no_reply_matches_latency_in_owning_home(rows, tmp_path, monkeypatch):
    from hermes_cli.observability import shared_metrics_catalog as catalog
    from tests.hermes_cli.test_shared_metrics_gateway import _install_platform_plugin

    home = tmp_path / "owner"
    cleanup = _install_platform_plugin(home, monkeypatch, dir_name="vk", platform="vk", catalog_name="vk-platform")
    src = source("vk")
    scope = set_hermes_home_override(str(home))
    try:
        smg.start_reply_clock(src)
        smg.stop_reply_clock(src, src.chat_id)
        token = smg.start_reply_clock(src)
    finally:
        reset_hermes_home_override(scope)
    try:
        assert smg.end_reply_clock(src, src.chat_id, "silent", token=token)
        smg.drain()
        assert {mark for mark, _, _ in rows} == {"hermes.gateway.no_reply", "hermes.gateway.reply_latency"}
        assert all(dims["platform"] == "vk-platform" and owner == str(home) for _, dims, owner in rows)
    finally:
        cleanup()
        catalog._catalog_platform_owner.cache_clear()
