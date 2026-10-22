from __future__ import annotations

import json
from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter
from tests.gateway.test_feishu import _admits_group


def _message() -> SimpleNamespace:
    return SimpleNamespace(mentions=[], content="", message_type="text")


def _sender() -> SimpleNamespace:
    return SimpleNamespace(open_id="ou_alice", user_id=None)


def _adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(extra={"default_group_policy": "open", "require_mention": True}))
    adapter._bot_open_id = "ou_bot"
    return adapter
def test_group_rule_overlay_reloads_on_next_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter()
    assert not _admits_group(adapter, _message(), _sender(), "oc_hot")
    (tmp_path / "feishu_group_rules.json").write_text(
        json.dumps({"group_rules": {"oc_hot": {"require_mention": False}}}), encoding="utf-8"
    )
    assert _admits_group(adapter, _message(), _sender(), "oc_hot")
def test_overlay_keeps_default_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = FeishuAdapter(PlatformConfig(extra={"default_group_policy": "allowlist"}))
    adapter._bot_open_id = "ou_bot"
    (tmp_path / "feishu_group_rules.json").write_text(
        json.dumps({"group_rules": {"oc_hot": {"require_mention": False, "allowlist": None}}}), encoding="utf-8"
    )
    assert not _admits_group(adapter, _message(), _sender(), "oc_hot")


def test_invalid_overlay_keeps_last_valid_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter()
    path = tmp_path / "feishu_group_rules.json"
    path.write_text(json.dumps({"group_rules": {"oc_hot": {"require_mention": False}}}), encoding="utf-8")
    assert _admits_group(adapter, _message(), _sender(), "oc_hot")
    path.write_bytes(b"\xff")
    assert _admits_group(adapter, _message(), _sender(), "oc_hot")
