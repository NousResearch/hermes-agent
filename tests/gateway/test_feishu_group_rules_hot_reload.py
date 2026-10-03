"""Hot-reload for Feishu per-group ``group_rules`` (especially ``require_mention``)."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from tests.gateway.test_feishu import _admits_group


def _unmentioned_group_message():
    return SimpleNamespace(mentions=[], content="", message_type="text")


def _sender():
    return SimpleNamespace(open_id="ou_alice", user_id=None)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    merged = {"require_mention": True, "default_group_policy": "open"}
    if extra:
        merged.update(extra)
    adapter = FeishuAdapter(PlatformConfig(extra=merged))
    adapter._bot_open_id = "ou_bot"
    return adapter


def _write_rules(hermes_home, payload: dict):
    path = hermes_home / "feishu_group_rules.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(path, None)
    return path


def test_hot_reload_require_mention_without_restart(hermes_home):
    adapter = _make_adapter()
    message = _unmentioned_group_message()
    sender = _sender()

    assert _admits_group(adapter, message, sender, "oc_hot") is False

    _write_rules(
        hermes_home,
        {"group_rules": {"oc_hot": {"require_mention": False}}},
    )

    assert _admits_group(adapter, message, sender, "oc_hot") is True


def test_missing_hot_file_keeps_boot_config(hermes_home):
    adapter = _make_adapter(
        extra={"group_rules": {"oc_hot": {"require_mention": True}}},
    )
    message = _unmentioned_group_message()
    sender = _sender()

    assert not (hermes_home / "feishu_group_rules.json").exists()
    assert _admits_group(adapter, message, sender, "oc_hot") is False
    assert adapter._group_rules["oc_hot"].require_mention is True


def test_invalid_json_keeps_last_successfully_loaded_rules(hermes_home):
    adapter = _make_adapter()
    message = _unmentioned_group_message()
    sender = _sender()

    path = _write_rules(
        hermes_home,
        {"group_rules": {"oc_hot": {"require_mention": False}}},
    )
    assert _admits_group(adapter, message, sender, "oc_hot") is True

    path.write_text("{not-json", encoding="utf-8")
    later = time.time() + 2
    os.utime(path, (later, later))

    assert _admits_group(adapter, message, sender, "oc_hot") is True


def test_hot_rule_missing_require_mention_inherits_global(hermes_home):
    adapter = _make_adapter()
    _write_rules(hermes_home, {"group_rules": {"oc_hot": {"policy": "open"}}})

    assert _admits_group(adapter, _unmentioned_group_message(), _sender(), "oc_hot") is False
    assert adapter._group_rules["oc_hot"].require_mention is None


def test_empty_chat_id_inherits_global_require_mention(hermes_home):
    adapter = _make_adapter()
    _write_rules(
        hermes_home,
        {"group_rules": {"oc_hot": {"require_mention": False}}},
    )

    assert _admits_group(adapter, _unmentioned_group_message(), _sender(), "") is False


def test_same_mtime_different_size_reloads_without_utime_bump(hermes_home):
    adapter = _make_adapter()
    message = _unmentioned_group_message()
    sender = _sender()

    path = _write_rules(
        hermes_home,
        {"group_rules": {"oc_hot": {"require_mention": True}}},
    )
    first = path.stat()
    assert _admits_group(adapter, message, sender, "oc_hot") is False

    path.write_text(
        json.dumps({"group_rules": {"oc_hot": {"require_mention": False, "note": "bigger"}}}),
        encoding="utf-8",
    )
    os.utime(path, (first.st_mtime, first.st_mtime))
    assert path.stat().st_mtime == first.st_mtime
    assert path.stat().st_size != first.st_size
    assert _admits_group(adapter, message, sender, "oc_hot") is True
