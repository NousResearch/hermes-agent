"""Topic fallback follows the real YAML loader and profile-scoped env precedence."""
import json
from pathlib import Path

import pytest

from gateway.config import Platform, load_gateway_config
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.mark.parametrize("value,expected", [
    (None, "parent_chat"), ("", "parent_chat"), ("  ", "parent_chat"),
    ("parent_chat", "parent_chat"), ("parent_then_home", "parent_then_home"), ("main_chat", None),
    ("error_notice", "error_notice"), ("silent", "silent"),
    ("invalid", None), (False, None), (123, None),
])
def test_real_yaml_extra_loader_accepts_default_and_rejects_invalid(tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FEISHU_TOPIC_DELIVERY_FALLBACK", raising=False)
    (tmp_path / "config.yaml").write_text(json.dumps({"platforms": {"feishu": {"enabled": True, "extra": {
        "topic_delivery_fallback": value}}}}))
    config = load_gateway_config().platforms[Platform.FEISHU]
    if expected is None:
        with pytest.raises(ValueError, match="topic_delivery_fallback"):
            FeishuAdapter._load_settings(config.extra)
    else:
        assert FeishuAdapter._load_settings(config.extra).topic_delivery_fallback == expected


def test_profile_a_b_a_env_over_yaml_without_launch_leak(tmp_path, monkeypatch):
    import agent.secret_scope as ss
    from gateway.run import _profile_runtime_scope

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_TOPIC_DELIVERY_FALLBACK", "invalid-launch-value")
    a, b = tmp_path / "a", tmp_path / "b"
    for home, mode in ((a, "silent"), (b, "error_notice")):
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"platforms": {"feishu": {"enabled": True, "extra": {
            "topic_delivery_fallback": mode}}}}))
    (a / ".env").write_text("FEISHU_TOPIC_DELIVERY_FALLBACK=parent_chat\n")
    (b / ".env").write_text("FEISHU_TOPIC_DELIVERY_FALLBACK=  \n")
    previous = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    try:
        seen = []
        for home in (a, b, a):
            with _profile_runtime_scope(home, hydrate_secrets=False):
                cfg = load_gateway_config().platforms[Platform.FEISHU]
                seen.append(FeishuAdapter._load_settings(cfg.extra).topic_delivery_fallback)
        assert seen == ["parent_chat", "error_notice", "parent_chat"]
        (b / "config.yaml").write_text('{"platforms":{"feishu":{"enabled":true}}}')
        with _profile_runtime_scope(b, hydrate_secrets=False):
            assert FeishuAdapter._load_settings(load_gateway_config().platforms[Platform.FEISHU].extra).topic_delivery_fallback == "parent_chat"
    finally:
        ss.set_multiplex_active(previous)
