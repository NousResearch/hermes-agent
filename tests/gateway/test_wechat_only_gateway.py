"""HERMES_GATEWAY_WECHAT_ONLY retains only Weixin after env overrides."""

import json
from pathlib import Path


def test_hermes_gateway_wechat_only_strips_other_platforms(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_GATEWAY_WECHAT_ONLY", "1")
    monkeypatch.setenv("WEIXIN_TOKEN", "real-token-at-least-four-chars-xx")
    monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "wxacct")

    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    (home / "gateway.json").write_text(
        json.dumps(
            {
                "platforms": {
                    "telegram": {"enabled": True, "token": "tg-token-from-json"},
                    "weixin": {
                        "enabled": True,
                        "token": "wx-from-json",
                        "extra": {"account_id": "wxacct"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    from gateway.config import Platform, load_gateway_config

    cfg = load_gateway_config()
    assert set(cfg.platforms.keys()) == {Platform.WEIXIN}


def test_wechat_only_without_weixin_clears_platforms(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_GATEWAY_WECHAT_ONLY", "true")
    monkeypatch.delenv("WEIXIN_TOKEN", raising=False)
    monkeypatch.delenv("WEIXIN_ACCOUNT_ID", raising=False)

    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    (home / "gateway.json").write_text(
        json.dumps({"platforms": {"telegram": {"enabled": True, "token": "x"}}}),
        encoding="utf-8",
    )

    from gateway.config import load_gateway_config

    cfg = load_gateway_config()
    assert cfg.platforms == {}
