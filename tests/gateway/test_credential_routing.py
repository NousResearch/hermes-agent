import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.credential_routing import route_runtime_kwargs, select_entry_for_gateway_route
from gateway.session import SessionSource


class FakeEntry:
    def __init__(self, label, *, priority=0, request_count=0, quota=None, token=None):
        self.id = label
        self.label = label
        self.priority = priority
        self.request_count = request_count
        self.extra = {"quota": quota} if quota is not None else {}
        self.access_token = token or f"token-{label}"
        self.runtime_api_key = self.access_token
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.runtime_base_url = self.base_url


class FakePool:
    def __init__(self, entries):
        self._entries = list(entries)

    def _available_entries(self, *, clear_expired=False, refresh=False):
        return list(self._entries)

    def entries(self):
        return list(self._entries)


def _fresh_quota(primary=1, secondary=20):
    return {
        "primary_percent": primary,
        "secondary_percent": secondary,
        "observed_at": time.time(),
    }


def test_select_route_prefers_allowed_pro_and_excludes_deny():
    pool = FakePool([
        FakeEntry("device_code", priority=0),
        FakeEntry("gpt-pro-account-7", priority=5, request_count=0),
        FakeEntry("gpt-pro-account-5", priority=1, request_count=2),
    ])
    rule = {
        "allow_labels": ["gpt-pro-account-5", "gpt-pro-account-7"],
        "deny_labels": ["device_code"],
    }

    selected = select_entry_for_gateway_route(pool, rule)

    assert selected.label == "gpt-pro-account-7"


def test_conditional_plus_requires_fresh_quota_and_fails_closed_when_missing():
    pool = FakePool([
        FakeEntry("codex-plus-account-3"),
        FakeEntry("gpt-pro-account-7", priority=9),
    ])
    rule = {
        "allow_labels": ["codex-plus-account-3", "gpt-pro-account-7"],
        "conditional_labels": ["codex-plus-account-3"],
        "max_secondary_percent": 80,
    }

    selected = select_entry_for_gateway_route(pool, rule)

    assert selected.label == "gpt-pro-account-7"


def test_conditional_plus_is_selectable_with_fresh_low_quota():
    pool = FakePool([
        FakeEntry("codex-plus-account-3", priority=0, quota=_fresh_quota(5, 20)),
        FakeEntry("gpt-pro-account-7", priority=9),
    ])
    rule = {
        "allow_labels": ["codex-plus-account-3", "gpt-pro-account-7"],
        "conditional_labels": ["codex-plus-account-3"],
        "max_primary_percent": 80,
        "max_secondary_percent": 80,
    }

    selected = select_entry_for_gateway_route(pool, rule)

    assert selected.label == "codex-plus-account-3"


def test_conditional_plus_over_threshold_is_skipped():
    pool = FakePool([
        FakeEntry("codex-plus-account-3", priority=0, quota=_fresh_quota(1, 89)),
        FakeEntry("gpt-pro-account-7", priority=9),
    ])
    rule = {
        "allow_labels": ["codex-plus-account-3", "gpt-pro-account-7"],
        "conditional_labels": ["codex-plus-account-3"],
        "max_secondary_percent": 80,
    }

    selected = select_entry_for_gateway_route(pool, rule)

    assert selected.label == "gpt-pro-account-7"


def test_route_runtime_kwargs_applies_weixin_route_and_direct_binds_selected_token():
    source = SessionSource(platform=Platform.WEIXIN, chat_id="wx-dm", chat_type="dm")
    pool = FakePool([
        FakeEntry("gpt-pro-account-7", token="secret-token"),
    ])
    cfg = {
        "gateway_credential_routing": {
            "rules": [
                {
                    "provider": "openai-codex",
                    "platform": "weixin",
                    "allow_labels": ["gpt-pro-account-7"],
                    "fallback_policy": "fail_closed",
                }
            ]
        }
    }

    routed = route_runtime_kwargs(
        {"provider": "openai-codex", "api_key": "old", "credential_pool": pool},
        source=source,
        user_config=cfg,
        pool_loader=lambda provider: pool,
    )

    assert routed["api_key"] == "secret-token"
    assert routed["credential_pool"] is None
    assert routed["credential_label"] == "gpt-pro-account-7"


def test_fail_closed_route_raises_when_no_candidate():
    source = SessionSource(platform=Platform.FEISHU, chat_id="feishu", chat_type="dm")
    pool = FakePool([FakeEntry("device_code")])
    cfg = {
        "gateway_credential_routing": {
            "rules": [
                {
                    "provider": "openai-codex",
                    "platform": "feishu",
                    "allow_labels": ["gpt-pro-account-7"],
                    "fallback_policy": "fail_closed",
                }
            ]
        }
    }

    with pytest.raises(RuntimeError):
        route_runtime_kwargs(
            {"provider": "openai-codex", "credential_pool": pool},
            source=source,
            user_config=cfg,
            pool_loader=lambda provider: pool,
        )
