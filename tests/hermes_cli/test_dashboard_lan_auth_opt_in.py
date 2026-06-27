from __future__ import annotations

import copy

from hermes_cli.config import DEFAULT_CONFIG, save_config
from hermes_cli import web_server


def _write_dashboard_flag(enabled: bool) -> None:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["dashboard"]["allow_unauthenticated_lan"] = enabled
    save_config(cfg)


def test_non_loopback_dashboard_requires_auth_by_default():
    _write_dashboard_flag(False)

    assert web_server.should_require_auth("192.168.2.228") is True


def test_non_loopback_dashboard_can_be_explicitly_lan_exposed():
    _write_dashboard_flag(True)

    assert web_server.should_require_auth("192.168.2.228") is False
    assert web_server.should_require_auth("127.0.0.1") is False
