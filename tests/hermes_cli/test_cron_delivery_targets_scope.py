"""Regression: the delivery-targets route read a profile secret unscoped.

The dashboard/desktop ``serve`` backend flips multi-profile hosting
(``tui_gateway.launch_profile_policy.activate_multi_profile_hosting`` →
``agent.secret_scope.set_multiplex_active(True)``) as soon as it hosts a second
profile home; after that ``get_secret`` fails closed for an unscoped read.

``GET /api/cron/delivery-targets`` calls
``cron.scheduler_delivery.cron_delivery_targets()``, which resolves each
platform's home chat id through ``get_secret``. It ran outside any profile
scope, so under multi-profile hosting every poll raised ``UnscopedSecretError``
(caught and logged as an error) and the response silently lost every configured
platform — leaving only the implicit ``local`` entry. The read must run inside
``_config_profile_scope``, like the sibling cron routes.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")


def test_delivery_targets_route_binds_profile_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("TELEGRAM_HOME_CHANNEL=12345\n")

    from agent import secret_scope

    secret_scope.set_multiplex_active(True)

    seen: dict = {}

    def _fake_targets():
        seen["chat"] = secret_scope.get_secret("TELEGRAM_HOME_CHANNEL")
        return []

    from cron import scheduler_delivery

    monkeypatch.setattr(scheduler_delivery, "cron_delivery_targets", _fake_targets)

    from hermes_cli.web_routers import cron as cron_router

    result = asyncio.run(cron_router.get_cron_delivery_targets())

    assert "chat" in seen, "route must call cron_delivery_targets()"
    assert seen["chat"] == "12345", "route must bind the profile secret scope before the read"
    assert result["targets"][0]["id"] == "local"
