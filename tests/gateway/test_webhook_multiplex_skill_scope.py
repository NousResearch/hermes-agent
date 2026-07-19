"""Regression test: multiplexed webhooks must load the URL profile's skills.

Bug: a POST to ``/p/<profile>/webhooks/<route>`` resolved the profile from
the URL (for session/credential scoping) but injected skill content BEFORE
entering that profile's runtime scope, so ``get_skill_commands()`` always
resolved skills against the process-default ``HERMES_HOME`` instead of the
named profile's — regardless of which profile the webhook was addressed to.

Covers:
- ``/p/<profile>/webhooks/<route>`` loads that profile's skill content, not
  the default profile's.
- Back-to-back requests for different profiles each get their own profile's
  skill content (the ``get_skill_commands()`` process-global cache must
  invalidate on profile change, not just platform change — see #14536 for
  the platform analogue).
"""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import agent.skill_commands as skill_commands_module
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _make_skill(skills_dir, name, marker):
    """A minimal skill whose body embeds ``marker`` so tests can tell which
    profile's copy actually got loaded."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""\
---
name: {name}
description: Test skill for {marker}.
---

# {name}

{marker}
"""
    )
    return skill_dir


def _make_adapter(routes, multiplex: bool) -> WebhookAdapter:
    config = PlatformConfig(
        enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes}
    )
    adapter = WebhookAdapter(config)

    class _Runner:
        config = GatewayConfig(multiplex_profiles=multiplex)
        adapters = {}

        def get_home_channel(self, *a, **kw):
            return None

    adapter.gateway_runner = _Runner()
    return adapter


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    app.router.add_route(
        "*", "/p/{profile}/webhooks/{route_name}", adapter._handle_webhook
    )
    return app


class TestWebhookMultiplexSkillScope:
    @pytest.mark.asyncio
    async def test_profile_prefixed_webhook_loads_that_profiles_skills(
        self, tmp_path, monkeypatch
    ):
        default_home = tmp_path / "default"
        coder_home = tmp_path / "coder"
        (default_home / "skills").mkdir(parents=True)
        (coder_home / "skills").mkdir(parents=True)
        _make_skill(default_home / "skills", "greet", "DEFAULT-PROFILE-SKILL")
        _make_skill(coder_home / "skills", "greet", "CODER-PROFILE-SKILL")

        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [("default", default_home), ("coder", coder_home)],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: coder_home if name == "coder" else default_home,
        )
        # Reset the process-global skill command cache so no earlier test
        # in this process biases the first scan.
        monkeypatch.setattr(skill_commands_module, "_skill_commands", {})
        monkeypatch.setattr(skill_commands_module, "_skill_commands_platform", None)
        monkeypatch.setattr(skill_commands_module, "_skill_commands_dir", None)

        routes = {
            "notify": {
                "secret": _INSECURE_NO_AUTH,
                "deliver_only": True,
                "deliver": "telegram",
                "deliver_extra": {"chat_id": "1"},
                "prompt": "ping",
                "skills": ["greet"],
            }
        }
        adapter = _make_adapter(routes, multiplex=True)

        from unittest.mock import AsyncMock

        mock_target = AsyncMock()
        mock_target.send = AsyncMock(return_value=SendResult(success=True))
        adapter.gateway_runner.adapters = {Platform("telegram"): mock_target}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/p/coder/webhooks/notify",
                data=json.dumps({}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Delivery": "d-coder",
                },
            )
            assert resp.status == 200

            resp = await cli.post(
                "/p/default/webhooks/notify",
                data=json.dumps({}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Delivery": "d-default",
                },
            )
            assert resp.status == 200

            # Same profile again, to prove the cache invalidation isn't a
            # one-shot toggle — it must follow every request.
            resp = await cli.post(
                "/p/coder/webhooks/notify",
                data=json.dumps({}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Delivery": "d-coder-2",
                },
            )
            assert resp.status == 200

        await asyncio.sleep(0.05)

        assert mock_target.send.await_count == 3
        delivered = [call.args[1] for call in mock_target.send.await_args_list]

        assert "CODER-PROFILE-SKILL" in delivered[0]
        assert "DEFAULT-PROFILE-SKILL" not in delivered[0]

        assert "DEFAULT-PROFILE-SKILL" in delivered[1]
        assert "CODER-PROFILE-SKILL" not in delivered[1]

        assert "CODER-PROFILE-SKILL" in delivered[2]
        assert "DEFAULT-PROFILE-SKILL" not in delivered[2]
