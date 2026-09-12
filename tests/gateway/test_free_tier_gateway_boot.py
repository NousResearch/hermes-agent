"""The messaging gateway is a boot owner of the Nous free tier.

Rung 5 made every demand-time site a read (provider resolution, ``/login``, the connector token), so a
process that never runs the bootstrap can never have an identity. `cmd_chat` and `hermes serve` run it;
this file pins that `hermes gateway run` does too, and does it BEFORE any adapter connects, so a fast
first DM cannot arrive with nothing to resolve.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
import gateway.run_startup as run_startup


@pytest.mark.asyncio
async def test_gateway_boot_runs_the_free_tier_bootstrap_before_any_adapter_connects(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    order: list[str] = []

    def fake_bootstrap() -> None:
        order.append("bootstrap")

    async def fake_prefilter(self):
        order.append("prefilter-platforms")
        # No pending connects: startup exits cleanly at the no-connections check, which is exactly the
        # path the cold cell drives. Shape mirrors the real return.
        return (False, 0, [], [])

    monkeypatch.setattr(GatewayRunner, "_start_free_tier_bootstrap", staticmethod(fake_bootstrap))
    monkeypatch.setattr(run_startup.GatewayStartupMixin, "_start_prefilter_platforms", fake_prefilter)

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=False)},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    ok = await runner.start()

    assert ok is True
    assert order[:2] == ["bootstrap", "prefilter-platforms"], order


def test_gateway_bootstrap_seam_calls_the_one_creator(monkeypatch):
    """The seam delegates to `free_tier_bootstrap.run_bootstrap`; nothing else in the gateway may mint."""
    import hermes_cli.free_tier_bootstrap as ftb
    calls: list[dict] = []
    monkeypatch.setattr(ftb, "run_bootstrap", lambda **kw: calls.append(kw))

    GatewayRunner._start_free_tier_bootstrap()

    assert calls == [{"announce": False}]
