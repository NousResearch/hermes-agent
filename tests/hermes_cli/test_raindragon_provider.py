"""RainDragon first-class provider wiring: the providers.py resolver must know it,
or a saved ``provider: raindragon`` is discarded and env auto-detect wins (the
same failure test_upstage_provider.py guards for Upstage)."""

from __future__ import annotations

import sys
import types

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv


class TestRainDragonResolver:

    def test_resolve_provider_full_recognizes_raindragon(self):
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full("raindragon", {}, [])
        assert pdef is not None
        assert pdef.id == "raindragon"
        assert pdef.base_url == "https://inference.raindragon.ai/v1"
        assert "RAINDRAGON_API_KEY" in pdef.api_key_env_vars

    def test_alias_resolves(self):
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full("raindragon-ai", {}, [])
        assert pdef is not None and pdef.id == "raindragon"
