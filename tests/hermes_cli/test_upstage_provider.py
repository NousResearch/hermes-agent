"""Focused tests for Upstage Solar first-class provider wiring.

Regression guard for the bug where `hermes model` saved `provider: upstage`
correctly but, on re-entry, showed a different provider as active. Root cause:
`hermes_cli/providers.py` (the resolver behind `resolve_provider_full`) had no
`upstage` overlay, so `resolve_provider_full("upstage")` returned None, the
config provider was discarded, and resolution fell through to env auto-detect.
"""

from __future__ import annotations

import sys
import types


if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv


class TestUpstageResolver:
    """The providers.py resolver must recognise upstage (the actual bug)."""

    def test_resolve_provider_full_recognizes_upstage(self):
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full("upstage", {}, [])
        assert pdef is not None, (
            "resolve_provider_full('upstage') returned None — config "
            "`provider: upstage` would be discarded and auto-detect would win"
        )
        assert pdef.id == "upstage"
        assert pdef.base_url == "https://api.upstage.ai/v1"
        assert "UPSTAGE_API_KEY" in pdef.api_key_env_vars






