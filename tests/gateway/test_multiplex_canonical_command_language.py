"""Canonical slash commands must run in the routed profile's locale.

Before the fix the handler read the multiplexer launch home's language. Real
localized /new notices and other canonical commands then appeared in English.
"""
from pathlib import Path

import pytest

from agent.i18n import reset_language_cache, t
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
async def test_canonical_slash_commands_use_routed_locale_a_b_a(tmp_path):
    launch = tmp_path / "launch"
    family = tmp_path / "profiles" / "family"
    other = tmp_path / "profiles" / "other"
    for home, lang in ((launch, "en"), (family, "ru"), (other, "en")):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(f"display:\n  language: {lang}\n")
    reset_language_cache()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda source: family if source.profile == "family" else other

    async def handle_new(event, source, key):
        return True, (get_hermes_home(), t("gateway.reset.header_default"))

    runner._hm_cmd_new = handle_new
    try:
        with _profile_runtime_scope(launch):
            for profile, expected_home, expected_text in (
                ("family", family, "Сеанс"),
                ("other", other, "Session"),
                ("family", family, "Сеанс"),
            ):
                source = SessionSource(platform=Platform.TELEGRAM, user_id="demo", chat_id="demo", profile=profile)
                handled, (resolved_home, response) = await runner._hm_dispatch_canonical_command(
                    None, source, "synthetic-route", "new"
                )
                assert handled
                assert resolved_home == expected_home
                assert expected_text in response
                assert get_hermes_home() == launch  # no profile leakage
    finally:
        reset_language_cache()
