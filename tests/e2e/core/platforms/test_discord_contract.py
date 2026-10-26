"""Messaging-adapter contract, Discord leg: the REAL gateway + REAL ``plugins/platforms/discord`` adapter.

The child is ``hermes gateway run`` on a throwaway HOME; the adapter's own SDK (discord.py via the ``discord_shim`` sitecustomize) talks to
``tests/fakes/platforms/discord_standin.py``, a local stand-in shaped per the platform's published
API. Scenarios live in ``_contract.py`` and are identical for every adapter; this file only binds the
Discord driver and lists the scenarios that are red on main (``KNOWN``: scenario -> (the bug's failure-message
pattern, reason); see ``_suite.py``: xfail only on that message, pass once the fix lands).
"""

from __future__ import annotations

import sys

import pytest

from tests.e2e.core.platforms._drv_discord import DiscordDriver
from tests.e2e.core.platforms._suite import rig_fixtures, run_scenario, scenario_params

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group gateway harness"),
]

KNOWN: dict[str, tuple[str, str]] = {
    "planned_restart_notice": (
        r"a redelivered /restart restarted the gateway again|a second restart ack means the replayed /restart was obeyed",
        "#121325 a replayed /restart restarts the gateway again (guard needs Telegram update ids)"),
}
SKIP: dict[str, str] = {
    "heic_as_image": "image attachments are fetched by URL behind the SSRF guard, which refuses the loopback "
                     "stand-in's CDN URL (a real cdn.discordapp.com URL passes)",
}

rig, rig_stream = rig_fixtures(DiscordDriver)


@pytest.mark.parametrize("scenario", scenario_params(SKIP))
def test_contract(scenario: str, request: pytest.FixtureRequest, tmp_path) -> None:
    run_scenario(scenario, request, tmp_path, KNOWN)
