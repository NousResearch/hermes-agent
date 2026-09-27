"""Messaging-adapter contract, Telegram leg: the REAL gateway + REAL ``plugins/platforms/telegram`` adapter.

The child is ``hermes gateway run`` on a throwaway HOME; the adapter's own SDK (python-telegram-bot via ``extra.base_url``) talks to
``tests/fakes/platforms/telegram_standin.py``, a local stand-in shaped per the platform's published
API. Scenarios live in ``_contract.py`` and are identical for every adapter; this file only binds the
Telegram driver and lists the scenarios that are red on main (``KNOWN``: scenario -> (the bug's failure-message
pattern, reason); see ``_suite.py``: xfail only on that message, pass once the fix lands).
"""

from __future__ import annotations

import sys

import pytest

from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._suite import rig_fixtures, run_scenario, scenario_params

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group gateway harness"),
]

KNOWN: dict[str, tuple[str, str]] = {
    "heic_as_image": (
        r"a HEIC photo sent as a file did not reach the model as an image",
        "#119593 HEIC photo sent as a file is refused by the image cache (never reaches the model)"),
}
SKIP: dict[str, str] = {}

rig, rig_stream = rig_fixtures(TelegramDriver)


@pytest.mark.parametrize("scenario", scenario_params(SKIP))
def test_contract(scenario: str, request: pytest.FixtureRequest, tmp_path) -> None:
    run_scenario(scenario, request, tmp_path, KNOWN)
