"""Messaging-adapter contract, Slack leg: the REAL gateway + REAL ``plugins/platforms/slack`` adapter.

The child is ``hermes gateway run`` on a throwaway HOME; the adapter's own SDK (slack_bolt Socket Mode + slack_sdk via the ``slack_shim`` sitecustomize) talks to
``tests/fakes/platforms/slack_standin.py``, a local stand-in shaped per the platform's published
API. Scenarios live in ``_contract.py`` and are identical for every adapter; this file only binds the
Slack driver and lists the scenarios that are red on main (``KNOWN``: scenario -> (the bug's failure-message
pattern, reason); see ``_suite.py``: xfail only on that message, pass once the fix lands).
"""

from __future__ import annotations

import sys

import pytest

from tests.e2e.core.platforms._drv_slack import SlackDriver
from tests.e2e.core.platforms._suite import rig_fixtures, run_scenario, scenario_params

pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group gateway harness"),
]

_PARTIAL = (r"a partial copy of the answer was left visible after a rejected finalize",
            "#95430 a rejected closing appendStream/stopStream leaves the partial stream next to the re-posted answer")
KNOWN: dict[str, tuple[str, str]] = {
    "stream_finalize_rejected": _PARTIAL,
    "stream_finalize_rejected_group": _PARTIAL,
    "stream_trailing_whitespace": (
        r"a streamed reply ending in whitespace is shown != once",
        "#121326 native streaming re-posts the whole reply when it ends in whitespace"),
    "planned_restart_notice": (
        r"a redelivered /restart restarted the gateway again|a second restart ack means the replayed /restart was obeyed",
        "#121325 a replayed /restart restarts the gateway again (guard needs Telegram update ids)"),
}
SKIP: dict[str, str] = {
    "heic_as_image": "the adapter only downloads https://*.slack.com file URLs (SSRF guard); a loopback "
                     "stand-in cannot serve one",
}

rig, rig_stream = rig_fixtures(SlackDriver)


@pytest.mark.parametrize("scenario", scenario_params(SKIP))
def test_contract(scenario: str, request: pytest.FixtureRequest, tmp_path) -> None:
    run_scenario(scenario, request, tmp_path, KNOWN)
