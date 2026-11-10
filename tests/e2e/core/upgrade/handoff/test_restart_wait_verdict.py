"""The handoff cell's restart-wait property judges the gateway log: every way the wait can end inside the
cap passes, a wait that outlives the cap or never ends fails and names its last reported holders."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.e2e.core.upgrade.handoff import _handoff as X

OPENED = ("2026-11-06 23:31:20,808 INFO gateway.run: Restart requested with 1 active work unit(s); "
          "deferring stop() until they finish (cap=120s) so in-flight turns are not amputated (#77184)")
HOLDING = ("2026-11-06 23:31:20,810 INFO gateway.run: Restart deferred: waiting on 1 active work unit(s) "
           "(0 wedged and excluded, 0 restart-safe and excluded; 120s remaining before force drain): "
           "[{'kind': 'managed', 'session': 'local-1', 'pid': 289}]")
CLOSES = {
    "drained": "INFO gateway.run: Restart deferred wait complete — active work drained; proceeding to stop()",
    "cap": "WARNING gateway.run: Restart after-turn wait timed out after 120s with 1 still active; proceeding",
    "excluded": "WARNING gateway.run: Restart deferred wait: 1 excluded work unit(s) remain (1 wedged, 0 restart-safe "
                "external cron); proceeding to stop()/drain",
}


def _verdict(tmp_path, *lines: str) -> str:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "gateway.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return X.restart_wait_verdict(SimpleNamespace(hermes_home=tmp_path))


@pytest.mark.parametrize("close", sorted(CLOSES))
def test_a_wait_that_ends_inside_the_cap_passes(tmp_path, close):
    assert _verdict(tmp_path, OPENED, HOLDING, f"2026-11-06 23:31:50,000 {CLOSES[close]}") == ""


def test_a_wait_that_outlives_the_cap_names_its_holders(tmp_path):
    verdict = _verdict(tmp_path, OPENED, HOLDING, f"2026-11-06 23:34:22,514 {CLOSES['drained']}")
    assert "held 182s" in verdict and "closed late" in verdict and "'session': 'local-1'" in verdict


def test_a_wait_that_never_closes_fails(tmp_path):
    assert "never closed" in _verdict(tmp_path, OPENED, HOLDING)


def test_a_later_wait_neither_closes_nor_names_the_holders_of_an_earlier_one(tmp_path):
    later_open = OPENED.replace("23:31:20,808", "23:32:00,000")
    later_holding = HOLDING.replace("23:31:20,810", "23:32:00,001").replace("local-1", "local-2")
    verdict = _verdict(tmp_path, OPENED, HOLDING, later_open, later_holding,
                       f"2026-11-06 23:32:10,000 {CLOSES['drained']}")
    assert "never closed" in verdict and "local-1" in verdict and "local-2" not in verdict
