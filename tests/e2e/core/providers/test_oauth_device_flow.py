"""RFC 8628 device-code login: the real ``hermes auth add nous`` CLI against a fake portal.

The portal base URL override (``--portal-url``) is the product's documented
channel; the fake portal is the only thing not ours. Poll arrival times are
recorded SERVER-side, so the gaps are what the vendor would observe.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import pytest

from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.providers._oauth_helpers import kill_tagged, make_home, run_hermes
from tests.fakes.providers.oauth_token_server import OAuthTokenServer, unsigned_jwt

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tagged-tree cleanup uses /proc")

PENDING, SLOW_DOWN, APPROVE = "authorization_pending", "slow_down", "approve"
# Scheduler/request latency only ever makes a gap LONGER; this absorbs clock granularity.
SLACK_S = 0.1


@dataclass(frozen=True)
class Case:
    interval: int
    script: tuple[str, ...]


CASES: dict[str, Case] = {
    # interval 1 is below the product's cap, so this pins the baseline contract on main:
    # every pending answer is followed by a wait, the login completes, the pair persists.
    "pending_waits_interval": Case(1, (PENDING, PENDING, APPROVE)),
    "honors_server_interval": Case(3, (PENDING, PENDING, APPROVE)),
    "slow_down_adds_five_seconds": Case(1, (PENDING, SLOW_DOWN, PENDING, APPROVE)),
}

class PolledFasterThanAllowed(AssertionError):
    """Raised ONLY at the poll-cadence assertion, the signature of #121163 / #121254.

    It is the type ``known_gate`` accepts: a failed login, a wrong poll count or a timeout is a
    real failure even in a KNOWN cell."""


# Red on main for a tracked, open bug: case -> (the bug's own failure-message pattern, reason).
KNOWN: dict[str, tuple[str, str]] = {
    "honors_server_interval": (
        r"client polled faster than the server allows \(server interval 3s, .*required at least \[3\.0, 3\.0\]",
        "#121163 (dup #87432) device-code poll interval is capped to 1s "
        "(DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS used as a ceiling)"),
}


def expected_min_gaps(case: Case) -> list[float]:
    """RFC 8628 3.5: wait ``interval`` between polls; each slow_down adds 5 s for good."""
    gaps, interval = [], case.interval
    for step in case.script[:-1]:
        if step == SLOW_DOWN:
            interval += 5
        gaps.append(float(interval))
    return gaps


@pytest.mark.parametrize("name", list(CASES))
def test_device_code_login_poll_cadence(name: str, tmp_path) -> None:
    case = CASES[name]
    fh = make_home(tmp_path)
    srv = OAuthTokenServer(access_token_factory=lambda n: unsigned_jwt(
        {"scope": "inference:invoke", "exp": int(time.time()) + 3600, "sub": f"user-{n}"})).start()
    flow = srv.start_device_flow(interval=case.interval, script=list(case.script))
    try:
        proc = run_hermes(fh, ["auth", "add", "nous", "--type", "oauth", "--no-browser",
                               "--portal-url", srv.base_url, "--inference-url", f"{srv.base_url}/v1"],
                          timeout=90)
    finally:
        kill_tagged(fh.tag)
        srv.stop()
    out = f"exit={proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"

    assert proc.returncode == 0, f"login failed\n{out}"
    assert srv.device_requests == 1, f"expected one device authorization request\n{out}"
    assert len(flow.polls) == len(case.script), (
        f"expected {len(case.script)} token polls (script {case.script}), got {len(flow.polls)}\n{out}")
    approved = [g for g in srv.grants if g.issued_refresh_token]
    assert len(approved) == 1
    pool = fh.read_auth().get("credential_pool", {}).get("nous") or []
    assert [e.get("refresh_token") for e in pool] == [approved[0].issued_refresh_token], (
        f"auth.json does not hold the refresh token the portal issued: {pool}")

    gaps = [round(b - a, 3) for a, b in zip(flow.polls, flow.polls[1:])]
    want = expected_min_gaps(case)
    short = [(i, got, need) for i, (got, need) in enumerate(zip(gaps, want)) if got + SLACK_S < need]
    with known_gate(KNOWN, name, raises=PolledFasterThanAllowed):
        if short:
            raise PolledFasterThanAllowed(
                f"client polled faster than the server allows (server interval {case.interval}s, script {case.script}): "
                f"observed gaps {gaps}, required at least {want}; short polls (index, got, need): {short}")
