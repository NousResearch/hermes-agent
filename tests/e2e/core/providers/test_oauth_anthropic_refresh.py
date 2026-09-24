"""Anthropic OAuth access token expires mid-turn: one refresh, the turn completes, the rotation persists.

Real ``hermes -z`` process, native ``anthropic`` provider on the OAuth route
(``sk-ant-oat…`` pool credential), inference against a loopback Messages
endpoint (accepted base-URL override ``http://127.0.0.1:<port>/anthropic``),
and the vendor token endpoint (hardcoded ``https://platform.claude.com``)
reached through the TLS-intercepting proxy — nothing of ours is faked.

The first model call is answered with a tool call; the follow-up call (same,
now expired, bearer) gets the vendor's 401. The single-use refresh token must
be spent exactly once, the retry must carry the new bearer, and auth.json must
hold the NEW, still-live refresh token afterwards — otherwise the next process
replays a spent token and the login is lost.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid

import pytest

from tests.e2e.core.providers._oauth_helpers import OLD_ACCESS, run_hermes, start_anthropic_rig, wait_until

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="tagged-tree cleanup uses /proc")

FINAL = "OAUTH-REFRESH-TURN-COMPLETE"
EXPIRED = ("error", 401, "authentication_error", "OAuth token has expired")
SLOW_TOKEN_ENDPOINT_S = 2.0

# Red on main for a tracked, open bug. Strict: XPASS fails, forcing the entry out with the fix.
KNOWN: dict[str, str] = {
    "aux_call_shares_expired_token": (
        "#120815 a stale credential pool (the auxiliary title call's) persists the spent refresh token "
        "over the rotation the main turn just committed"),
}


def _has_tool_result(body: dict) -> bool:
    return any(isinstance(m.get("content"), list) and any(b.get("type") == "tool_result" for b in m["content"])
               for m in body.get("messages") or [])


def _run_turn(tmp_path, *, aux_401: bool):
    canary = f"CANARY-{uuid.uuid4().hex[:10]}"
    note = tmp_path / "note.txt"
    note.write_text(canary, encoding="utf-8")
    # Race order that loses the login on main, pinned with events: the auxiliary (title) call is
    # sent with the expired bearer before the main turn refreshes, and its 401 lands while the
    # main turn's refresh POST is in flight (the vendor token endpoint answers slowly).
    aux_sent, refresh_in_flight, aux_answered, aux_settled = (threading.Event() for _ in range(4))
    aux_answered_at: list[int] = []

    def answer_aux() -> tuple:
        aux_answered_at.append(time.time_ns())
        aux_answered.set()
        return EXPIRED

    def decide(record: dict) -> tuple:
        body, bearer = record["body"], record["bearer"] or record["x_api_key"]
        if not body.get("tools"):  # auxiliary call (title generation)
            if aux_401 and bearer == OLD_ACCESS:
                aux_sent.set()
                return ("hold", refresh_in_flight, ("call", answer_aux))
            return ("text", "OAuth refresh title")
        if not _has_tool_result(body):
            return ("tool", "read_file", {"path": str(note)})
        if bearer == OLD_ACCESS:
            return ("hold", aux_sent, EXPIRED) if aux_401 else EXPIRED
        return ("hold", aux_settled, ("text", FINAL)) if aux_401 else ("text", FINAL)

    rig = start_anthropic_rig(tmp_path, decide, title_generation=aux_401)

    def slow_token_endpoint(_grant) -> None:
        refresh_in_flight.set()
        aux_answered.wait(60)
        threading.Event().wait(SLOW_TOKEN_ENDPOINT_S)  # vendor latency window, not synchronization

    def settle() -> None:
        # The aux client has handled its 401 once auth.json is written after that 401 was sent
        # (bounded: a build that never writes there just releases the turn at the deadline).
        try:
            wait_until(lambda: aux_answered_at, 90, "the aux 401 to be sent")
            wait_until(lambda: rig.fh.auth_path.stat().st_mtime_ns > aux_answered_at[0] + int(
                SLOW_TOKEN_ENDPOINT_S * 1e9), 15, "a write after the rotation")
        except AssertionError:
            pass
        aux_settled.set()

    if aux_401:
        rig.tokens.before_refresh_response = slow_token_endpoint
        threading.Thread(target=settle, daemon=True).start()
    try:
        proc = run_hermes(rig.fh, ["-z", "Read the note file and report."], extra_env=rig.child_env, timeout=150)
    finally:
        for ev in (aux_sent, refresh_in_flight, aux_answered, aux_settled):
            ev.set()
        rig.stop()
    return rig, proc, canary


@pytest.mark.parametrize("case", [
    pytest.param("main_turn_only", id="main_turn_only"),
    pytest.param("aux_call_shares_expired_token", id="aux_call_shares_expired_token",
                 marks=pytest.mark.xfail(strict=True, reason=KNOWN["aux_call_shares_expired_token"])),
])
def test_expired_oauth_token_mid_turn_refreshes_once_and_persists(case: str, tmp_path) -> None:
    rig, proc, canary = _run_turn(tmp_path, aux_401=case != "main_turn_only")
    out = f"exit={proc.returncode}\nstdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-2500:]}"
    mains = rig.messages.main_requests()
    grants = rig.tokens.refresh_grants()

    assert proc.returncode == 0 and FINAL in proc.stdout, f"turn did not deliver the answer\n{out}"
    assert not rig.proxy.refused or all("claude" not in r and "anthropic" not in r for r in rig.proxy.refused)
    assert len(grants) == 1 and grants[0].status == 200, (
        f"expected exactly one successful refresh grant, got {[(g.presented, g.status, g.error) for g in grants]}")
    assert grants[0].presented == rig.seed_refresh, "the refresh spent a token other than the stored one"
    new_access, new_refresh = grants[0].issued_access_token, grants[0].issued_refresh_token

    tool_rounds = [r for r in mains if _has_tool_result(r["body"])]
    assert any(canary in str(r["body"]) for r in tool_rounds), "the tool result never reached the model"
    assert [r["bearer"] or r["x_api_key"] for r in tool_rounds][-1] == new_access, (
        "the retried request did not carry the refreshed bearer")

    row = rig.pool_row("e1")
    assert rig.tokens.is_live(row.get("refresh_token", "")), (
        f"auth.json holds a refresh token the vendor no longer accepts (login lost on the next process): "
        f"disk={row.get('refresh_token')!r} status={row.get('last_status')!r}; issued={new_refresh!r}, "
        f"spent={sorted(rig.tokens.spent)}")
    assert row.get("access_token") == new_access
