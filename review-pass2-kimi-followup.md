# Channel Claim-Lock — Pass 2 Follow-up (Kimi K2.6)

**Reviewer:** Kimi K2.6  
**Branch:** release/claim-lock @ 80795e4b80  
**Scope:** Verify the 1 Major + 2 Minor findings from pass-2 are closed at HEAD.

---

## Verdict

**ADVISE-RELEASE.** All three pass-2 findings are resolved at 80795e4b80.

---

## Major Finding — Unreadable Claim File Disabling the Gate

**Status:** Closed.

`acquire_relay_claim` in `tools/bot_relay.py` now detects an unreadable claim file inside the `O_EXCL FileExistsError` arm. When the file is unreadable and younger than the lease, `_foreign_unreadable_claim` returns a synthetic foreign claim. The caller `_check_relay_claim_gate` therefore stands down instead of admitting the message. This closes the silent double-answer hole.

The young-unreadable fail-closed window mutes the channel for at most the 300 s lease. This is bounded and is the correct trade-off. An unreadable file implies an unknown writer or a crashed write; treating it as a live foreign claim is the only safe choice. The `_heal_unreadable_claim` path removes stale unreadable files after the lease expires, so the mute cannot extend beyond one lease period. A transient read error on a valid claim is also bounded by the lease because the valid writer renews the claim; the standby would see a normal foreign claim during that window anyway.

## Regression Check — FileExistsError-Arm Restructure

**Status:** No regressions.

The unreadable logic stays strictly inside the `FileExistsError` handler. It cannot misfire on the valid create race where a peer's complete file lands between the initial read and the exclusive create. `test_acquire_concedes_a_lost_create_race` still covers that race.

`_reacquire_healed` retries with `os.O_CREAT | os.O_EXCL | os.O_WRONLY`. If a peer recreates the file first, the call raises `FileExistsError` and the function returns `(False, <claim>)`, conceding the claim. This mirrors the original CAS semantics exactly.

## Minor Finding — Fallback Acquire Result Ignored

**Status:** Closed.

`_finish_channel_claim` in `gateway/session.py:1133-1144` now binds the fallback `acquire_channel_claim` result to `(fresh_ok, fresh)`. When `fresh_ok` is False and `fresh` is not None, it logs a warning that names the foreign holder instance and session. This gives operators visibility into the sub-millisecond race.

## Minor Finding — Release Silently Skips Unreadable File

**Status:** Closed.

`release_relay_claim` in `tools/bot_relay.py:856-866` now checks for the case where `existing` is None but `path.exists()`. It logs a warning and unlinks the unreadable claim file. This prevents a corrupted claim from standing the channel down for the rest of the lease.

## Regression Tests

Three new tests cover the new code:

- `test_unreadable_young_claim_fails_closed`
- `test_unreadable_dead_claim_heals_then_acquires`
- `test_release_removes_unreadable_claim`

CI evidence at HEAD: 151/151 unit tests pass, 6/6 e2e replay tests pass, 12/12 checks pass.

---

*End of follow-up report.*
