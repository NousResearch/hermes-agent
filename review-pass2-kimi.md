# Channel Claim-Lock — Pass 2 Code Review (Kimi K2.6)

**Reviewer:** Kimi K2.6 (second pass, adversarial)  
**Branch:** release/claim-lock @ 82bb1a4b05  
**Delta:** fork/main (47676981f55de91231fdeef2f0eec47e7c209e78) → HEAD  
**Reviewable files:** gateway/session.py, hermes_state_runtime.py, tools/bot_relay.py, gateway/run_turn_runner.py, gateway/session_recovery.py, tests/gateway/test_channel_claims.py, tests/gateway/test_channel_claim_e2e_replay.py, tests/tools/test_bot_relay_relay_claims.py

---

## Verdict

**FIX-FIRST (1 Major + 2 Minor).**

The pass-1 fixes are structurally sound. The Major finding is a new adversarial edge uncovered in the relay-claim read path: a corrupted or empty claim file silently disables the cross-instance gate per-channel. This is a first-pass blind spot, not a regression from the fixes.

---

## Pass-1 Fix Verification

| # | Pass-1 Major / Minor | Status | Evidence |
|---|---|---|---|
| 1 | Orphan placeholder lingers after relay stand-down, blocks channel for lease. `_release_orphan_claim_placeholder` at `gateway/session.py:1159–1169` called inside `except ChannelClaimBlocked` at line 1225. | **RESOLVED** | Code present. The `except ChannelClaimBlocked` in `_get_or_create_session_impl` immediately calls `_release_orphan_claim_placeholder(source, session_key)` then re-raises. Regression test `test_relay_standdown_releases_the_sqlite_placeholder` passes (148/148). |
| 2 | Thread-fold at drain-time ran in monitor mode (routing delta). | **RESOLVED** | Lines 989–1006 gate `elif fold_live:` behind `not monitor`: monitor logs “would fold” and returns the original key, zero routing delta. Test `test_monitor_keeps_the_claim_store_maintained` asserts `reply is not parent` and `reply.session_key != parent.session_key`, i.e., no fold. |
| 3 | Relay claim fresh-create not CAS; two gateways could both “win”. | **RESOLVED** | `tools/bot_relay.py:752` opens with `os.O_CREAT | os.O_EXCL`; loser catches `FileExistsError`, re-reads the winner’s claim (line 757), and returns `(False, existing)`. Test `test_acquire_concedes_a_lost_create_race` verifies the loser concedes. E2E cross-instance test `test_cross_instance_second_gateway_stands_down` verifies the full gateway-level stand-down. |
| 4 | Minors (stand-down warning order, relay path segment validation, BOM tolerance). | **RESOLVED** | Signatures present in `_normalize_for_filename`, `BOM_UTF8 check`, and log callback ordering in `run_turn_runner.py` `_make_bg_review_callbacks`. |

---

## Findings

### Critical

*None.*

### Major

**`tools/bot_relay.py:707–757` — Corrupted or empty relay claim file silently disables the cross-instance gate.**

`_read_relay_claim` returns `None` for any `OSError` or `json.JSONDecodeError`. `acquire_relay_claim` treats `existing=None` as “no claim yet”, tries `O_EXCL` creation, gets `FileExistsError`, re-reads, gets `None` again, and returns `(False, None)`. The caller `_check_relay_claim_gate` skips its stand-down block because `claim` is `None`, and the message is admitted.

**Why it matters:** A crash during `fd.write` (power loss, ENOSPC, signal) can leave a zero-byte or partial claim file permanently. A second gateway will then *silently* believe the channel is free and double-answer. This is the exact failure mode the feature was built to prevent, triggered by a filesystem edge case, not by a design choice.

**Suggested fix:** Make `acquire_relay_claim` write atomically: marshal the candidate to a temp file in the same directory, then `os.replace` onto the target path. This removes the visible empty-file window and the persistent partial-write state. If a legacy corrupted file is ever encountered, treat it as a live foreign claim (fail-closed for the relay gate).

### Minor

**`gateway/session.py:1133–1138` — `_finish_channel_claim` ignores the return value of the fallback `acquire_channel_claim`.**

After a failed `transfer_channel_claim` (placeholder stolen or row gone), the code falls through to `acquire_channel_claim(db, ..., session_id=new_sid)`. The return pair `(ok, claim)` is discarded. If the row was taken by another instance, `ok` is `False` and a foreign claim exists; the session proceeds without a claim. The next admission will not route into it, and the channel is briefly unprotected.

**Why it matters:** A sub-millisecond race window in high-concurrency burst admission leaves a live session without a SQLite claim. The lease expiry (~300 s) eventually heals it, but during the window two threads could create two sessions.

**Suggested fix:** Check the return value. If not `ok` and a foreign claim is present, log a `logger.warning` so operators can detect the race, and consider standing down this admission.

**`tools/bot_relay.py:796` — `release_relay_claim` silently skips when `_read_relay_claim` returns `None`.**

If the claim file is corrupted (same Major root cause), the release function does nothing. The file stays on disk with its old `lease_expire`. The caller `_finish_route_transition` logs a debug line and moves on. Because the relay claim is *not* renewed during the turn, the file could persist past its 300-second lifetime if the gateway resets before expiry and never re-acquires it for that channel. On the next gateway boot, the stale file is treated as a live foreign claim for up to the remaining lease.

**Why it matters:** Bounded by 300 s and a rare path, but it means a release failure (caused by the same corruption) silently extends the relay claim lifetime, not shortens it. A stale file can block a new session slightly longer.

**Suggested fix:** In `release_relay_claim`, if `existing` is `None` but `path.exists()`, log a warning and remove the file rather than silently leaving it. A `None` read is sufficiently diagnostic that the file is garbage.

### Nit

*None.*

---

## Architectural Summary of Residual Risk

1. **Mid-turn lease expiry:** There is no in-turn renewal. A turn longer than 300 s allows a peer gateway to steal the SQLite claim and start a parallel session. This is an accepted design parameter (the turn loop is not instrumented for mid-turn SQLite writes), and chat turns rarely exceed 300 s. The risk is noted, not a defect.

2. **Shared-filesystem `O_EXCL` atomicity:** The relay claim relies on POSIX `O_CREAT | O_EXCL` across a shared directory. On NFS without proper `lockd` or on exotic networked filesystems, this is not guaranteed atomic. The feature is gated behind `enabled`; production deployments on shared storage should verify the filesystem semantics or treat the relay gate as advisory.

3. **Recovery / `_finish_channel_claim` early-return coupling:** The recovery path deliberately returns early from `_finish_channel_claim` because `_route_admit` already bound the claim. This is correct for single-gateway logic, but if a second gateway with a shared state database races the recovery admit, the first gateway will steal the stale self-claim in `_route_admit` while the recovering gateway is still in `_finish_channel_claim`. The e2e unit simulates a shared store but not a shared store *under* two live gateways; that scenario is implicitly covered by the relay claim (`O_EXCL`) plus the SQLite CAS update in `transfer_channel_claim`, but real-world validation would need an integration cluster.

4. **E2E harness fidelity:** The harness correctly pins `ClaimLockConfig(mode="enforce")`, asserts `max_channel_sessions_seen == 1`, heartbeat stream uniqueness, one answer per message, thread-root routing, queue drain, lease expiry post-kill, and cross-instance stand-down. It does not, however, simulate a corrupted relay claim file or a filesystem that lacks `O_EXCL` atomicity. These are adversarial scenarios that would require fault-injection testing.

---

*End of report.*
