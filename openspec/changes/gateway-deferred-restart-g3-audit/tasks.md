## 1. G3 definition paths (layer 1)

- [x] mention-level deferred-restart detection (no inert-heredoc stripping:
      a data heredoc that mentions the command is refused too — blocks
      write-then-run `cat > child.sh` generation)
- [x] per-carrier scanning: prompt, monitor source, referenced script paths are
      each scanned independently; combined-query splitting prevents cross-carrier
      splicing from slipping between carriers
- [x] referenced scripts resolve under the ACTIVE home binding (routed fire's
      profile home), never a caller-default home
- [x] create/edit/import/re-enable entrypoints refuse with a clear error

## 2. G3 dispatch pre-execution (layer 2)

- [x] `check_deferred_restart_job` at the real dispatch entry, fail-closed,
      covering pre-existing jobs (agent / script / monitor shapes)
- [x] refused job books a failed execution with the named reason and reaches
      zero user-code/model side effects (`run_job` sentinel tests)

## 3. G3 runtime control point (CLI entrypoint)

- [x] trusted cron provenance (`HERMES_CRON_SESSION` truthy, external-worker
      marker) refuses at `cmd_defer_restart` with exit 7 before any request
      file write or signal
- [x] real-chain test: bash variable assembly → real entrypoint → refused in
      cron context (zero side effects), accepted in the non-cron control
      (request + signal boundaries reached)

## 4. Audit trail

- [x] append helper hardening (G2-R4): os.write length checked (short write is
      never success); crashed-writer incomplete tail isolated before the next
      append (earlier complete events and the next event stay readable); safe
      open — O_NOFOLLOW, regular-file + owner verification, over-open
      pre-existing files tightened to 0600
- [x] placement rule (G2-R4): decision audit fires only after the authoritative
      result is persisted; a failed acceptance write that cancels the schedule
      emits no premature accepted event
- [x] guarantee scope documented (G2-R3): attempt + report + result-authoritative
      + gaps allowed; no no-gaps completeness wording anywhere
- [x] bounded actor_ref (≤256 chars, else empty) — attribution only, never
      authorization; invalid actor leaves the audit field empty (malformed
      request is rejected outright)

## 5. Tests

- [x] tests/cron/test_deferred_restart_guard.py — definition/dispatch/path-binding
      + real-dispatch zero-side-effect suite
- [x] tests/hermes_cli/test_gateway_defer_restart.py — real-chain dynamic
      assembly (cron refused / control allowed)
- [x] tests/gateway/test_defer_restart_audit.py + test_defer_restart_audit_hardening.py
      — provenance, replay, rejection, failure-isolation, partial-tail,
      short-write, permission-tightening, symlink-rejection
- [x] G1 regression: 15 fact-finding probes, verdicts byte-identical
- [x] full suite green (197 passed, 2 skipped on macOS; 2 skips are windows-only)
