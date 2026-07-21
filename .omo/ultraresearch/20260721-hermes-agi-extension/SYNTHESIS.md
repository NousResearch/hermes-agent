# Ultraresearch Synthesis: Hermes Goal-Learning Contract Integrity

Workers: 3 roles · Waves: 3 · Verification gates: focused tests, compile,
lint, diff, and independent review.

Hermes now binds a learning candidate to the exact final completion criteria
that were evaluated, without retaining raw criteria text. It also invalidates
old workspace verification after any attempted foreground terminal execution,
including timeout and final error outcomes. A subsequent recognized verification
can establish fresh proof.

Confirmed by local execution: legacy v4 receipt DBs migrate to v5; canonical
contracts are deterministic; ordered subgoals affect the digest; terminal
success, failure, timeout, and error paths stale old evidence; 147 targeted
tests pass. The change adds no automatic memory write, self-modification,
prompt injection, or background-process provenance claim.

Remaining scope boundary: a plain SHA-256 digest avoids raw criterion storage
but does not promise secrecy for guessable criteria, and background process
write provenance remains separate work.
