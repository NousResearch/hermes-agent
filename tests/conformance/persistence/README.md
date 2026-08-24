# Crash/resume persistence conformance cells

Phase 1 of the machine-checked conformance suite proposed in #80921, following
the contract framing of "Resume Means Resume" (arXiv:2608.03836): each cell is
a deterministic, LLM-free probe of one persistence contract clause, run against
the **real `SessionDB`** with a real `SIGKILL` delivered to a separate OS
process mid-write.

## Cells

| cell | contract clause | origin |
|---|---|---|
| 1 — `test_cell1_prefix_durability` | acknowledged appends survive a hard crash; contiguous prefix; deterministic recovery | adapted from the tracking issue's spot-probe (29.5K-message original, scaled to a ≥200-append kill window with identical assertions) |
| 2 — `test_cell2_consume_once` | a parked handoff is claimed by exactly one of N racing processes | adapted from the tracking issue's spot-probe (8-process file-barrier race) |
| 3 — `test_cell3_rotation_atomicity` | a compression rotation is visible entirely or not at all — never a compression-ended parent without a continuation (the #80337 orphan shape; recovery for the legacy population merged in #80487) | new in this suite |
| 4 — fork determinism on edit/rewind | recovery yields exactly the chosen prefix after a fork | **stub** — interlocked with the rewind/archive redesign (#82956–#82959) |
| 5 — delivery-outbox effect exactly-once | crash between provider send and durable record must not double-deliver on catch-up | **stub** — needs a fake-transport seam; cron delivery scope in flight (#83197/#83557) |

## Method

- Real `SessionDB(db_path=...)` in an isolated `tmp_path`; no mocks on the
  persistence layer.
- Crashes are real `SIGKILL`s to a separate interpreter, asserted to be
  **alive at kill time** (a clean early exit cannot masquerade as a crash
  test); acknowledgement journals tolerate a torn final line (the kill can
  interrupt the journal write itself).
- Every wait is deadline-bounded; coordination uses file barriers, never
  sleeps-for-correctness.
- Journal-mode matrix (cells 1 and 3): the resolver's default, explicit
  `DELETE`, and explicit `WAL` — each leg steers the child's own resolver
  via an isolated `HERMES_HOME` config, then **audits the on-disk mode after
  the run** and skips when the environment didn't honor the request (e.g.
  the resolver's WAL-reset downgrade gate, the tracking issue's 3.50.4
  caveat). A leg that ran in a different mode never counts as evidence for
  the advertised one. Cell 2 runs on the resolver's default only (the
  consume-once property is journal-mode-independent: it rests on a single
  predicated UPDATE).

## Semantics

These are **conformance** cells: they are expected GREEN on main (cells 1–2
reproduce the tracking issue's passing probes; cell 3 pins the atomicity the
#80337 forensics established). A failing cell is a **fire**: report it on
#80921 with the cell's evidence — do not silence it, and do not attach a fix
to this suite.
