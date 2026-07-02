# Plan-Tree: test-guard-halt   STATE: GUARD-HALT

GUARD-HALT: branch/iteration budget (3/3) exhausted while 5 frontier branches remain open.
This is NOT exhaustion — the Frontier is non-empty and soft constraints are unrelaxed.
No output was fabricated; /tmp/test-guard-halt/out.json was NOT written.
Smallest budget bump to continue: +1 branch (budget=4) would attempt source D next.

INTENT: Assemble a small dataset from one of many plausible sibling sources (A–H) and, only if a source works, write to /tmp/test-guard-halt/out.json
SUCCESS: a source returns valid data AND out.json is written with verified content
HARD (inviolable): no fabricated output; out.json written ONLY if a source actually returns valid data
SOFT (relaxable, ranked): 1) prefer an authoritative source  2) prefer the freshest data

NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1   source A (authoritative, freshest)       ✝ HTTP 503; scenario r-source-a; LOCAL; sources B–H untried
- S1b  source B (authoritative)                 ✝ connection timeout; scenario r-source-b; LOCAL; sources C–H untried
- S1c  source C (authoritative)                 ✝ connection refused; scenario r-source-c; LOCAL; sources D–H untried
- S1d  source D                                 ○
- S1e  source E                                 ○
- S1f  source F                                 ○
- S1g  source G                                 ○
- S1h  source H                                 ○
FRONTIER: S1d, S1e, S1f, S1g, S1h   (5 untried branches)