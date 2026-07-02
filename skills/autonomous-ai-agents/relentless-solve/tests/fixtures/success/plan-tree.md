# Plan-Tree: demo-backtrack   STATE: SUCCESS

INTENT: produce a data file with a top-level key "ok" (network preferred)
SUCCESS: a data file exists and contains valid JSON with a top-level key "ok"
HARD (inviolable): output must be a data file with valid JSON containing key "ok"
SOFT (relaxable, ranked): 1) prefer a network source over local cache (relaxed)

NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1   alfa (primary source)     ✝ tombstone · primary network source is down · STRUCTURAL
- S1b  bravo (mirror)             ✝ tombstone · mirror network source is down · STRUCTURAL · both network siblings exhausted → relax soft constraint 1
- S2   charlie (local cache)      ✓ progress · local cache hit (sim: r-charlie) · opened delta
- S3   delta (verify)             ✓ success · verified valid JSON with key ok (sim: r-delta)
FRONTIER: (empty)