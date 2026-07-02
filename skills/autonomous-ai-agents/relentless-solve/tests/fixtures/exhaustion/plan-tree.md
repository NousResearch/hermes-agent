# Plan-Tree: demo-exhaustion   STATE: EXHAUSTION-STOP

INTENT: obtain the data from the only available source
SUCCESS: data retrieved from alfa (the sole source)
HARD (inviolable): must use alfa — it is the only available source
SOFT (relaxable, ranked): none declared

NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1   alfa              ✝ sim: r-alfa → tombstone (the sole source is unreachable) · STRUCTURAL · sole source, no siblings exist
FRONTIER: (empty — no alternative methods, no soft constraints to relax)