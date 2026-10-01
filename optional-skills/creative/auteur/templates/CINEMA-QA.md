# CINEMA-QA — <project>

> Fill every row with PASS/FAIL + evidence (metric value or screenshot filename). Any FAIL loops back to its phase. Ship only on all-PASS.

| # | Check | PASS/FAIL | Evidence |
|---|-------|-----------|----------|
| 1 | slopscan exit 0, all suppressions carry real reasons | | paste the `Summary:` line verbatim, from a run AFTER the last edit |
| 2 | Screenshot journey reviewed frame-by-frame (390/768/1440) | | shots dir + frames flagged→fixed |
| 3 | No text overflow / ugly wraps at any breakpoint | | e.g. "fixed S4 h1 at 390" |
| 4 | No blank / half-fired scenes in any frame | | |
| 5 | Adjacent scenes differ in layout & motion family | | frame pairs compared |
| 6 | Exactly one peak (intensity ≥8) on the built page | | scene # |
| 7 | Scrub replays cleanly scrolling UP | | manual pass |
| 8 | Reduced-motion cut watchable end-to-end, nothing blank | | rm- frames |
| 9 | Body contrast ≥4.5:1 (large ≥3:1, placeholders ≥4.5:1) | | measured pairs |
| 10 | LCP <2.5s (throttled) | | value |
| 11 | CLS <0.1 | | value |
| 12 | INP <200ms | | value |
| 13 | Hero video ≤2MB; poster ≤300KB; frames ≤150KB | | sizes |
| 14 | Motion budget ≤3 families, matches commit-sheet | | list |
| 15 | Keyboard: focus visible & designed, no traps, ESC works | | manual pass |
| 16 | No-JS: content readable, nothing hidden behind reveals | | manual pass |
| 17 | No console errors during full journey | | shoot.mjs summary |
| 18 | Mobile: pinned scenes shortened/unpinned, assets 720p | | |
| 19 | Sound (if any): off by default, gesture-gated, toggle visible | | or n/a |
| 20 | Watched the film: one slow + one fast full scroll, no felt jank | | |
| 21 | Perf measured at DPR 2 on the production build, not a dev server | | motionqa line, verbatim |
| 22 | Reference diff done: own frame vs the recon reference, greyscaled too | | what it did better / what changed / what was left |
| 23 | chromadiff vs the reference passes (no cheerful drift) | | paste the chromadiff line |
| 24 | Background lightness within ±0.12 of commit-sheet §2 | | `--target-l` line |
| 25 | The two house tells named in commit-sheet §7 are actually broken | | which two, and what replaced them |

**Verdict:** SHIP / LOOP BACK TO PHASE …
