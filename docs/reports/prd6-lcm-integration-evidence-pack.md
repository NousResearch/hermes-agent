# PRD-6 LCM Context Engine — Integration Evidence Pack (t80)

Generated: 2026-06-16 15:59 PT
Branch: `prd6-lcm/integration` @ `e4086c2dd`
Base: descends from `origin/main` (verified `git merge-base --is-ancestor origin/main HEAD`)
Files changed vs origin/main: 48 (45 added, 3 modified — host double-wrap fix only)

## Lane → commit → tests (every lane Apollo-verified)

| Lane | Commit | Tests | Evidence |
|------|--------|-------|----------|
| t00 vendor LCM v0.16.2 | 9b9143509 | 36 | loader maps `lcm`→`LCMEngine` (7 lcm_* tools); host double-wrap fix |
| t10 adoption smoke + ABC drift | 52fcf1c6e | 10 | probe 7/7 |
| t20 fail-open degraded telemetry | 7a7ce2e82 | 11 | I-1 over-limit ContextCompressor fallback + LCMFailOpenRecoveryError |
| t30 storage/redaction/encryption | 5b716c6ee | 13 | I-6/I-8 redaction corpus + retention status/doctor (Apollo finished wiring) |
| t40 live-recovery harness | 41ddc862f | 6 | N=180 gate, Wilson lower-bound BINDING (spec bug 0.95→0.90 caught+fixed) |
| t50 benchmark battery | a387eea62 | 4 | 3 arms raw/compressor/lcm, GO/NARROW-GO/NO-GO verdicts |
| t60 Aegis stress + config preflight | 65b5c9454 | 10 | sha256 artifact drift gate; **zero live-restart code (safety-verified)** |
| t70 upstream drift checker | e4086c2dd | 5 | I-10 provenance PASS/WARN/FAIL, append-only metadata |

## Combined test matrix (run by Apollo on integration tip)
- `tests/context_engine` (12 lane suites): **48 passed**
- Neighbor matrix (context_engine loader + host contract + plugin init + context dir): **81 passed**
- `git diff --check`: clean · `git status -sb`: clean on prd6-lcm/integration

## PRD acceptance-criterion → evidence map
- Lossless recall / live model recovery → t40 harness (live-recovery gate, N≥180, Wilson LB≥0.90 binding)
- Fail-open never blocks a turn → t20 (degraded telemetry + over-limit fallback)
- Data-at-rest / redaction → t30 (encryption + redaction corpus over raw/DAG/summary/FTS)
- Cross-gateway artifact/config drift → t60 (sha256 preflight)
- Upstream security drift → t70 (offline checker, Apollo-owned cadence)
- Benchmark GO/NO-GO → t50 (3-arm battery)

## Remaining LIVE gates (NOT in this branch — require explicit Ace go)
- Aegis activation (privileged): enable LCM context engine on aegis profile
- Apollo cutover (high blast radius): after Aegis stress/E2E pass
- Push to fork remote: not done; branch is local-only
- No profile config or gateway-restart code is committed to this branch.

## Lane commits (no-merge)
```
e4086c2dd feat(lcm): add upstream drift metadata checker
65b5c9454 feat(lcm): add Aegis stress and config preflight gates
a387eea62 feat: add LCM benchmark battery
41ddc862f fix(lcm): Wilson lower-bound gate default 0.95 -> 0.90 per PRD-6
f7fd96ab5 feat(lcm): add live recovery statistical harness
5b716c6ee feat(lcm): storage security, redaction corpus, encryption, retention
7a7ce2e82 Implement LCM fail-open degraded fallback
52fcf1c6e test: add isolated LCM smoke gate
9b9143509 feat: vendor LCM context engine
```
