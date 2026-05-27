# Hermes Agent Audit Findings

## Previous Audit (R1) — RESOLVED

All findings from the previous audit have been addressed. The corresponding PRs were opened against `NousResearch/hermes-agent`:

- **N-series**: N2–N27 (security, stability, correctness fixes)
- **T-series**: T1–T11 (shell injection, unsafe YAML, async blocking, unbounded queues, ffmpeg timeout, orphaned tasks, memory integrity)
- **N42/N43**: Gateway O(n²) watcher recovery and silent plugin errors
- **N26/N27**: Approval revocation enforcement and YOLO mode bypass

### Status
All 30+ PRs submitted to upstream. Some marked as duplicates by maintainer (alt-glitch) pointing to canonical PRs. Awaiting merge decisions from maintainers.

---

## New Audit — IN PROGRESS

Fresh scan starting from latest upstream/main.

Date: 2026-05-27
Base: `2d5dcfabc` (upstream/main)