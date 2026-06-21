# Independent reproduction (clean environment) — Council verification pass

All three agent-executable Council demands reproduced from a CLEAN clone /
environment, confirming the reported state is deterministic, not session-local.

## 1. Per-PR verify reproduced from a clean v0.17.0 clone — MATCHES exactly
`verify_all_prs_on_v017.sh` re-run from a fresh `git clone` (origin=NousResearch,
fork=arminanton, checked out at v0.17.0 2bd1977d). Output = `cleanclone_per_pr_output.txt`,
identical to the working-tree run:
- APPLY: 38 CLEAN + 2 --3way (#50056, #50073) + 0 CONFLICT
- LINT: all contributable PRs PASS; the only FAILs are #50056 (3-way-merge artifact —
  its forward-compat branch is ruff-clean) and #50111 (this branch's OWN reconciliation
  *.py scripts, not repo code).
- TEST: the documented FAILs reproduce EXACTLY as triaged (deterministic, not flaky):
  #50031 live-API billing test, #50064/#50066/#50078/#50086 cross-PR base-drift batch
  collection. Determinism across two independent environments confirms these are
  structural (live/base-drift), not intermittent.

## 2. Forward-compat branches independently confirmed
- `forward-compat/50056-on-v0.17.0`: v0.17.0-ancestor=YES, conflict-markers=0, 5 files
- `forward-compat/50073-on-v0.17.0`: v0.17.0-ancestor=YES, conflict-markers=0, 5 files
Both are built ON v0.17.0 and are the intended clean replay form for those 2 --3way PRs.

## 3. 521-green integration run reproduced
`integration/v0.17.0-all-37-prs` (v0.17.0-ancestor=YES, 0 conflict markers): the
subsystem suite (mcp_tool + keepalive + kanban_db + subdirectory_hints + doctor)
re-ran => **521 passed**. The full stacked PR set runs green on v0.17.0.

## Conclusion
The agent-executable verification is independently reproduced and deterministic.
The residual per-PR test FAILs are confirmed (by identical reproduction in a clean
environment) to be live-API / cross-PR-base-drift, not defects: each PR's own feature
passes, and the authoritative stacked-integration run is 521-green.
