# Wave 1 — learning/evidence researcher digest

## Verified findings

- The evidence database has additive schema handling for verification events,
  session state, a monotonic workspace edit marker, and outcome receipts.
  Raw goal text is represented only by a SHA-256 digest.
- Same-session/same-workspace confirmation is idempotent and current evidence
  must still pass before a receipt is reusable. Workspace edits invalidate
  older receipt evidence across sessions.
- Receipt retrieval is not connected to memory, prompt construction, skills,
  or automatic action. Its only existing uses were docs and tests.
- Evidence events have retention limits, while outcome receipts persist and
  naturally drop out of the reusable view once their current evidence can no
  longer pass.
- A terminal command that edits files may not update the file-tool stale
  marker; this is a separately bounded, unverified E2E lead and is excluded
  from the current change.

## Worker EXPAND marker (verbatim)

none — 학습·증거·freshness·스키마·메모리 routing·테스트·Git 이력의 발견 리드를 모두 추적했고, 남은 것은 위의 명시적 읽기 전용 구현 후보입니다.
