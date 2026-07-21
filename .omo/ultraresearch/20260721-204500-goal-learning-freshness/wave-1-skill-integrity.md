# Wave 1 — skill proposal integrity

## Digest

- Skill pending replay drops the original trusted background provenance.  A
  staged background proposal can therefore lose agent-created attribution and
  bypass the background ownership/pin/external/bundled checks at approval.
- `/skills diff` reads live state and uses a different matcher than mutation;
  it is not a static review artifact.  Every one of the six mutation actions
  needs a distinct snapshot and a conditional-write boundary, so Skills V2 is
  not a safe single-action addition in this change.
- The bounded, independently useful repair is record-aware replay that
  preserves the stored provenance.  Full static artifacts and CAS remain a
  separately designed feature.

## Evidence

`tools/write_approval.py:211-219,625-688`,
`hermes_cli/write_approval_commands.py:154-166`, and
`tools/skill_manager_tool.py:297-435,777-806,985-1033,1323-1439`.
Focused baseline: `tests/tools/test_write_approval.py` = 63 passed, 1 skipped;
the wider skill suite has two stale post-auto-stage expectations and one
Windows symlink-permission setup failure.

## EXPAND

- LEAD: background-stage 이후 replay가 original origin을 잃어 pinned skill patch와 agent-created marking을 우회함 — WHY: AI-agent learning ownership/approval 경계를 현재 직접 위반함 — ANGLE: record-aware replay API와 origin-preserving regression tests 설계.
- LEAD: `skill_pending_diff`가 stage snapshot이 아닌 live disk와 단순 `str.replace`를 사용함 — WHY: approver가 본 diff와 fuzzy patch 실행 결과가 달라질 수 있음 — ANGLE: action별 static artifact builder를 actual mutation planner와 공유.
- LEAD: skill mutation layer에 target file lock 또는 conditional compare-and-swap가 없음 — WHY: stale check 뒤 concurrent writer를 덮어쓸 TOCTOU가 남음 — ANGLE: Windows/POSIX 공통 per-skill lock과 hash-validated write primitive 조사.
- LEAD: background staging으로 무효화된 legacy dispatcher tests 2개와 Windows symlink 권한 미-skip test가 현재 suite를 red로 만듦 — WHY: V2 회귀 판단의 baseline이 신뢰할 수 없음 — ANGLE: stage→approve expectation으로 test 재작성 및 WinError 1314 conditional skip 추가.
