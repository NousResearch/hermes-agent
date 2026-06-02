---
name: change-review
description: Use after edits when you need Codex to inspect the diff, run the narrowest relevant verification, review for regressions and missing tests, and report rollback information for risky changes.
---

## Use when
- edits are already made
- you want a disciplined post-change check
- config, scripts, or automation changed
- you want risk and rollback called out clearly

## Do not use when
- no files were changed yet
- the task is only to plan future work
- the request is a pure explanation with no local changes

## Input template
```text
이번 작업은 `$change-review` 방식으로 진행해줘.
목표: [방금 한 변경 점검]
완료조건: [검증 통과 / 리스크 정리]
검증방법: [테스트/확인 명령]
```

## Procedure
1. Inspect the changed files with the best available diff command.
2. Run the narrowest relevant verification command first.
3. If a broader test exists and is still relevant, run it after the narrow check.
4. If the change is non-trivial, prefer a separate checker lane by default.
5. Review for regressions, hidden side effects, and missing tests.
5. Call out rollback paths when config or automation changed.
6. If no test can be run, say so explicitly and explain why.

## Output contract
- changed scope
- verification run
- findings or residual risk
- rollback path when needed

## Negative example
- Wrong fit: "프로젝트 구조를 설명해줘."
- Reason: there are no edits to review.

## Example prompt
```text
이번 작업은 `$change-review` 방식으로 진행해줘.
목표: 방금 수정한 설정 점검
완료조건: 영향 범위와 검증 결과 정리
검증방법: 관련 명령 실행 + diff 확인
```
