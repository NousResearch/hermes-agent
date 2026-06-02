---
name: issue-to-fix
description: Use when the user reports a bug, regression, or broken behavior and you need Codex to narrow the ownership area, apply the smallest defensible fix, and verify the reported failure path first.
---

## Use when
- there is a bug report or failure symptom
- behavior changed unexpectedly
- the user wants a fix, not only diagnosis
- reproduction and regression checks matter

## Do not use when
- the task is a brand new feature with no failure to reproduce
- the user only wants a code review after changes
- the request is only to summarize an issue ticket

## Input template
```text
이번 작업은 `$task-kickoff` + `$issue-to-fix` 방식으로 진행해줘.
목표: [고칠 문제 1개]
완료조건: [문제 해결 기준]
제약: [건드리면 안 되는 것]
검증방법: [재현/확인 명령]
```

## Procedure
1. Restate the failure in plain words.
2. Check whether the issue is reproducible from local evidence.
3. Identify the smallest likely ownership area in code, config, or docs.
4. Make the smallest defensible fix.
5. Verify the reported failure path first.
6. If the fix is non-trivial, assign `review` or `tester` as a separate checker lane by default.
7. Verify nearby behavior that could regress.
8. Report root cause, fix, evidence, and remaining risks.

## Output contract
- failure summary
- likely ownership area
- fix applied
- verification evidence
- remaining risk

## Negative example
- Wrong fit: "새 결제 화면을 만들어줘."
- Reason: this is feature delivery, not failure-driven repair.

## Example prompt
```text
이번 작업은 `$task-kickoff` + `$issue-to-fix` 방식으로 진행해줘.
목표: 저장 버튼이 동작하지 않는 문제 수정
완료조건: 저장 성공 확인
제약: 다른 인증 로직은 건드리지 말 것
검증방법: 저장 후 다시 열어 값 유지 확인
```
