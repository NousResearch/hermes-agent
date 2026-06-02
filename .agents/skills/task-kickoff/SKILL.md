---
name: task-kickoff
description: Use when starting implementation, debugging, refactoring, or config work and you need Codex to lock scope, inspect the current state with commands, and define the smallest safe verification plan before editing.
---

## Use when
- the task has not started yet
- the user wants code or config changes
- the problem looks broad and needs scope control
- you need a command-based current-state snapshot before editing

## Do not use when
- the user only wants explanation or translation
- the work is already in the review stage after edits
- the user asks for brainstorming only with no execution

## Input template
```text
이번 작업은 `$task-kickoff` 방식으로 진행해줘.
목표: [끝낼 작업 1개]
완료조건: [성공 기준 1~2개]
제약: [건드리면 안 되는 것]
검증방법: [테스트/확인 명령]
```

## Procedure
1. Restate the goal in one sentence.
2. Inspect the current state with commands before proposing changes.
3. Write down assumptions and scope fence before editing.
4. Run a subagent eligibility check right after scope lock.
5. If the task is non-trivial and has an independent reader, checker, docs, review, or isolated edit lane, add `$subagent-orchestration` by default.
6. If the task stays single-lane, name the reason explicitly: `trivial`, `risky/destructive`, `blocked critical path`, `overlapping write scope`, or `runtime hard limit`.
7. Identify the smallest safe change that can satisfy the request.
8. Define verification before editing anything.
9. If the task is large, split it into 2-5 concrete steps and execute in order.
10. Prefer command evidence over memory.
11. End the kickoff with a short path: inspect -> split-or-stay-single -> change -> verify.

## Output contract
- goal
- current state
- smallest safe change
- verification plan

## Negative example
- Wrong fit: "이 문장을 영어로 번역해줘."
- Reason: no inspection or code change workflow is needed.

## Example prompt
```text
이번 작업은 `$task-kickoff` 방식으로 진행해줘.
목표: 설정 저장 실패 원인 점검
완료조건: 수정 전 현재 상태와 확인 계획 정리
제약: 아직 파일 수정 금지
검증방법: 관련 설정 파일과 실행 로그 확인
```
