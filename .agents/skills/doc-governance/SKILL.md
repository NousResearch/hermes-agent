---
name: doc-governance
description: Use when updating operational project documents such as handoff notes, indexes, policies, or rollover-managed files and you need Codex to keep docs concise, current, and non-sprawling.
---

## Use when
- a major phase ended and docs must be updated
- handoff or operational notes changed
- long-running docs need rollover checks
- document index or policy entries need maintenance

## Do not use when
- the task is only code implementation with no doc impact
- the user asks for a one-off casual answer
- the file is temporary scratch text that should not be tracked

## Input template
```text
이번 작업은 `$doc-governance` 방식으로 진행해줘.
목표: [정리할 문서 작업]
완료조건: [갱신 대상 문서 반영]
```

## Procedure
1. Update only the documents touched by the current work.
2. Keep AGENTS.md concise; move long procedures into docs or skills.
3. Before ending a major phase, run the document rollover check if the project uses it.
4. If rollover is required, archive first and then keep the live file short.
5. Update indexes or handoff notes when the work changes project operations.
6. Treat long-running docs as logs, not permanent dumping grounds.

## Output contract
- docs updated
- rollover check result
- index or handoff impact

## Negative example
- Wrong fit: "버튼 클릭 오류 고쳐줘."
- Reason: this is bug-fix workflow, not document maintenance.

## Example prompt
```text
이번 작업은 `$doc-governance` 방식으로 진행해줘.
목표: 운영 문서와 handoff 정리
완료조건: 관련 문서만 최소 갱신
```
