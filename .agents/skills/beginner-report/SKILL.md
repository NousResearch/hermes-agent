---
name: beginner-report
description: Use when the final answer must be understandable to a non-developer operator in plain Korean, with short wording, minimal jargon, and a fixed what/why/result/next structure.
---

## Use when
- the user is a non-developer or wants simple wording
- the work is done and needs a clear final report
- the result should be copy-paste friendly

## Do not use when
- the user explicitly asks for detailed code-level explanation only
- the main task is implementation with no reporting yet
- the output must be raw machine-readable JSON only

## Input template
```text
이번 작업은 `$beginner-report` 방식으로 보고해줘.
목표: [설명받을 작업]
필수형식: what/why/result/next
```

## Procedure
1. Use Korean by default.
2. Keep wording concrete and short.
3. Explain technical terms in one short line only when needed.
4. Prefer the fixed shape: what, why, result, next.
5. Include copy-paste commands only when the user must act manually.
6. Avoid large code dumps unless the user asked for code.
7. Separate verified facts from inference for OpenAI or Codex behavior.

## Output contract
- what
- why
- result
- next

## Negative example
- Wrong fit: "이 함수의 시간복잡도를 증명해줘."
- Reason: this needs a technical explanation, not an operator report.

## Example prompt
```text
이번 작업은 `$beginner-report` 방식으로 보고해줘.
목표: 오늘 적용한 Codex 설정 설명
필수형식: what/why/result/next
```
