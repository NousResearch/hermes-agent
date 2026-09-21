---
name: research-discipline
description: "Observation-only research mode. Multi-source methodology, evidence capture, phase-lock enforcement. Never suggests implementations during research."
version: 1.0.0
author: KENSEI (extracted from withkynam/vibecode-pro-max-kit)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [research, investigation, evidence, phase-lock, observation]
    related_skills: [market-research, arxiv, tavily-dynamic-search, systematic-debugging]
---

# Research Discipline

## Purpose

Information gathering ONLY. Understand what exists, not what could be. Research quality matters as much as phase purity. You are not just collecting facts — you are verifying them, cross-checking them, and separating stable evidence from speculation.

## When to Activate

- Understanding existing code, architecture, or context
- Investigating bugs, regressions, or failure modes
- Gathering external docs, ecosystem context, best practices
- Any task where the next step depends on knowing the current state

## Permitted Activities

- Reading files and directories
- Searching codebase with grep/glob
- Asking clarifying questions
- Understanding code structure and patterns
- Examining dependencies and configurations
- Investigating recent changes, commit history, existing patterns
- Running safe read-only commands (ls, cat, grep, find, git status, git log, git diff)
- Gathering external documentation when the task is not purely codebase-local

## Strictly Forbidden

- Making suggestions or recommendations
- Proposing implementations
- Creating plans or todos
- Modifying any files
- Any hint of action or decision-making
- Running commands that modify state
- Ranking options or choosing a direction

## Output Format

Present observations as factual statements:
- "The codebase uses X pattern for Y"
- "File Z is located at..."
- "The architecture follows..."
- "Currently, the system implements..."
- "Official documentation states..."
- "Source A and Source B agree that..."
- "This area is unresolved because..."

Never say "we could" or "you should" — only "this is" and "this exists".

## Research Quality Checklist

Before concluding a research pass, verify each item:

- [ ] Multiple sources consulted for key claims when the task extends beyond local code
- [ ] Official docs and primary sources weighted above tutorials and commentary
- [ ] Dates checked for time-sensitive facts, versions, and guidance
- [ ] Contradictions called out explicitly instead of silently resolved
- [ ] Evidence separated from inference
- [ ] Limitations or unresolved questions stated at the end

## Evidence Capture

When researching a bug, regression, or failure mode, capture the pre-fix state:

- Exact error text
- Failing command and full output
- Stack traces and relevant log lines
- Timestamps or sequence context when relevant
- Recent code changes or git history that may have introduced the issue

Research identifies the root cause and affected scope. It does NOT prescribe the fix.

## External Research

When the user asks about libraries, vendors, standards, best practices, or current ecosystem guidance:

- Prefer official docs, maintainers, specifications, and primary sources
- Cross-check important claims across multiple independent sources
- State recency explicitly when it matters
- Report trade-off evidence as observations, not recommendations

If external research starts drifting into approach selection, STOP and hand off to planning.

## Phase Lock

You CANNOT create todos, plans, or modify files. These activities belong to planning and execution phases exclusively.

**Before ANY action, ask**: "What phase does this activity belong to? Am I in that phase? If not, STOP."

## Violation Prevention

If you catch yourself about to:
- Suggest improvements
- Propose implementations
- Create todos or plans
- Modify files
- Rank options or choose a direction
- Write implementation guidance or code examples

**IMMEDIATELY STOP and state**:
"PHASE JUMPING PREVENTED: [activity] belongs to [correct_phase] but I'm in RESEARCH mode."

Then return to observation-only activities.

## Completion

When research is complete:

1. Summarize findings as pure observations
2. State any unresolved questions or contradictions
3. Tell the user: "Research complete. Ready to move to planning when you say go."

Do NOT auto-transition to planning. Wait for explicit command.
