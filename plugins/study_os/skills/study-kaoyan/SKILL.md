---
name: study-kaoyan
description: Guide 考研 learning with StudyOS.
platforms: [linux, macos, windows]
---

# StudyOS 考研 Domain Pack

Use this domain pack with StudyOS projects whose `domain_pack` is `kaoyan.v1`.
Before long project-specific reasoning, call `study_prompt_context` with the
active intent and `domain_pack="kaoyan.v1"`. Treat fragments as turn-local
context only; never mutate system prompts mid-conversation.

## Default Project

The first-slice default is `kaoyan-2027`:
- title: `2027 考研学习计划`
- exam_type: `考研`
- exam_date: `2027-12-20`
- phase: `foundation`
- subjects: 数学, 英语一, 政治

## Planning Heuristics

- Build schedules from curriculum and 考点, not motivational prose.
- Keep subject blocks explicit: 数学, 英语, 政治, and any user-added major course.
- Foundation phase emphasizes definitions, formulas, textbook examples, and
  representative exercise-book problems.
- Review phase emphasizes 错题 clusters, weak prerequisites, and timed practice.

## Workflow Routing

Default 考研 workflows: `study-plan`, `study-organize`, `study-review`,
`study-assessment`.

Use `study-teach` only for explicit concept teaching or missing prerequisites.
Use `study-lesson` only for explicit diagrams/visuals or geometry, process
flow, timeline, or state transitions. Use `study-grill` only for strategic
study decisions, never routine schedules, 整理, 复习, or 错题 remediation.

## Math Examples

For 高数 planning, curriculum entries should map textbook sections to 考点 such
as 导数定义, 微分, 泰勒展开, 极限, 积分, and series. Representative problems can
come from 张宇, 李永乐, 660, 1000题, or user-provided sources.

## Constraints

- Stay domain-neutral in storage: 考研 is a domain pack, not a separate plugin.
- Persist plans as StudyOS project manifests, curriculum JSON, and validated
  schedule artifacts.
- Desktop calendar remains read-only and renders only validated JSON.
