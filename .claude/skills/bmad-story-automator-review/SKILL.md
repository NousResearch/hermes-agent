---
name: bmad-story-automator-review
description: 'Runs the autonomous code review flow used by story automator sessions, including auto-fix handling and sprint-status sync. Use when the story automator asks for a non-interactive review of a story.'
---

1. Read `./workflow.yaml`.
2. Then read `./instructions.xml`.
3. Use `./checklist.md` as the validation checklist.
4. Follow the workflow deterministically. If the invocation asks for automatic fixes, apply them without pausing for manual menus.
