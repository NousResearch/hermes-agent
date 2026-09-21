#!/usr/bin/env python3
"""Template for generating cron prompts with Discord-compatible formatting.

Use this to create consistent cron prompts. All examples use Discord Markdown,
not Telegram HTML.

Template:

Line 1: {{emoji}} **{{Name}}** · DD Mon · HH:MM

**Findings**
• item 1 — detail
• item 2 — detail

**Actions**
• what to do, with `command` if applicable

**Full report**
`/home/kensei/.hermes/runbooks/{{cron-name}}/YYYY-MM-DD/{{name}}.html`

Mandatory rules for ALL cron prompt output:
1. Only Discord-safe Markdown (bold via `**text**`, code via `` `code` ``)
2. NO HTML tags (`<b>`, `<code>`, `<blockquote expandable>`)
3. All detail ONLY in the HTML file
4. Always create directory and HTML file before sending
5. Wrap every ID, path, command in backticks  (`` ` ``)
6. UK date format: DD/MM/YY HH:MM:SS
7. Prefix timestamp: {name} · DD/MM/YY HH:MM:SS
8. Shortened anchor links: name — url (not full HTML anchors)
9. If no interesting findings: output [SILENT] only


Section: **Findings**
- Contains 2-3 top picks with `url` and one-sentence relevance notes
- Each finding: `title` — one-line pitch. Why now: [strategic rationale]


Section: **Actions**
- If audit finds issues: file kanban tasks with `hermes kanban create --triage`
- Use Findings section for the filed tasks
- Put full audit results in the HTML file


Section: Backlog promotion pattern
1. Call `backlog_list(state="raw", limit=1)`. If empty, try `state="deferred"`. If still empty, output [SILENT].
2. Pick the lowest-hanging item (or oldest).
3. Write a brief migration plan: what needs changing, which files, estimated effort.
4. Call `backlog_update(item_id=<id>, state="ready")` to promote it.
5. Include the migration plan in your output.


Section: Kanban review pattern
1. List completed tasks: `kanban_list(status="completed", limit=5)`.
2. Review each completed task: check its children, verify they actually completed.
3. Check output paths and verify files exist.
4. Approve passing tasks (kanban update to done) and block failing ones with feedback.
"""

if __name__ == "__main__":
    print(__doc__)