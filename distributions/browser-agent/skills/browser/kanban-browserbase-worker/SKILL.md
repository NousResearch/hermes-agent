---
name: kanban-browserbase-worker
description: >
  Use when the browser-agent Kanban profile picks up a task tagged `browser`.
  Defines the full execution loop: claim the task, run Browserbase browser
  automation, verify the result visually, and complete/block on the board.
version: 1.0.0
author: Kyle Jeong / Hermes
license: MIT
metadata:
  hermes:
    tags: [kanban, browserbase, browser, worker, automation]
    related_skills: [kanban-codex-lane, browserbase-agent]
---

# Kanban Browserbase Worker

This skill governs how the `browser` Hermes profile executes Kanban tasks.
It is the browser-side counterpart to `kanban-codex-lane`.

## Trigger

Load this skill when:
- You are running as the `browser` Hermes profile
- A Kanban task has been dispatched to you (assignee = `browser`)
- The task body describes a web automation, scraping, form-fill, or verification job

## Execution Loop

### 1. Parse the task

Read the task title and body. Extract:
- **Target URL(s)** — where to navigate
- **Action** — scrape | click | fill | verify | extract
- **Success criteria** — what does done look like?
- **Output format** — raw text, JSON, screenshot, table

### 2. Set up the browser session

The `browser` toolset is pre-configured to use Browserbase cloud.
No setup required — just call browser tools directly.

```
browser_navigate(url)        → loads the page via Browserbase
browser_snapshot()           → get accessibility tree + ref IDs
browser_click(ref)           → click an element
browser_type(ref, text)      → type into a field
browser_vision(question)     → visual inspection / screenshot
browser_scroll(direction)    → scroll the page
browser_press(key)           → keyboard input
```

### 3. Execute and verify

- Always call `browser_snapshot()` before clicking to confirm element refs
- After completing the action, call `browser_vision("Did the action succeed? Describe what you see.")` to visually verify
- For data extraction, collect the structured output

### 4. Handle common failure modes

| Symptom | Action |
|---------|--------|
| Login wall / CAPTCHA | Block the task: `"Requires authenticated session — no credentials available"` |
| Element not found | Scroll and re-snapshot; try once more |
| Page loads wrong content | Screenshot + vision to diagnose; retry navigation |
| JS-heavy SPA slow to render | `browser_scroll(down)` to trigger lazy-load; wait implicitly |
| Rate limit / bot detection | Browserbase has stealth+proxies enabled — log the error and retry once |

### 5. Complete the task

On success:
- Write a clear summary of what was found/done
- Include any extracted data inline
- Mark the Kanban task complete

On failure after retries:
- Screenshot the failure state with `browser_vision`
- Write a precise failure reason
- Block the Kanban task with the failure context

## Task body format (what the Kanban creator should provide)

```
URL: https://example.com/page
Action: scrape the product prices table
Output: JSON array of {name, price, sku}
Success: table has at least 1 row
```

## Pitfalls

- Never use `terminal` tools — this profile does not have shell access by design
- Do not mark a task complete based on navigation alone — always verify the outcome visually
- Browserbase sessions are cloud-managed; do not try to manage CDP URLs manually
- If `browser_navigate` returns a crash error, the VM has no local Chromium — this is expected and means the Browserbase plugin is not active; check that `browser-browserbase` plugin is enabled in this profile's config
