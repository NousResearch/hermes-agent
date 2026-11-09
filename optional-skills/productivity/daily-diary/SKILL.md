---
name: daily-diary
description: >-
  Daily diary system — record entries via #diary-start / #diary-end commands,
  store by ISO week in markdown files, optional cron reminder that skips if
  already written. First-run setup wizard walks through directory and
  reminder time.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Diary, Journal, Note-Taking, Cron, Productivity]
    related_skills: [obsidian]
    requires_toolsets: [terminal, file]
    category: productivity
---

# Daily Diary — Markdown Journal

> **This skill is read by the AI agent (Hermes), not by the human user.**
> Write instructions in the imperative, targeting the agent.

## When This Skill Activates

When the user says **any** of these, load this skill and respond accordingly:

| User says (examples) | Your task |
|---|---|
| `#diary-start` | Begin recording a diary entry. |
| `#diary-end` | Stop recording, confirm save. |
| `#view-diary` | Read back today's entry. |
| "set up my diary" / "帮我设置日记" / "init diary" | Run the first-time setup wizard. |
| "change diary" / "reconfigure diary" / "改提醒" | Run reconfiguration workflow. |
| "remove diary" / "delete diary" / "移除日记" | Remove cron job + clear config from memory. |

If the user says `#diary-start` but memory has no `diary_storage_dir`, run the
setup wizard first instead of trying to write.

## How to Load Reference Files

When a scenario requires detail beyond this core file, load the reference:

```
skill_view('daily-diary', file_path='references/setup.md')
```

Available references:

| Scenario | Load this |
|---|---|
| User wants first-time setup (no config in memory) | `references/setup.md` |
| User wants to change path, time, or remove diary | `references/reconfiguration.md` |
| You need to know the exact markdown format or image layout | `references/format.md` |
| You need the cron job's smart-skip logic details | `references/cron-reminder.md` |
| You hit an error — check common pitfalls | `references/pitfalls.md` |

## Daily Mode (diary_storage_dir is set in memory)

### Starting an entry (`#diary-start`)

1. Read `diary_storage_dir` from your memory block (format: `diary_storage_dir=/path/to/dir`)
2. Determine today's ISO week: run `date +%V` → yields `Www` (e.g. `W21`)
3. Build directory: `{diary_storage_dir}/{YYYY-Www}/`
4. Build file: `{diary_storage_dir}/{YYYY-Www}/{YYYY-MM-DD}.md`
5. If this date already has a `# YYYY-MM-DD` heading in the file → append a new `## HH:MM` section
6. If new day → write the full header (`# YYYY-MM-DD DayOfWeek`, weather blockquote, `## HH:MM`)
7. Optionally fetch weather via `curl -s wttr.in/{location}?format=3` (default: Beijing, configurable via memory `diary_default_location`)
8. Save images to `{diary_storage_dir}/{YYYY-Www}/images/` with filename `{YYYY-MM-DD}-NNN.jpg`
9. Tell the user: "Diary started. Send me your entry."

### Receiving content during an entry

- **Text** → append under the current `## HH:MM` section
- **Image** → save to `images/` dir, reference as `![caption](images/YYYY-MM-DD-NNN.jpg)` in markdown
- **No vision model** → save the image file but note: "Image saved — I can't see its contents with this model."

### Ending an entry (`#diary-end`)

Confirm the file path and entry count, then close the recording session.

### Viewing entries (`#view-diary`)

Read `{diary_storage_dir}/{YYYY-Www}/{YYYY-MM-DD}.md` and return its content.

## Memory Format

All diary config is stored in your persistent memory block as key-value pairs
separated by semicolons:

```
diary_storage_dir=/path/to/dir; diary_reminder_time=21:00;
diary_reminder_job_id=abc123; diary_start_command=#diary-start;
diary_end_command=#diary-end; diary_initialized=true;
diary_default_location=Beijing
```

When reading, split by `;` then by `=`. When writing, use the `=;` format.
