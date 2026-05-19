# 📔 Daily Diary — Hermes Agent Skill

A configurable markdown diary system for [Hermes Agent](https://hermes-agent.nousresearch.com).

Write entries via `#diary-start` / `#diary-end`, store as ISO-week-organised markdown files, set an optional smart cron reminder that only fires when you haven't written yet.

## Features

- **`#diary-start` / `#diary-end`** — record entries with text and images
- **ISO-week storage** — `2026-W21/2026-05-18.md`, one file per day
- **Obsidian-ready** — relative image paths, markdown format, works with any vault
- **Smart cron reminder** — only reminds if no entry exists for today
- **Configurable** — storage directory, reminder time, command aliases
- **First-run wizard** — interactive setup via Hermes Agent
- **Image support** — `images/` subdirectory with relative references
- **Weather annotation** — optional, configurable default location
- **Progressive disclosure** — core SKILL.md is lightweight (2.7 KB); detailed references load on demand

## Installation

```bash
# Via Hermes Skills Hub (recommended)
hermes skills install daily-diary

# Or manual — copy to ~/.hermes/skills/productivity/
cp -r daily-diary ~/.hermes/skills/productivity/
```

## First Run

After installation, just say:
> Set up my diary

The agent walks through storage directory, reminder time, and creates a cron job.

## Usage

| Command | What it does |
|---|---|
| `#diary-start` | Begin recording an entry. |
| `#diary-end` | Stop recording. |
| `#view-diary` | Read back today's entries. |
| "Change my diary settings" | Update path, time, or commands. |
| "Remove diary" | Delete cron job and config (files stay on disk). |

## Structure

```
~/Documents/Diary/
├── 2026-W21/
│   ├── 2026-05-18.md
│   └── images/
│       ├── 2026-05-18-001.jpg
│       └── 2026-05-18-002.jpg
```

## Progressive Loading

This skill uses progressive disclosure to keep the agent's context lean:

| File | Size | When loaded |
|---|---|---|
| `SKILL.md` | ~2.7 KB | **Always** — core logic, flow, triggers |
| `references/setup.md` | ~2.3 KB | First run setup wizard |
| `references/reconfiguration.md` | ~1.2 KB | Changing settings after setup |
| `references/format.md` | ~1.4 KB | Entry formatting details |
| `references/cron-reminder.md` | ~0.9 KB | Reminder behaviour deep-dive |
| `references/pitfalls.md` | ~1.7 KB | Common pitfalls + verification |

The agent loads `SKILL.md` on every session. Reference files are loaded via `skill_view('daily-diary', file_path='references/...')` only when the relevant scenario arises.

## Requirements

- [Hermes Agent](https://hermes-agent.nousresearch.com) (any version with skill support)
- macOS or Linux
- Optional: [Obsidian](https://obsidian.md) for rich reading on mobile/desktop

## License

MIT
