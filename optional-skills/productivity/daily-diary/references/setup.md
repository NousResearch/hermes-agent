# Setup Wizard — First Run

> **Target: AI agent.** These are instructions for Hermes, not for the human user.
> Load this file via `skill_view('daily-diary', file_path='references/setup.md')` when:
> - User says "set up diary" / "帮我设置日记" / similar, **AND**
> - Memory does NOT contain `diary_storage_dir`

## Pre-check (MANDATORY)

Before running any step below, **always check your memory block first**:

1. Scan your memory for `diary_storage_dir`
2. If found → do NOT run the wizard. Tell the user: "Diary is already set up. Do you want to change settings instead?"
3. If not found → proceed with Step 1 below.

## Step-by-step Wizard

### Step 1: Explain

Tell the user:

> I'll set up a daily diary. You'll use `#diary-start` and `#diary-end` to write entries, stored as markdown files.

### Step 2: Ask for storage directory

> Where should I save your diary files?
> (Press enter for default: `~/Documents/Diary/`)

Decision tree for suggesting a default:

- User says "Obsidian" → suggest `~/Documents/Obsidian Vault/Diary/`
- User says "iCloud" → suggest `~/Library/Mobile Documents/com~apple~CloudDocs/Diary/`
- Otherwise → suggest `~/Documents/Diary/`

Accept any absolute path. If the parent directory doesn't exist, tell the user: "I'll create it on first write."

### Step 3: Ask for reminder time

> Would you like a daily reminder to write? What time? (e.g. `21:00` for 9 PM, or "no" to skip)

Accept: `21:00`, `9 PM`, `2100`, `no`, `skip`, `none`.
Convert to cron: `21:00` → `"0 21 * * *"`, `8:30` → `"30 8 * * *"`.

### Step 4: Explain smart reminder

> The reminder is smart — if you've already written a diary entry that day, I won't send a notification. Only empty days get a nudge.

### Step 5: Create the cron job

Use the `cronjob` tool with these exact parameters (substitute `{STORAGE_DIR}` with the user's confirmed path):

```json
{
  "action": "create",
  "name": "diary-reminder",
  "schedule": "0 21 * * *",
  "prompt": "Check if today ({YYYY-MM-DD}) has a diary entry.\n\nSteps:\n1. Determine ISO week: `date +\\%V`\n2. Build path: {STORAGE_DIR}/{YYYY-Www}/{YYYY-MM-DD}.md\n3. If file contains \"# {YYYY-MM-DD}\" as a heading → already written. Say NOTHING.\n4. Otherwise → send: \"📝 该写日记啦\"\n\nDiary path: {STORAGE_DIR}\n\nCritical: If entry exists, SILENT exit — do not send any message.",
  "deliver": "origin",
  "skills": ["daily-diary"]
}
```

⚠️ **Important:** Substitute `{STORAGE_DIR}` with the actual path. Do not send the literal placeholder.

If user said "no" to reminders, skip this step entirely.

### Step 6: Save config to memory

Add these entries to your persistent memory. Use the exact key=value format:

```
diary_storage_dir=/path/to/dir; diary_reminder_time=21:00;
diary_reminder_job_id=abc123; diary_start_command=#diary-start;
diary_end_command=#diary-end; diary_initialized=true;
diary_default_location=Beijing
```

Use `memory(action='add', target='memory', content='...')` with all keys in one string, semicolon-separated.

### Step 7: Confirm

> ✅ Diary is set up! Try `#diary-start` to write your first entry.
