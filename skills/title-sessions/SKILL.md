---
name: title-sessions
category: productivity
description: Generate short conversation titles for CLI sessions stored in ~/.hermes/state.db
---

# Title Sessions

Automatically generates concise, descriptive titles for untitled CLI sessions stored in the Hermes state database.

## How It Works

1. Opens `~/.hermes/state.db` (SQLite)
2. Finds sessions where `title IS NULL` or `title = 'CLI Session'`
3. For each untitled session, reads the first user message from the `messages` table
4. Generates a short title (3-8 words max) that captures the topic
5. Writes the title back: `UPDATE sessions SET title = ?, title_source = 'auto' WHERE id = ?`

## Important Rules

- **Only title once**: Skips sessions that already have a real title (not NULL and not 'CLI Session'). Never overwrites manually-written titles.
- **Title source tracking**: Uses a `title_source` column to distinguish between auto-generated ('auto') and manual ('manual') titles. Only overwrites titles where `title_source = 'auto'` or title is NULL/'CLI Session'.
- **Duplicate handling**: If a generated title already exists, appends a counter (e.g., "Search Web (2)", "Search Web (3)").

## Usage

### Normal Mode (Most Recent Session)
```bash
python ~/.hermes/skills/title_sessions.py
```
Titles only the most recent session that needs a title.

### Backfill Mode (Last 4 Days)
```bash
python ~/.hermes/skills/title_sessions.py --backfill
# or
python ~/.hermes/skills/title_sessions.py backfill
```
Titles all sessions from the last 4 days that don't have titles (or have 'CLI Session').

### Full Mode (All Sessions)
```bash
python ~/.hermes/skills/title_sessions.py --full
# or
python ~/.hermes/skills/title_sessions.py full
```
Titles ALL untitled sessions regardless of age. Use this for initial setup or after database changes.

## Database Schema

The skill requires the `sessions` table to have two columns:
- `title_source` - Tracks whether a title was auto-generated ('auto') or manual ('manual')
- `archived` - Marks sessions as archived (1) or active (0)

If these columns don't exist, the skill will create them automatically:

```sql
ALTER TABLE sessions ADD COLUMN title_source TEXT DEFAULT NULL
ALTER TABLE sessions ADD COLUMN archived INTEGER DEFAULT 0
```

## Title Generation

Titles are generated using a heuristic-based approach that:
- Removes workspace tags like `[Workspace: /path/to/dir]`
- Removes system prefixes like `[SYSTEM: ...]`
- Removes common prefixes like `Note:`, `WARNING:`, `INFO:`
- Filters out stop words (the, and, for, are, but, not, you, all, can, had, etc.)
- Takes the first 6 meaningful words
- Capitalizes the first letter of each word
- Truncates to 50 characters max
- Skips sessions with only system messages (no meaningful content)

## Archiving System-Generated Sessions

The skill can help clean up system-generated sessions (cron jobs, automated tasks) by marking them as archived. These sessions typically have titles like:
- "System User Invoked [Skill] Skill Indicating"
- "System Running Scheduled Cron Job Delivery"
- "System Following Skills Were Listed Job"

To archive these sessions:

```sql
-- Archive all system-generated sessions
UPDATE sessions SET archived = 1
WHERE title LIKE 'System User Invoked%'
   OR title LIKE 'System Running%'
   OR title LIKE 'System Following%';

-- View archived sessions
SELECT id, title, started_at FROM sessions WHERE archived = 1;

-- Restore archived sessions if needed
UPDATE sessions SET archived = 0 WHERE id = 'session_id';
```

The UI should filter out archived sessions by default (WHERE archived = 0).

## Cron Job

A cron job runs this skill every 5 minutes to automatically title new sessions:

```bash
*/5 * * * * python ~/.hermes/skills/title_sessions.py
```

The cron job is lightweight since it skips already-titled sessions.

## Examples

Generated titles from real sessions:
- "What Your Name"
- "Search Web Always Use Searchx Instead"
- "Deep Osint Research More Targets Each"
- "Execute Full Research Expansion Pipeline Peter"
- "Run Pipeline Scout Research Weave Upsert"

## Implementation

The skill is implemented as a Python script at `~/.hermes/skills/title_sessions.py` with the following key functions:

- `get_first_user_message()`: Fetches the first user message from a session
- `generate_title()`: Creates a short title from a message using heuristics
- `get_untitled_sessions()`: Retrieves sessions that need titles
- `update_session_title()`: Updates a session's title with duplicate handling
- `main()`: Entry point that parses arguments and orchestrates the process