---
name: time-awareness
description: Enables agents to accurately perceive the current date, time, timezone, and elapsed session duration. Agents learn that their system prompt timestamp is frozen at session start and must actively check the real clock for time-sensitive decisions. Provides timezone-aware datetime commands, duration calculation, and scheduling awareness.
version: 1.0.0
author: Vivere Vitalis
license: MIT
metadata:
  hermes:
    tags: [time, date, timezone, scheduling, awareness, cron, clock]
    related_skills: []
    requires_toolsets: [terminal]
---

# Time Awareness — Know What Time It Actually Is

Agents receive a `Conversation started:` timestamp in their system prompt, but that timestamp is **frozen at session creation**. During long sessions, cron-triggered tasks, or async dispatches, the frozen timestamp becomes increasingly stale. This skill teaches agents when and how to check the real clock.

## When to Use

Load this skill when:
- The user asks "what time is it?" or "what day is it?"
- A task involves scheduling, deadlines, due dates, or time-sensitive decisions
- A task involves cron jobs, reminders, or "do this later" instructions
- You need to calculate how long something has been running (elapsed time)
- You are dispatched asynchronously and the session timestamp may be hours old
- You need to compare dates (e.g., "was this file modified today?")
- You need to produce timestamps in logs, filenames, or reports
- The user mentions a specific time ("call me at 3pm", "remind me tomorrow morning")
- You are reasoning about whether something is "recent", "stale", "overdue", or "upcoming"

## The Problem: Frozen Timestamps

Your system prompt includes a line like:

```
Conversation started: Friday, April 17, 2026 09:12 AM
```

This timestamp is captured **once** when the session begins and **never updated**. This means:

1. **Long sessions drift.** If a session runs for 2 hours, the frozen timestamp becomes increasingly wrong.
2. **Cron/dispatch tasks are stale.** A cron job or async dispatch may start minutes or hours after the timestamp was set.
3. **No timezone guarantee.** If `timezone` is not configured in `config.yaml`, the timestamp uses server local time, which may not match the user's timezone.
4. **No elapsed time awareness.** The system prompt tells you when the session started, but not how long it has been running.

**Rule: Never trust the system prompt timestamp for time-sensitive decisions. Always check the real clock.**

## Quick Reference

### Check Current Time (Always Use This Instead of System Prompt)

```bash
# Full human-readable datetime with timezone
date +"%A, %B %d, %Y at %I:%M %p %Z"

# ISO 8601 (for logs, filenames, API calls)
date -u +"%Y-%m-%dT%H:%M:%SZ"

# Compact date (for filenames, IDs)
date +"%Y-%m-%d"

# Unix timestamp (for comparisons)
date +%s

# Time only
date +"%H:%M %Z"
```

### Check User's Configured Timezone

```bash
# What timezone is Hermes configured for?
grep "^timezone:" ~/.hermes/config.yaml 2>/dev/null || echo "(not configured — using server local time)"

# Or check the environment variable
echo "${HERMES_TIMEZONE:-not set}"

# System timezone
readlink /etc/localtime 2>/dev/null || timedatectl show -p Timezone 2>/dev/null || echo "unknown"
```

### Calculate Elapsed Time

```bash
# How long has this session been running?
# Use the session ID timestamp (format: YYYYMMDD_HHMMSS)
session_id="20260417_091200_abc123"
session_start=$(date -j -f "%Y%m%d_%H%M%S" "$(echo $session_id | cut -c1-15)" +%s 2>/dev/null || echo "0")
now=$(date +%s)
elapsed=$(( now - session_start ))
echo "Session has been running for $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m"
```

### Time-Aware Decision Making

| Task Type | What to Check | Command |
|-----------|--------------|---------|
| Is something "recent"? | Current time vs. event time | `date +%s` then compare |
| Is it "today"? | Current date | `date +"%Y-%m-%d"` |
| Is a deadline overdue? | Current time vs. deadline | `date -u +"%Y-%m-%dT%H:%M:%SZ"` |
| Schedule something "later" | Current time + offset | `date -v+2H +"%Y-%m-%dT%H:%M:%S"` |
| Time-sensitive logging | Precise timestamp | `date -u +"%Y-%m-%dT%H:%M:%SZ"` |
| User in different timezone | Convert from their TZ | `TZ=America/New_York date` |
| Is it a weekday? | Day of week | `date +"%A"` |
| Is it business hours? | Current hour in user TZ | `date +"%H"` |

### Convert Timezones

```bash
# Current time in a specific timezone (no installation needed)
TZ="America/New_York" date +"%I:%M %p %Z"
TZ="Europe/London" date +"%I:%M %p %Z"
TZ="Asia/Tokyo" date +"%I:%M %p %Z"
TZ="UTC" date +"%I:%M %p %Z"
```

### Relative Time Calculations (macOS/BSD date)

```bash
# 2 hours from now
date -v+2H +"%Y-%m-%d %H:%M"

# Tomorrow at midnight
date -v+1d +"%Y-%m-%d 00:00"

# 30 minutes ago
date -v-30M +"%Y-%m-%d %H:%M"

# Next Monday
date -v+1w +"%Y-%m-%d"
```

### Relative Time Calculations (GNU date / Linux)

```bash
# 2 hours from now
date -d "+2 hours" +"%Y-%m-%d %H:%M"

# Tomorrow at midnight
date -d "tomorrow" +"%Y-%m-%d 00:00"

# 30 minutes ago
date -d "30 minutes ago" +"%Y-%m-%d %H:%M"

# Next Monday
date -d "next monday" +"%Y-%m-%d"
```

## Procedure

### Rule 1: Always Check the Clock for Time-Sensitive Decisions

Before making any decision that depends on "now" — scheduling, deadlines, "is this recent?", "should I do this today?" — run `date` first. Do not trust the `Conversation started:` timestamp.

```
# WRONG — trusting frozen system prompt timestamp
"The conversation started at 9 AM, so it's probably still morning."

# RIGHT — checking the real clock
$ date +"%H:%M"
"22:47" — it's actually almost 11 PM. The session started 14 hours ago.
```

### Rule 2: Use the User's Timezone for User-Facing Output

When presenting times to the user, use their configured timezone. Check `timezone` in `config.yaml` or `HERMES_TIMEZONE` env var. If neither is set, use the server's local time (which may differ from the user's timezone).

```bash
# Check configured timezone
tz=$(grep "^timezone:" ~/.hermes/config.yaml 2>/dev/null | sed 's/timezone: *//' | tr -d "'\"")
if [ -n "$tz" ]; then
    TZ="$tz" date +"%A, %B %d, %Y at %I:%M %p %Z"
else
    date +"%A, %B %d, %Y at %I:%M %p %Z"
fi
```

### Rule 3: Calculate Elapsed Time for Long-Running Tasks

When a session has been running for more than a few minutes, calculate elapsed time from the session ID (embedded in the `Conversation started:` line format `YYYYMMDD_HHMMSS`).

```bash
# Extract session start from session ID
session_start_ts=$(echo "20260417_091200" | sed 's/_/ /' | xargs date -j -f "%Y%m%d %H%M%S" +%s 2>/dev/null)
now_ts=$(date +%s)
elapsed=$(( now_ts - session_start_ts ))
echo "Session running for $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m"
```

### Rule 4: Timestamp Filenames and Logs in ISO 8601

When creating timestamps for filenames, logs, or API calls, always use ISO 8601 format:

```bash
# For filenames (sortable, no spaces)
date -u +"%Y-%m-%dT%H-%M-%SZ"

# For logs (standard format)
date -u +"%Y-%m-%dT%H:%M:%SZ"

# For user-facing display (human-readable with timezone)
date +"%A, %B %d, %Y at %I:%M %p %Z"
```

### Rule 5: Be Explicit About Time Zones in Communication

When telling a user when something will happen, always include the timezone:

```
# BAD: "I'll check at 3pm"
# GOOD: "I'll check at 3:00 PM PDT"
# GOOD: "The cron job runs at 09:00 America/Los_Angeles daily"
```

### Rule 6: Respect Business Hours and Weekends

When scheduling tasks or deciding when to notify users:

```bash
hour=$(date +"%H")
day=$(date +"%u")  # 1=Monday, 7=Sunday

if [ "$day" -le 5 ] && [ "$hour" -ge 9 ] && [ "$hour" -lt 17 ]; then
    echo "Business hours — OK to notify"
else
    echo "Outside business hours — queue for later"
fi
```

## Pitfalls

1. **The system prompt timestamp is FROZEN.** It never updates during a session. For any time-sensitive decision, run `date` to get the real current time.

2. **Server time may not be user time.** If `timezone` is not configured in `config.yaml`, the system uses the server's local time. If the server is in UTC and the user is in `America/Los_Angeles`, all timestamps will be 7-8 hours off. Always check: `grep "^timezone:" ~/.hermes/config.yaml`.

3. **macOS `date` vs. GNU `date`.** The `-v` flag (relative time) is macOS/BSD only. On Linux, use `-d "2 hours ago"` instead. When writing cross-platform scripts, detect the platform first:
   ```bash
   if date -v+1H +%s 2>/dev/null; then
       # macOS/BSD — use -v flag
       date -v+2H +"%Y-%m-%d %H:%M"
   else
       # Linux/GNU — use -d flag
       date -d "+2 hours" +"%Y-%m-%d %H:%M"
   fi
   ```

4. **Session IDs encode the start time.** The format is `YYYYMMDD_HHMMSS_<hex>`. You can parse this to calculate elapsed time, but remember it's when the session was created, not when the current task started.

5. **Daylight Saving Time transitions.** Between DST transitions, `date` may produce unexpected results (1-hour jumps, ambiguous times). For scheduling across DST boundaries, use UTC and convert at display time.

6. **Cron vs. real time.** Cron expressions like `0 9 * * *` run at 9 AM in the system timezone, not the user's timezone. If scheduling via `hermes cron create`, confirm which timezone applies.

7. **`date +%s` is always UTC.** Unix timestamps are timezone-independent. Use `date +"%Y-%m-%dT%H:%M:%S %Z"` when you need human-readable local time.

## Verification

After loading this skill, verify it works:

```bash
# 1. Confirm current time is available
date +"%A, %B %d, %Y at %I:%M %p %Z"

# 2. Confirm timezone configuration
grep "^timezone:" ~/.hermes/config.yaml 2>/dev/null || echo "(not configured)"

# 3. Confirm you can calculate elapsed time
echo "Session started at: $(date -j -f '%Y%m%d_%H%M%S' "$(date +%Y%m%d_%H%M%S)" +'%A, %B %d, %Y at %I:%M %p' 2>/dev/null || date +'%A, %B %d, %Y at %I:%M %p')"
echo "Current time: $(date +'%A, %B %d, %Y at %I:%M %p %Z')"

# 4. Confirm timezone conversion works
TZ="UTC" date +"%I:%M %p %Z"
TZ="America/New_York" date +"%I:%M %p %Z"
TZ="America/Los_Angeles" date +"%I:%M %p %Z"
```

If all four checks produce output, the skill is working.

## Configuration

To set the timezone for all Hermes profiles, add to `~/.hermes/config.yaml`:

```yaml
timezone: 'America/Los_Angeles'
```

Or set the environment variable:

```bash
export HERMES_TIMEZONE='America/Los_Angeles'
```

Both use IANA timezone identifiers (e.g., `America/New_York`, `Europe/London`, `Asia/Tokyo`). The environment variable takes priority over config.yaml.