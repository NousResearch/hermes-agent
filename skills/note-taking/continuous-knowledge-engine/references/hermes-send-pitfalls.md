# Hermes Send Pitfalls

Guide for robust `hermes send telegram` usage in scripts and cronjobs.

## Required Syntax

`--to` is **mandatory** — omission causes exit code 2:

```
hermes send: --to PLATFORM[:channel[:thread]] is required
```

Correct:
```bash
hermes send --to telegram "Morning report ready"
hermes send --to telegram "Key insight: patterns detected in 5+ videos"
```

Wrong:
```bash
hermes send telegram "Morning report ready"    # Missing --to flag
hermes send "Morning report ready"              # Missing platform entirely
```

## Common Failures

### Chat ID Not Configured
```bash
hermes send --to telegram "Hello"
# Error: no configured target for telegram
```
**Fix:** Verify Telegram config exists:
```bash
grep -A 5 "telegram:" ~/.hermes/config.yaml
```

### Platform Name Mismatch
```bash
hermes send --to Telegram "Hello"    # Wrong case
hermes send --to tg "Hello"           # Wrong shorthand
```
**Fix:** Use exact lowercase platform name: `telegram`, `discord`, etc.

## In Cronjob Prompts

When using `hermes send` inside cronjob prompts, always embed the correct syntax:

```
cronjob action='create'
  name='morning-report'
  schedule='0 6 * * *'
  deliver='local'
  prompt='Run the morning report. When done, use terminal to run:
  hermes send --to telegram "Morning report: {summary}"
  Use the exact --to flag syntax.'
```

## Quick Reference

| What | Syntax |
|------|--------|
| Send to Telegram | `hermes send --to telegram "message"` |
| Send to Discord | `hermes send --to discord "message"` |
| Send to specific thread | `hermes send --to telegram:CHAT_ID:THREAD_ID "message"` |
| Send to all connected | `hermes send --to all "message"` |
