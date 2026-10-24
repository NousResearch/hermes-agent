---
name: bark
description: Push notifications to iOS devices via Bark.
version: 1.0.0
author: Randool
license: MIT
platforms: [macos, linux, windows]
required_environment_variables:
  - name: BARK_TOKEN
    prompt: Bark device token
    help: Open Bark app on iPhone; the token is the last segment of the URL shown on the home page (e.g. https://api.day.app/XXXX → XXXX)
    required_for: all functionality
metadata:
  hermes:
    tags: [notification, ios, apple, push]
---

# Bark Skill

Send free push notifications to iOS devices via [Bark](https://github.com/finb/bark). Supports title/subtitle/body, custom sounds, notification levels (including critical alerts), and end-to-end encryption.

Does not support Android — use ntfy, Gotify, or Telegram Bot for cross-platform needs.

## When to Use

- Notify the user after long-running tasks (builds, training, report generation)
- Push cron job execution results
- Monitoring alerts (threshold triggers, anomaly detection)
- Any "let me know when it's done" scenario

## Prerequisites

1. Install **Bark** from the iOS App Store (by Finb) and allow push notifications
2. Copy the device token from the Bark app home page (the last segment of the URL, e.g. `https://api.day.app/XXXX`)
3. Set the `BARK_TOKEN` environment variable — the skill will prompt for it on first load if not configured

## How to Run

All pushes are executed through the `terminal` tool using `curl`.

**Simple push (GET):**
```bash
curl "https://api.day.app/$BARK_TOKEN/Hello%20from%20Hermes"
```

**Title + body (GET):**
```bash
curl "https://api.day.app/$BARK_TOKEN/Task%20Complete/Report%20generated"
```

**Full-featured push (POST JSON):**
```bash
curl -X POST "https://api.day.app/$BARK_TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"title":"Task Complete","body":"Report generated","sound":"minuet"}'
```

## Quick Reference

| Goal | Method |
|------|--------|
| Simple push | `GET /$TOKEN/body` |
| Title + body | `GET /$TOKEN/title/body` |
| Full features | `POST /$TOKEN` with JSON body |
| Batch push | `POST /push` with `device_keys` array |
| Critical alert | `"level":"critical"` in POST body |
| Withdraw notification | `POST` with `{"id":"...","delete":"1"}` |

## Procedure

1. Ensure `BARK_TOKEN` is set (the skill prompts on first use if missing)
2. Choose GET (simple) or POST (full-featured) based on your needs
3. For GET: URL-encode the message and call the endpoint via `terminal`
4. For POST: build a JSON body with desired parameters and call via `terminal`
5. Check the response — `{"code":200,"message":"success"}` means delivered

## Pitfalls

- **GET URL length limits**: Use POST JSON for content longer than ~200 characters
- **URL encoding**: GET requests require URL-encoding of spaces, non-ASCII, and special characters (`&`, `?`, etc.)
- **Delivery not guaranteed**: iOS may throttle or drop notifications under heavy load
- **Critical alerts**: Require iOS to grant critical alert permission to Bark in Settings
- **Self-hosted Bark**: Replace `api.day.app` with your server address in all commands

## Verification

Send a test notification:

```bash
curl "https://api.day.app/$BARK_TOKEN/Bark%20configured%20successfully"
```

Expected response: `{"code":200,"message":"success"}`. Confirm the notification appears on the iOS device.
