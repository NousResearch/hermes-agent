---
name: bark
description: Send free push notifications to an iPhone.
version: 1.0.0
author: Randool, iam1maker
license: MIT
platforms: [macos, linux, windows]
required_environment_variables:
  - name: BARK_TOKEN
    prompt: Bark device key
    help: Open Bark on the iPhone and copy the key from the test URL on its home page (https://api.day.app/<key>).
    required_for: sending pushes
metadata:
  hermes:
    tags: [notifications, ios, push, apple, alerts]
    category: productivity
    config:
      - key: bark.server
        description: Bark server base URL; keep the default for the public api.day.app.
        default: "https://api.day.app"
        prompt: Bark server URL (https://api.day.app for the public server)
---

# Bark Skill

Send push notifications to an iPhone through a Bark server, either the public
`https://api.day.app` or a self-hosted instance. The device key from the Bark
app identifies the phone, so no account or bot token is involved.

It does not target Android or desktop clients, and it does not implement
Bark's encrypted-push payloads: those need a key configured in the app plus
client-side AES encryption, which this skill leaves to the official Bark
documentation.

## When to Use

- A long task finished (build, training run, report generation) and the user is away from the machine
- Delivering cron job results or monitoring alerts to a phone
- Any "tell me when it is done" request where the user prefers a phone notification over chat

## Prerequisites

1. Install **Bark** from the App Store (`id1403753865`) and allow notifications for it.
2. Open the app and copy the key from the test URL on its home page (`https://api.day.app/<key>`).
3. Store the key as a secret in `~/.hermes/.env`:

   ```
   BARK_TOKEN=<key>
   ```

   Hermes loads that file at startup; the skill prompts for the key on first load when it is missing.
   Exporting `BARK_TOKEN` in the shell works too, but does not survive a restart.
4. Optional, for a self-hosted server: set the skill config and point the app at the same address.

   ```bash
   hermes config set skills.config.bark.server https://bark.example.com
   ```

## How to Run

Invoke the helper through the `terminal` tool from this skill's directory. It handles UTF-8
bodies, JSON escaping and response codes, which URL-path pushes get wrong as soon as the
message contains non-ASCII text, a slash or an ampersand.

```bash
python3 scripts/bark_push.py --title "Hermes" --body "Build finished: 3 tests failed"
```

The helper prints the server response and exits non-zero when the push was rejected.
When the skill config sets a non-default server, pass it through: `--server https://bark.example.com`.

## Quick Reference

| Goal | Command |
|------|---------|
| Title, subtitle, body | `python3 scripts/bark_push.py --title T --subtitle S --body B` |
| Time-sensitive push | add `--level timeSensitive` |
| Critical alert | add `--level critical` (needs the app permission) |
| Markdown body | add `--markdown "**done** in 42s"` instead of `--body` |
| Repeat ringtone for 30s | add `--call` |
| Group notifications | add `--group hermes` |
| Open a URL on tap | add `--url https://example.com` |
| Badge number | add `--badge 1` |
| Withdraw a push | `--id <id> --delete` (same id as the original push) |
| Self-hosted server | add `--server https://bark.example.com` |
| Preview only | add `--dry-run` |

## Procedure

1. Confirm `BARK_TOKEN` is set; the helper exits with the exact fix when it is missing.
2. Build the push with `scripts/bark_push.py`: `--title` for the first line, `--body` for the
   message, `--level` for interruption behaviour.
3. Run it through the `terminal` tool and read the printed JSON.
4. Use `--id` with a stable value for progress notifications: Bark replaces the previous push
   with the same id instead of stacking a new one.
5. Send `--id <id> --delete` to withdraw a notification once the task no longer needs attention.

## Pitfalls

- **The URL-path form cannot carry arbitrary text.** Spaces, Chinese characters, `/` and `&`
  break the path or arrive as mojibake; the helper posts JSON instead.
- **`--level critical` is silent without permission.** iOS downgrades the push to a normal
  notification unless Bark was granted Critical Alerts in iOS Settings; the request still returns
  `code: 200`.
- **Focus mode swallows `active` pushes.** Use `timeSensitive` or `critical` for alerts that must
  interrupt.
- **Withdraw needs Background App Refresh.** `--delete` is delivered as a silent push, so it does
  nothing when Background App Refresh is off for Bark.
- **Batch `device_keys` is inconsistent across Bark's docs.** The parameters page documents up to
  10 keys on the public server while the batch page says the public server does not support it at
  all (it needs bark-server 2.1.9+). Test before relying on it; self-host for volume.
- **The key sits in the URL path.** It ends up in proxy and server logs; treat it as a secret and
  never paste a real key into a shared transcript.
- **Use HTTPS for a self-hosted server.** Bark's privacy page recommends HTTPS, and iOS App
  Transport Security may reject a plain-HTTP address; the server address configured in the app
  must match `skills.config.bark.server`.

## Verification

Send a test push and check both the response and the phone:

```bash
python3 scripts/bark_push.py --title "Hermes" --body "Bark is configured"
```

A successful push returns `{"code":200,"message":"success"}` and the notification appears on the
device. A non-200 `code` means the push was rejected; check the key and the server address.
