---
name: mail-auto-draft
description: Build and deploy a Himalaya-based inbound email auto-reply workflow with self-reply protection, Reply-To aware recipient selection, and systemd user timers.
version: 1.0.0
author: jozrftamson
license: MIT
metadata:
  hermes:
    tags: [email, himalaya, imap, smtp, automation, systemd, gmail]
    category: email
prerequisites:
  commands: [himalaya, python3, systemctl]
---

# Mail Auto-Draft with Himalaya

Build a practical local inbound email automation workflow that:
- reads new mail from INBOX via Himalaya
- classifies simple inbound requests
- automatically replies only to safe standard cases
- avoids self-reply loops
- runs continuously via systemd user timers

This skill is intentionally optional. It is highly useful for users who want mailbox automation, but it depends on local mail account setup, machine-specific deployment choices, and provider-specific operational details.

## When to Use

Use this skill when the user wants to automate replies to inbound email on their own mailbox, especially on Ubuntu/Linux with Himalaya.

Typical requests:
- "Automatically answer simple incoming emails"
- "Set up a local email auto-responder with Gmail and Himalaya"
- "Keep it running in the background every minute"
- "Make it safe enough for production"

This skill is for user-owned mailboxes, not agent-owned inboxes like AgentMail.

## Quick Reference

Typical flow:
1. configure Himalaya for the mailbox
2. create or adapt `process_inbox.py` and `config.yaml`
3. test in draft mode
4. enable auto-send only for safe whitelisted categories
5. run continuously with a `systemd --user` timer

## Requirements

1. Himalaya CLI installed
2. Python 3 with `pyyaml` and `requests`
3. A working IMAP/SMTP account in `~/.config/himalaya/config.toml`
4. A local project directory containing:
   - `process_inbox.py`
   - `config.yaml`
   - `prompts/system_prompt.txt`
   - `prompts/user_prompt.txt`

## Recommended Project Files

Minimum structure:
- `process_inbox.py`
- `config.yaml`
- `config.example.yaml`
- `README.md`
- `SCHNELLSTART_AND_INSTALLATION.md`
- `deploy/systemd/mail-auto-draft.service`
- `deploy/systemd/mail-auto-draft.timer`
- `prompts/`

Local runtime dirs:
- `drafts/`
- `logs/`
- `data/`
- `runtime/`

## Safe Defaults

For production, prefer:
- `require_unseen: true`
- `require_new_in_inbox: true`
- `require_whitelist: true`
- `require_high_confidence: true`
- `confidence_threshold` around `70` to `85`
- `forbid_sensitive_categories` includes:
  - `sensibel`
  - `individuell`
  - `unklar`
  - `ignorieren`

## Critical Safety Rules

### 1. Always define own addresses

In `config.yaml`:

```yaml
own_addresses:
  - your-address@example.com
```

This prevents self-reply loops when your own replies appear in INBOX.

### 2. Prefer Reply-To over From

When selecting the reply recipient:
1. use `Reply-To` if present
2. otherwise use `From`
3. if the generated reply template has empty `To:`, backfill it before sending

### 3. Ignore own-sender mail

Any inbound mail whose sender matches `own_addresses` should be ignored with a reason like `self_sender`.

### 4. Do not auto-send uncertain categories

Keep these as draft-only or blocked:
- `sensibel`
- `individuell`
- `unklar`
- `ignorieren`

### 5. Treat Gmail Sent append failures carefully

On Gmail, SMTP may succeed while Himalaya still exits non-zero because append to Sent fails.
If stderr includes both:
- `cannot add IMAP message`
- `Folder doesn't exist`

then treat it as likely sent and avoid duplicate resend.

## Himalaya Config Notes

For Gmail, use an app password, not the normal password.

Recommended pattern:
- keep the app password out of the project repo
- store it in a local secret file such as:
  `~/.config/mail-auto-draft/secrets.env`

Example local secret file:

```bash
GMAIL_APP_PASSWORD='YOUR_APP_PASSWORD'
```

Example auth command in `~/.config/himalaya/config.toml`:

```toml
backend.auth.cmd = "sh -lc '. /home/USER/.config/mail-auto-draft/secrets.env && printf %s \"$GMAIL_APP_PASSWORD\"'"
message.send.backend.auth.cmd = "sh -lc '. /home/USER/.config/mail-auto-draft/secrets.env && printf %s \"$GMAIL_APP_PASSWORD\"'"
```

## Continuous Background Processing

Prefer a `systemd --user` oneshot service plus timer over an infinite Python loop.

Example service template:

```ini
[Unit]
Description=Process inbox and auto-reply via Himalaya
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=__PROJECT_DIR__
ExecStart=/usr/bin/flock -n __PROJECT_DIR__/runtime/process_inbox.lock /usr/bin/python3 __PROJECT_DIR__/process_inbox.py --limit 5
```

Example timer template:

```ini
[Unit]
Description=Run mail auto-draft every minute

[Timer]
OnBootSec=2min
OnUnitActiveSec=60s
Persistent=true
Unit=mail-auto-draft.service

[Install]
WantedBy=timers.target
```

## Procedure

### 1. Verify dependencies

```bash
himalaya --version
python3 -c "import yaml, requests"
python3 -m py_compile process_inbox.py
```

### 2. Test in draft mode first

```bash
python3 process_inbox.py --mode draft --limit 5
```

Review:
- generated drafts
- `logs/mail_actions.jsonl`

### 3. Switch to auto mode only after review

```bash
python3 process_inbox.py --mode auto --limit 1
```

### 4. Enable timer-based background processing

```bash
systemctl --user daemon-reload
systemctl --user enable --now mail-auto-draft.timer
systemctl --user start mail-auto-draft.service
```

## Useful Runtime Commands

Status:

```bash
systemctl --user status mail-auto-draft.timer --no-pager
systemctl --user status mail-auto-draft.service --no-pager
```

Logs:

```bash
journalctl --user -u mail-auto-draft.service -n 50 --no-pager
```

Manual run:

```bash
python3 process_inbox.py --limit 1
```

## Example Workflow

Example use case:
- a user wants automatic replies for simple inbound messages such as information requests, lightweight appointment requests, or simple acknowledgements
- Hermes helps configure Himalaya and the local processing script
- the workflow only auto-sends safe standard categories
- anything uncertain stays out of the auto-send path

## Troubleshooting

### It replies to itself repeatedly
Check:
- `own_addresses` is configured
- self-sender mail is ignored
- only new/unseen mail is processed

### Replies go to the wrong recipient
Check:
- recipient selection prefers `Reply-To`
- fallback uses `From`
- empty `To:` in reply templates is repaired before send

### It sends to newsletters or bulk mail
Tighten:
- subject keyword filters
- header ignore rules
- whitelist-only auto-send
- high confidence requirement

### Himalaya says send failed but recipient got the email
This is often the Gmail IMAP append issue after SMTP success.
Verify recipient delivery before resending.

## Verification

A successful setup should show:
- inbound external mail gets processed
- `chosen_reply_recipient` is logged correctly
- own mail is ignored with `self_sender`
- background timer runs every minute
- no repeated self-reply loops

## References

- `references/production-checklist.md`
- `references/setup-commands.md`
