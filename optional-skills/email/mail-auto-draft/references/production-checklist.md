# Production Checklist

Use this checklist before enabling automatic replies in production.

## Mail Safety

- [ ] `own_addresses` contains the mailbox's real sender address
- [ ] self-sender mail is ignored
- [ ] `Reply-To` is preferred over `From`
- [ ] empty `To:` gets backfilled before sending
- [ ] only new/unseen inbound mail is processed

## Classification Safety

- [ ] whitelist-only auto-send is enabled
- [ ] confidence threshold is not set too low
- [ ] `sensibel`, `individuell`, `unklar`, `ignorieren` are blocked from auto-send
- [ ] newsletter/system-mail filters are active

## Secrets

- [ ] no app password is committed to the repository
- [ ] local secret file permissions are restricted (for example `chmod 600`)
- [ ] Himalaya uses a local secret command instead of hardcoded plaintext in the repo

## Runtime

- [ ] `python3 -m py_compile process_inbox.py` passes
- [ ] draft mode test passed
- [ ] auto mode test passed with an external sender
- [ ] systemd timer is active
- [ ] JSONL log shows `chosen_reply_recipient`

## Gmail-specific

- [ ] Gmail app password is used
- [ ] team understands that SMTP may succeed even if IMAP append to Sent fails
- [ ] no duplicate resend logic is triggered on Gmail append failure
