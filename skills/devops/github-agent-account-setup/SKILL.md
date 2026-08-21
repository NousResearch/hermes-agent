---
name: github-agent-account-setup
description: Set up a GitHub account for an autonomous agent (e.g. CodyPlutoRocks), including org membership, PAT creation, and Gmail App Password for IMAP verification. Use when onboarding a new agent identity to GitHub and email.
tags: [github, gmail, agent, pat, imap, verification, openclaw]
---

# GitHub Agent Account Setup

Use when creating a GitHub account for an autonomous agent and wiring up email access so verification codes can be read automatically.

## The Core Problem

GitHub requires device verification via email on every new login from an unknown browser/IP. Google blocks IMAP with plain passwords. This creates a manual bottleneck unless you set up a Gmail App Password BEFORE the first GitHub login attempt.

## Correct Order of Operations

### PHASE 1 — Gmail App Password first (before GitHub login)

Do this BEFORE attempting any GitHub browser login:

1. Have Sander open Gmail for the agent account (e.g. codyplutorocks@gmail.com)
2. Navigate to: myaccount.google.com/apppasswords
3. Create App Password with name: Hermes-IMAP
4. Copy the 16-character app password (shown only once)
5. Store it immediately:

```bash
ssh sander@100.108.223.25 "echo 'CODY_GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx' >> ~/.hermes/.env"
```

6. Test IMAP access with himalaya or python imaplib before proceeding

### PHASE 2 — GitHub account creation

- CAPTCHA on signup requires Sander to do it manually
- Username convention: [AgentName]PlutoRocks (e.g. CodyPlutoRocks)
- Password convention: Pluto0s[Name]!2026
- Email: [agentname]plutorocks@gmail.com

### PHASE 3 — GitHub login + device verification

Now that Gmail App Password is set up, read the verification code automatically:

```python
import imaplib, email, re

def get_github_code(gmail_user, app_password):
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(gmail_user, app_password)
    mail.select('inbox')
    _, data = mail.search(None, 'FROM', 'noreply@github.com', 'UNSEEN')
    for num in data[0].split()[-3:]:  # check last 3 unread
        _, msg_data = mail.fetch(num, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        body = str(msg.get_payload())
        match = re.search(r'Verification code:\s*(\d{6})', body)
        if match:
            return match.group(1)
    return None
```

### PHASE 4 — PAT creation

After login:
1. Navigate to: https://github.com/settings/tokens/new
2. Token name: OpenClaw-[AgentName]
3. Expiration: No expiration
4. Scopes: repo, workflow, write:packages, admin:org, user
5. Generate and store:

```bash
ssh sander@100.108.223.25 "echo 'CODY_GITHUB_PAT=ghp_xxxx' >> ~/.hermes/.env"
ssh sander@100.108.223.25 "echo 'github_pat: ghp_xxxx' >> ~/.openclaw/.cody-credentials"
```

### PHASE 5 — Org membership

From Sander's main GitHub account (AlexanderWatersOxygen):
- Org name: Alexander-waters-oxygen
- Invite the agent account as Owner role
- Agent account: accept the invite (can be done via browser as agent)

Verify with:
```bash
export PATH=$HOME/.local/bin:$PATH && gh api orgs/Alexander-waters-oxygen/members --jq '.[].login'
```

## Pitfalls

- Google blocks browser login to Gmail from automated browsers ("This browser may not be secure") — use IMAP with App Password instead, never browser
- GitHub sends a NEW code each login attempt — old codes expire immediately when a new one is sent
- The App Password must be set up BEFORE the first GitHub login, not after
- Browser sessions time out — type verification codes within 2 minutes of receiving them
- Org name on GitHub may differ from account name: AlexanderWatersOxygen → org is Alexander-waters-oxygen (check with `gh api user/orgs`)
- IMAP plain password login is blocked by Google since 2022 — always use App Password

## Credentials Storage Convention

```
~/.hermes/.env:
  CODY_GITHUB_PAT=ghp_...
  CODY_GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

~/.openclaw/.cody-credentials:
  github_pat: ghp_...
  gmail_app_password: xxxx xxxx xxxx xxxx
```

## Credential Rotation (when password changes)

When Sander drops a new password/token in a file on the SSD (e.g. /mnt/ssd_vm/pluto/Exports/secret.txt),
follow this exact sequence:

### 1 — Read the file over SSH
```bash
ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 \
  'base64 /mnt/ssd_vm/pluto/Exports/secret.txt'
```
Use base64 to avoid shell escaping issues with special characters in tokens.
Decode in Python: `base64.b64decode(output).decode()`

### 2 — Store credentials securely on Pluto
```bash
ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 \
  'mkdir -p ~/.openclaw/credentials && chmod 700 ~/.openclaw/credentials && \
   printf "GITHUB_USER=CodyPlutoRocks\nGITHUB_TOKEN=ghp_xxx\n" \
   > ~/.openclaw/credentials/github_cody.env && \
   chmod 600 ~/.openclaw/credentials/github_cody.env'
```

### 3 — Re-configure gh CLI with the new token
```bash
ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 \
  'export PATH=$HOME/.local/bin:$PATH && \
   echo "ghp_xxx" | gh auth login --hostname github.com --with-token && \
   gh auth status'
```

### 4 — Securely destroy the plaintext file
```bash
ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 \
  'shred -u /mnt/ssd_vm/pluto/Exports/secret.txt && \
   ls /mnt/ssd_vm/pluto/Exports/'
```
Use `shred -u` (not `rm`) — overwrites before unlinking so recovery is impossible.

### 5 — Update agent's MEMORY.md
Note the rotation date but do NOT write the new password in plaintext:
```
## GitHub Account (Cody)
Username: CodyPlutoRocks
Wachtwoord: GEWIJZIGD op DD-MM-YYYY - staat opgeslagen in ~/.openclaw/credentials/github_cody.env
Token: PAT geconfigureerd via gh CLI
```

## Pitfalls (credential rotation)

- Token in a text file may be truncated visually but full in the file — always use base64 to read, don't trust cat output
- `shred` requires the filesystem to support overwrite (ext4/xfs OK; COW filesystems like btrfs/ZFS may not fully overwrite — warn Sander if on such FS)
- Never write the actual password into agent markdown files (MEMORY.md, IDENTITY.md) — only reference the credentials store path
- gh CLI keyring caching: after `gh auth login --with-token`, verify with `gh auth status` to confirm active account switched correctly
