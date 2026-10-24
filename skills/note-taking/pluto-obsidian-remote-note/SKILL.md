---
name: pluto-obsidian-remote-note
description: Save a note into Sander's Obsidian vault on Pluto over SSH and verify the write.
---

# Pluto remote Obsidian note

Use this when a result should be persisted in Sander's central Obsidian vault on Pluto instead of only being returned in chat.

## Environment facts
- Pluto SSH endpoint: `sander@100.108.223.25`
- Preferred SSH key: `/home/sanderubuntu/.ssh/hermes_access_ed25519`
- Obsidian vault root: `/mnt/ssd_vm/pluto/vault/`

## Recommended workflow
1. Draft the markdown note locally first.
2. Choose the final vault folder and filename before copying.
3. Create the remote directory with `mkdir -p` over SSH.
4. Copy the file with `scp`.
5. Verify success remotely with both file size and a short preview.
6. In the user reply, report the exact final path.

## Example pattern
```bash
ssh -i /home/sanderubuntu/.ssh/hermes_access_ed25519 sander@100.108.223.25 \
  'mkdir -p "/mnt/ssd_vm/pluto/vault/04_Technologie/OpenClaw/Discord"'

scp -i /home/sanderubuntu/.ssh/hermes_access_ed25519 /tmp/note.md \
  sander@100.108.223.25:'/mnt/ssd_vm/pluto/vault/04_Technologie/OpenClaw/Discord/2026-04-08 Example.md'

ssh -i /home/sanderubuntu/.ssh/hermes_access_ed25519 sander@100.108.223.25 \
  'wc -c "/mnt/ssd_vm/pluto/vault/04_Technologie/OpenClaw/Discord/2026-04-08 Example.md" && printf "\n---\n" && head -n 8 "/mnt/ssd_vm/pluto/vault/04_Technologie/OpenClaw/Discord/2026-04-08 Example.md"'
```

## Verification standard
Treat the note as saved only after:
- `scp` succeeds, and
- remote verification shows the expected file exists and contains the expected opening lines.

## Optional follow-up
If the user expected confirmation in Discord but the current chat is Telegram, schedule or send a short Discord confirmation that includes:
- what was saved,
- the exact Obsidian path,
- the one-line operational summary.

## Pitfalls
- Always quote vault paths; they may contain spaces in other folders.
- Do not claim completion before remote verification.
- Prefer Obsidian for durable plans/runbooks; keep Discord messages brief and non-repetitive.
- If using a prewritten local temp file, make sure the final user message reflects the remote path, not the temp path.
