# Researcher Profile Setup for Knowledge Base

When configuring a kanban worker profile to populate the Obsidian knowledge base,
these steps are required. The profile needs filesystem + git + Instagram access.

## Required skills

The profile's `skills/` directory must include:
- `productivity/knowledge-base` — full workflow, template, fact-checking
- `note-taking/obsidian` — file operations on the vault

```bash
cp -r /root/.hermes/skills/productivity/knowledge-base \
  /root/.hermes/profiles/researcher/skills/productivity/
cp -r /root/.hermes/skills/note-taking/obsidian \
  /root/.hermes/profiles/researcher/skills/note-taking/
```

## Required .env entries

Create a **minimal** `.env` at `/root/.hermes/profiles/<name>/.env` — only the variables the profile actually needs. Do NOT copy the host's full `.env` (428+ lines of irrelevant keys). The worker inherits host env for API keys (DeepSeek, etc.).

Minimal set:
```bash
OBSIDIAN_VAULT_PATH=/root/Documents/Obsidian Vault
FIRECRAWL_API_URL=http://localhost:3002
FIRECRAWL_API_KEY=<key>
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=<key>
MINIO_SECRET_KEY=<key>
MINIO_BUCKET=knowledge-base
HF_TOKEN=hf_...  # HuggingFace token (Read-only) — requis pour télécharger pyannote une fois
```

## Required git config

Copy the host's git config so the profile can push to the vault repo:

```bash
mkdir -p /root/.hermes/profiles/researcher/home
cp /root/.gitconfig /root/.hermes/profiles/researcher/home/.gitconfig
```

## Instagram cookies

Cookies are persisted at `/root/.hermes/cookies/ig_cookies.txt` across sessions. The worker
profile has access to `/tmp/` in its terminal environment. Rate-limiting
is mandatory — the SOUL.md should encode the exact flags.

## Kanban dispatch

The dispatcher runs in the main gateway. The researcher profile does NOT need
its own gateway — workers are spawned on demand when tasks enter `ready`.

```bash
hermes gateway status  # verify gateway is running
```

## Verification checklist

After setup, verify:
1. `grep OBSIDIAN_VAULT_PATH /root/.hermes/profiles/researcher/.env` → set
2. `ls /root/.hermes/profiles/researcher/skills/productivity/knowledge-base/SKILL.md` → exists
3. `ls /root/.hermes/profiles/researcher/home/.gitconfig` → exists
4. `hermes config show` → `kanban.dispatch_in_gateway: true`

## Common pitfalls

- **"Unknown skill" crash** — skill not copied to profile's skills dir. The profile
  only sees what's in its own `skills/` directory (curated per-profile, no blanket sync).
- **"OBSIDIAN_VAULT_PATH not set"** — .env missing the variable. Add it.
- **Git push fails** — .gitconfig not present in profile's home. Copy from host.
- **yt-dlp rate-limit** — cookies missing or expired. Re-export from browser.

## researcher-videos variant (YouTube)

For YouTube video processing, create a separate profile with CPU-isolated settings:

```bash
# Clone from existing researcher
hermes profile create researcher-videos --clone-from researcher

# Copy updated skills (with YouTube pipeline)
cp /root/.hermes/skills/productivity/knowledge-base/references/pipeline-youtube.md \
  /root/.hermes/profiles/researcher-videos/skills/productivity/knowledge-base/references/
cp /root/.hermes/skills/productivity/knowledge-base/references/youtube-note-template.md \
  /root/.hermes/profiles/researcher-videos/skills/productivity/knowledge-base/references/
cp /root/.hermes/skills/productivity/knowledge-base/SKILL.md \
  /root/.hermes/profiles/researcher-videos/skills/productivity/knowledge-base/SKILL.md

# Adapt config — max_spawn=1 (whisper is CPU-heavy), high turn budget for long videos
# Edit ~/.hermes/profiles/researcher-videos/config.yaml:
#   agent.max_turns: 240
#   agent.max_iterations: 240
#   kanban.max_spawn: 1

# Copy gitconfig
mkdir -p /root/.hermes/profiles/researcher-videos/home
cp /root/.gitconfig /root/.hermes/profiles/researcher-videos/home/.gitconfig

# YouTube cookies (user exports from desktop browser, then scp)
# On user's Mac:
#   yt-dlp --cookies-from-browser chrome --cookies /root/.hermes/cookies/yt_cookies.txt "https://www.youtube.com/" -O "done"
#   scp /root/.hermes/cookies/yt_cookies.txt root@<server>:/root/.hermes/cookies/yt_cookies.txt
```

### Key differences from researcher (Reels)

| Setting | researcher | researcher-videos |
|---------|-----------|-------------------|
| `max_spawn` | 3 | 1 (whisper is CPU-heavy) |
| `max_turns` | 90 | 240 (long videos) |
| `max_iterations` | 120 | 240 |
| Cookies | `/root/.hermes/cookies/ig_cookies.txt` | `/root/.hermes/cookies/yt_cookies.txt` |
| Rate-limit | 2 MB/s, sleep 3-15s | 4 MB/s, sleep 1-10s |
| Max per session | 2-3 Reels | 2 videos |
| MinIO folder | — | `videos/` |
