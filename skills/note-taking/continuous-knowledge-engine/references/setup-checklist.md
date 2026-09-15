# Setup Checklist

User-facing checklist for setting up the Continuous Knowledge Engine.

## Prerequisites

- [ ] Hermes Agent running with Telegram or Discord configured
- [ ] Git installed (`git --version`)
- [ ] Python 3.10+ installed (`python3 --version`)

## Step 1: Create Knowledge Base

```bash
KNOWLEDGE_BASE="${HOME}/knowledge-base"
mkdir -p "${KNOWLEDGE_BASE}"/{discord/threads,youtube/transcripts,faculdade/materias,obsidian/{00-Inbox,01-YouTube,02-Discord,03-Faculdade,04-Patterns},scripts,logs,config}
cd "${KNOWLEDGE_BASE}"
git init
```

## Step 2: Copy Scripts

Copy the gatherer and reporter scripts into `${KNOWLEDGE_BASE}/scripts/`:
- `knowledge-gatherer.py` (see `references/knowledge-gatherer-template.md`)
- `progress-reporter.py` (see `references/progress-reporter-template.md`)

## Step 3: Configure Data Sources

```bash
# YouTube channels
echo "CHANNEL_ID_1" > "${KNOWLEDGE_BASE}/config/youtube-channels.txt"
echo "CHANNEL_ID_2" >> "${KNOWLEDGE_BASE}/config/youtube-channels.txt"

# Academic materials
echo "MateriaName:/path/to/materials" > "${KNOWLEDGE_BASE}/config/faculdade-materias.txt"
```

## Step 4: Test Scripts

```bash
python3 "${KNOWLEDGE_BASE}/scripts/knowledge-gatherer.py"
python3 "${KNOWLEDGE_BASE}/scripts/progress-reporter.py" daily
```

## Step 5: Schedule Cronjobs via Hermes

In a Hermes session, use the `cronjob` tool to create scheduled jobs.
See `references/hermes-cronjob-usage.md` for syntax.

## Step 6: Verify

- [ ] Obsidian notes appear in `${KNOWLEDGE_BASE}/obsidian/`
- [ ] Git commits show in `git log --oneline`
- [ ] Morning/night reports delivered to Telegram

## Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `KNOWLEDGE_BASE` | `${HOME}/knowledge-base` | Root directory for all data |
| `VAULT_PATH` | `${HOME}/dev/workspace/Personal-Vault-backup` | Obsidian vault path |
