# Fresh Install Checklist — VPS Migration (2026-05-25)

Everything installed/configured after migrating from old machine (7.8GB RAM, 10GB swap) to new VPS (11GB RAM, no swap, no GPU).

## System

| Item | Command | Status |
|------|---------|--------|
| Swap 4GB | `fallocate -l 4G /swapfile && mkswap && swapon && fstab` | ✅ |
| Swap 6GB | `fallocate -l 6G /swapfile2 && mkswap && swapon && fstab` | ✅ |
| Total swap | 10GB (matching old machine) | ✅ |

## Python Packages

| Package | Why | Command |
|---------|-----|---------|
| yt-dlp | Instagram/YouTube downloads | `pip install yt-dlp` |
| playwright | IG carousel extraction | `pip install playwright && python -m playwright install chromium` |
| pyannote.audio >=4.0 | Speaker diarization | `pip install 'pyannote.audio>=4.0'` |
| ebooklib | ePub extraction | `pip install ebooklib` |
| pymupdf | PDF text extraction | `pip install pymupdf` |
| marker-pdf | PDF OCR (scanned) | `pip install marker-pdf` |
| mega.py | Mega.nz downloads | `pip install mega.py` |
| beautifulsoup4 | HTML parsing | `pip install beautifulsoup4` |

### Post-install fixes (marker-pdf cascade)

marker-pdf silently downgrades critical hermes-agent deps. After installing:

```bash
pip install 'openai==2.24.0' 'anthropic==0.87.0' 'tenacity==9.1.4' 'Pillow==12.2.0' 'huggingface-hub==1.16.1' 'tokenizers==0.23.1'
```

Also patch transformers version check (faster-whisper needs tokenizers 0.23.1, transformers wants <=0.23.0):

```bash
sed -i 's/tokenizers>=0.22.0,<=0.23.0/tokenizers>=0.22.0,<=0.24.0/' \
  $(find / -path "*/transformers/dependency_versions_table.py" -not -path "*/__pycache__/*" 2>/dev/null)
```

## Models

| Model | Size | Location | Command |
|-------|------|----------|---------|
| faster-whisper large-v3 | ~3GB | `~/.cache/huggingface/` | `snapshot_download('Systran/faster-whisper-large-v3')` |

## CLI Tools

| Tool | Install | Auth |
|------|---------|------|
| xurl (Twitter) | `npm install -g @xdevplatform/xurl` | OAuth2 via SSH tunnel (`ssh -L 8080:localhost:8080`) |
| GH CLI | System apt | `gh auth status` — token scopes: repo, workflow, gist, read:org |
| ffmpeg | System apt | No auth needed |

## Services

| Service | Command | Port |
|---------|---------|------|
| Firecrawl API | `cd /opt/firecrawl && docker compose up -d api` | 3002 |
| MinIO | Systemd (`minio.service`) | 9000, 9001 |
| Hermes Gateway | Systemd (`hermes-gateway.service`) | — |
| Hermes Dashboard | `hermes dashboard --host 0.0.0.0 --insecure --skip-build` | 9119 |

## Kanban / Cron

| Job | ID | Schedule | What |
|-----|----|----------|------|
| Block Watchdog | `e633161d8f0d` | every 5m | Scans blocked + crash-loop tasks, auto-unblocks |
| CI Watchdog | `10cb5de254d0` | every 2m | Light CI merge polling |
| Pre-Spawn Health | `ceead0ca5089` | every 5m | Scans for dispatch-blocking issues |

## Profile Dotfiles

Copied to `coder`, `reviewer`, `researcher`, `researcher-videos`:
- `~/.gitconfig` (user.email = sevenai@agentmail.to)
- `~/.config/gh/hosts.yml` (GitHub auth token)

xurl config: stored in system credential store (not file-based). Workers needing xurl must auth independently.

## Cookies

User provides from Mac via scp:
- `/root/.hermes/cookies/ig_cookies.txt` — Instagram (must have `sessionid`)
- `/root/.hermes/cookies/yt_cookies.txt` — YouTube

Export: `yt-dlp --cookies-from-browser chrome --cookies /tmp/XX_cookies.txt "URL" -O "done"`

## Repos

| Repo | Path | Remote |
|------|------|--------|
| Hermes Agent | `/usr/local/lib/hermes-agent` | Seven74AI/hermes-agent (fork) |
| Obsidian Vault | `/root/Documents/Obsidian Vault` | Seven74AI/obsidian-vault |
| Hermes Skills | synced via cron `4eee7fb0b484` | Seven74AI/hermes-skills |

## Not Reinstalled (not needed or not critical)

- Camofox browser plugin (dropped)
- Old whisper models other than large-v3
- Python `gh` package (shadowing real GH CLI — removed)
- uv-installed gh (removed, using apt version)

## ⚠️ Pitfalls discovered

1. **pyannote 4.x API change:** `out.speaker_diarization.itertracks()` not `diarization.itertracks()`
2. **marker-pdf cascade:** downgrades openai/anthropic/tenacity — always restore after
3. **Python `gh` package:** uv installs shadow `/usr/bin/gh` — remove it
4. **No GPU:** diarization ~3x realtime, whisper ~2x realtime on CPU
5. **xurl no config file:** uses system credential store — can't copy to profile homes
6. **Firecrawl API:** docker compose, not systemd — manual start after reboot
