# Dependencies Checklist

All packages required for the full knowledge-base pipeline (YouTube, Instagram Reels, image carousels, books, Mega.nz). Installed in `/usr/local/lib/hermes-agent/venv/`.

## Python packages

| Package | Version | Pipeline | Notes |
|---------|---------|----------|-------|
| `yt-dlp` | latest | YouTube + IG Reels | `--js-runtimes node` mandatory on datacenter IPs |
| `faster-whisper` | latest | YouTube + IG Reels | `large-v3` model, `compute_type='int8'` on CPU |
| `pyannote.audio` | **>=4.0** | YouTube + IG Reels (multi-speaker) | Pin: `pip install 'pyannote.audio>=4.0'`. 3.x crashes on torch >=2.5 |
| `playwright` | latest | IG carousels | `python -m playwright install chromium` after pip |
| `ebooklib` | latest | Books (ePub) | |
| `pymupdf` | latest | Books (PDF text) | |
| `marker-pdf` | latest | Books (PDF OCR) | **⚠️ downgrades hermes-agent deps — restore after install** |
| `mega.py` | latest | Mega.nz downloads | |
| `beautifulsoup4` | latest | HTML parsing (curl fallback) | |
| `ffmpeg` | system | All video/audio | `apt-get install ffmpeg` |
| `node` | ≥v20 | yt-dlp n-sig solver | System package |

## Whisper model

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Systran/faster-whisper-large-v3', cache_dir='/root/.cache/huggingface')
"
```

~3 GB download. Only needed once. Cached to `/root/.cache/huggingface/`.

## Post-install: restore hermes-agent dependencies

`marker-pdf` silently downgrades critical packages. After installing ALL packages:

```bash
pip install 'openai==2.24.0' 'anthropic==0.87.0' 'tenacity==9.1.4' \
  'Pillow==12.2.0' 'huggingface-hub==1.16.1' 'tokenizers==0.23.1'
```

Then verify: `pip check`. Expected only one warning: `mega-py wants tenacity<6` — cosmetic, mega.py works fine with tenacity 9.x.

## Post-install: fix GH CLI shadowing

The Python `gh` package (installed by uv) shadows `/usr/bin/gh` (GitHub CLI). Remove it:

```bash
rm /root/.local/bin/gh
hash -r
which gh  # should show /usr/bin/gh
```

## Post-install: Firecrawl API

If Docker containers exist but API isn't running on port 3002:

```bash
cd /opt/firecrawl && docker compose up -d api
```

Verify: `curl -s -X POST http://localhost:3002/v2/scrape -H "Content-Type: application/json" -d '{"url":"https://example.com","formats":["markdown"]}'`

## xurl (Twitter/X)

```bash
npm install -g @xdevplatform/xurl
```

## Environment variables

| Variable | Location | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | `researcher-videos/.env` | pyannote model download |
| `OBSIDIAN_VAULT_PATH` | `~/.hermes/.env` | Vault path |
| `GITHUB_TOKEN` | gateway env.conf | Git push |
| `ANTHROPIC_API_KEY` | `~/.hermes/.env` | Vision tool (Claude Haiku) |
| `DEEPSEEK_API_KEY` | `~/.hermes/.env` | Main model |

## Performance (CPU-only, 6 vCPU, 8 GB RAM, no GPU)

| Task | Duration | CPU |
|------|----------|-----|
| pyannote 4.x diarization, 13 min audio | ~30-40 min | ~350% |
| faster-whisper large-v3 int8, 13 min | ~25-30 min | ~250% |
| Single-speaker: whisper-only | ~25-30 min | ~250% |

On this machine, pyannote + whisper concurrent workers push total load >6. Kill kanban workers during heavy transcription to avoid thrashing.
