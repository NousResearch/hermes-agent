# Local Video File Pipeline (Mega.nz / external hosting)

For video files hosted on Mega.nz (or any direct-download URL) that need the full
knowledge-base treatment: transcription, summarization, note, and MinIO archival.

## When to use this (vs YouTube pipeline)

| Aspect | YouTube | Mega / external |
|--------|---------|-----------------|
| Download | yt-dlp + cookies | mega.py `download_url()` |
- **Cookies** | yt-dlp + cookies | None (anonymous public links)
- **Diarization** | pyannote 8kHz WAV | pyannote 8kHz WAV (same pipeline)
- **Transcription** | large-v3 int8 | large-v3 int8 (same pipeline)
- **js-runtimes node** | Mandatory | Not needed
| Rate-limiting | `--sleep-requests --limit-rate` | Download speed varies |

## Mega.nz download

```python
from mega import Mega
import shutil

m = Mega()
f = m.download_url('https://mega.nz/file/FILE_ID#FILE_KEY')
shutil.move(f, '/tmp/slug.mp4')
```

**Requirements:** `mega.py` in venv (`pip install mega.py`).
**Pitfall:** `mega.py` 1.0.8 requires `tenacity` upgrade on Python 3.11:
`pip install --upgrade tenacity`

To get file info without downloading:
```python
fid = url.split('/file/')[1].split('#')[0]
fkey = url.split('#')[1]
info = m.get_public_file_info(fid, fkey)
print(info['name'], info['size'])
```

## Two-phase kanban ticket pattern

For video series, split each video into **2 tickets** to keep the LLM context clean:

### Phase A: Download + Diarize + Transcribe (mechanical, no LLM)

```
Ticket: KB: Series Name — Ep.X [DOWNLOAD+TRANSCRIBE]
Assignee: researcher-videos
Body:
  1. Download from Mega with mega.py → /tmp/slug.mp4 (background+wait for >200MB)
  2. Extract dual audio:
     a) ffmpeg -y -i /tmp/slug.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/slug_16k.wav
     b) ffmpeg -y -i /tmp/slug.mp4 -vn -acodec pcm_s16le -ar 8000 -ac 1 /tmp/slug_8k.wav
  3. Diarization: pyannote on 8kHz WAV (mandatory for ALL video) — use canonical `scripts/diarize.py`.
     Follow `references/video-pipeline-global.md` (background+wait mandatory).
  4. Transcription: faster-whisper large-v3 int8 on 16kHz WAV, merged with diarization
     segments — follow `references/video-pipeline-global.md` (background+wait mandatory, 7200s timeout).
     See pipeline-youtube.md step 5b for the full Python script.
  5. Chapter detection NLP (gap > 3s) → add 'chapters' key to JSON
  6. Cleanup: rm /tmp/slug_8k.wav /tmp/slug_16k.wav /tmp/slug_diarization.json
     (KEEP .mp4 + _transcript.json)
  DO NOT: summarize, create note, upload MinIO, or push git.
  NEVER use heartbeats to wait — process wait blocks in a single turn.
```

### Phase B: Summarize + Note + Archive (LLM-focused, clean context)

```
Ticket: KB: Series Name — Ep.X [RESUME+NOTE+ARCHIVE]
Assignee: researcher-videos
Parent: <phase A ticket ID>
Body:
  1. Load /tmp/slug_transcript.json
  2. Follow the resume prompt: load knowledge-base skill →
     references/resume-prompt.md → two-pass summarization
     (Passe 1: extract concepts, Passe 2: full note)
  3. Note in Knowledge base/slug.md (use youtube-note-template)
  4. Upload to MinIO: .mp4, .mp3 (extracted), _transcript.json
  5. Cleanup ALL slug.* from /tmp/
  6. Git push vault
```

### Chaining with --parent

All 30 tickets (15×2) chained in series:
`1A → 1B → 2A → 2B → ... → 15B`

With `max_spawn=1` on researcher-videos, workers run one at a time.
Each --parent ensures the next ticket starts only after the previous completes.

### Cleanup safety

Each phase references its own slug explicitly (`rm /tmp/slug.wav`, then
`rm /tmp/slug.mp4 /tmp/slug.mp3 /tmp/slug_transcript.json`).
Never use `rm /tmp/*.mp4` or wildcards that could delete other workers' files.

## Worker profile

Use `researcher-videos` profile (not `researcher`):
- `max_spawn: 1` — whisper large-v3 + pyannote are CPU-heavy, avoid parallel transcription
- `max_turns: 300` — 30-min video with diarization + large-v3 transcription takes ~90 min on CPU
- `max_iterations: 300`
- Skills: knowledge-base + obsidian (same as researcher)
- Git config copied from host

## Full example: 15-episode series

From grill session (2026-05-24): "Le pouvoir secret du ventre" — 15 × 30min MP4
on Mega.nz, total 6.3 GB. Created 30 tickets on board `default`,
all chained with `--parent`, assigned to `researcher-videos`.

Ticket title format: `KB: Series Name — Ep.NN [DOWNLOAD+TRANSCRIBE]`
and `KB: Series Name — Ep.NN [RESUME+NOTE+ARCHIVE]`

Each ticket receives the Mega URL, slug, and file size in its body.
The worker discovers the episode title from the downloaded filename
(filenames on Mega are the actual titles).
