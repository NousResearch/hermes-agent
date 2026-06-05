# Video Pipeline — Global Rules

Applies to all video pipelines. Platform commands: `pipeline-youtube.md`, `pipeline-instagram.md`, `pipeline-mega.md`.

## Background execution

Run pyannote and whisper with:

```
terminal(background=true, notify_on_complete=true)
process(action="wait", timeout=7200)
```

Wait via `process(wait)` — one blocking call per job.

## Whisper

- Model: `large-v3`, `compute_type='int8'` on CPU
- Language: `language=None` (auto-detect); note content follows detected language
- Threads: `cpu_threads=6` on this server — included in `scripts/transcribe.py`

## Scripts

Invoke `scripts/diarize.py` and `scripts/transcribe.py` from this skill directory.
Both are mandatory — diarization runs first, then transcription, sequentially (never in parallel).

## Pre-flight (before each transcription)

1. Use canonical scripts from the skill — `scripts/diarize.py`, `scripts/transcribe.py`
2. One process per audio file — check `ps aux | grep 'diarize\|transcribe'`
3. Kill orphan workers from destroyed tickets — `ps aux | grep kanban`
4. Ticket body specifies `large-v3` when using kanban workers

## Rate-limiting

- 2 videos per worker session
- yt-dlp: `--sleep-requests 1 --sleep-interval 3 --max-sleep-interval 10 --limit-rate 4M`
- Beyond 2 video URLs: chain tickets with `--parent`
- Kanban tickets: up to 5 URLs, chained with `--parent`
- `max_spawn=1` for `researcher-videos` — `researcher-profile-setup.md`

## Audio extraction

| Target | Format |
|--------|--------|
| whisper | 16kHz WAV mono |
| pyannote | 8kHz WAV mono |

Run diarization and transcription sequentially (RAM).

## pyannote

`>=4.0` with torch ≥2.5. API: `diarization.speaker_diarization.itertracks()`.

On diarization failure: block the task and report — proceed only with diarization output.

## Transcription persistence

Save full transcriptions retrievably:

**Short Reels (≤60s)** — embed at bottom of note:

```markdown
## Full Transcription
<details>
<summary>Click to expand</summary>

> full transcription text here...

</details>
```

**Long videos (>2min)** — upload to MinIO:

```bash
mc cp /tmp/ig_transcript_XXXXX.json minio/knowledge-base/transcripts/<slug>.json
```

Frontmatter: `transcript_file: minio://knowledge-base/transcripts/<slug>.json`

### IG transcription JSON formats

**Format A:** `reel_id`, metadata, `segments`, `full_text`
**Format B:** `segments` + `language` only

Reconstruct `full_text` when missing:

```python
import json, glob
for f in glob.glob('/tmp/ig_transcript*.json'):
    d = json.load(open(f))
    full = d.get('full_text')
    if not full and d.get('segments'):
        full = ' '.join(s['text'] for s in d['segments'])
    print(f"{f}: {len(full)} chars")
```

Numbered fallbacks: `/tmp/ig_transcript_1.json`, `_2.json`, …

## Pipeline settings

- Transcription engine: faster-whisper `large-v3`
- Overlap segments: keep composite labels (`SPEAKER_00 | SPEAKER_01`) with `⚠️ Chevauchement`
- Speaker ID: heuristic from metadata; unmatched → `Unknown`
- Cookies: `/root/.hermes/cookies/yt_cookies.txt` (YouTube), `ig_cookies.txt` (Instagram)
- yt-dlp: `--js-runtimes node` on every call
- After `marker-pdf` install: re-pin packages per `dependencies.md`

Benchmarks: `whisper-model-comparison.md`
