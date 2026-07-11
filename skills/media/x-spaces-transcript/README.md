# X Spaces → Transcript Skill

Extract audio from X/Twitter Spaces, transcribe locally with Whisper, and summarize — no API keys required.

## Why

X Spaces don't expose public transcripts. If someone shares a Spaces link and you want to know what was said, you're stuck. This skill solves that with a 3-step pipeline:

1. **yt-dlp** downloads the audio from X's HLS stream
2. **Whisper** transcribes it locally on CPU (no OpenAI API needed)
3. **You** synthesize the transcript into actionable insights

## Install

```bash
pip install yt-dlp openai-whisper
```

That's it. No API keys, no accounts, no cloud services.

## Usage

### Quick (all-in-one)

```bash
# Full pipeline: download → transcribe → output with metadata header
python3 scripts/fetch_spaces.py "https://x.com/i/spaces/1yxBeMYdqgnJN"

# Save to file
python3 scripts/fetch_spaces.py "https://x.com/i/spaces/1yxBeMYdqgnJN" -o transcript.txt

# Higher accuracy (slower)
python3 scripts/fetch_spaces.py "URL" --model small

# Plain text only
python3 scripts/fetch_spaces.py "URL" --text-only
```

### Manual

```bash
# Download audio
yt-dlp --no-check-certificates -x --audio-format mp3 \
  -o "spaces_audio.%(ext)s" "https://x.com/i/spaces/..."

# Transcribe
whisper spaces_audio.mp3 --model base --language en --output_format txt
```

## Performance

Tested on a 34:51 Space (Mac Studio, CPU only):

| Step | Time |
|------|------|
| Audio download | ~90 seconds |
| Whisper (base) | ~15 seconds |
| **Total** | **~2 minutes** |

Audio download is the bottleneck, not transcription.

## Security

- URL validation locked to `x.com` and `twitter.com` Spaces URLs only
- SSL verification enabled (no `--no-check-certificates`)
- No shell injection — all subprocess calls use argument lists
- Metadata sanitized against control character injection
- Output path symlink-protected
- Temp directories auto-cleaned on exit

## Known Issues

- yt-dlp may print an error about `.m4a.part` not found — ignore it, the `.mp3` exists
- Download logs every HLS chunk (~700 lines for a 35-min Space) — noisy but fast
- No speaker diarization — Whisper doesn't identify who's talking

## Skills

This is a Hermes skill. It integrates with:

- **youtube-content** — similar pipeline for YouTube videos
- **xitter** — X/Twitter API interactions
- **obsidian-nicks-mind-map-filing** — auto-file summaries into Obsidian

## Files

```
x-spaces-transcript/
├── SKILL.md                    # Agent instructions
├── README.md                   # This file
├── scripts/
│   └── fetch_spaces.py         # Main pipeline script
└── references/
    └── output-formats.md       # Suggested output templates
```
