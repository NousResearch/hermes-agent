# Extracting Content from Instagram & JS-Heavy Sites

When `web_extract` is unavailable (Firecrawl credits exhausted, no paid backend),
use direct `curl` requests with search-engine user-agents to extract metadata.

## Instagram Reels

Instagram serves full metadata (title, description, author, likes, comments) in
`<meta>` tags when the request claims to be a search crawler.

### Extract title + description

```bash
curl -sL "https://www.instagram.com/reel/REEL_ID/" \
  -H "User-Agent: Googlebot/2.1" | \
  python3 -c "
import sys, re, html
content = sys.stdin.read()
og_title = re.search(r'<meta property=\"og:title\" content=\"([^\"]+)\"', content)
og_desc = re.search(r'<meta property=\"og:description\" content=\"([^\"]+)\"', content)
if og_title: print('TITLE:', html.unescape(og_title.group(1)))
if og_desc: print('DESC:', html.unescape(og_desc.group(1)))
"
```

### Extract author + engagement

```bash
curl -sL "https://www.instagram.com/reel/REEL_ID/" \
  -H "User-Agent: Googlebot/2.1" | \
  python3 -c "
import sys, re
content = sys.stdin.read()
author = re.search(r'\"author\":\\\"([^\\\"]+)\\\"', content)
if author: print('AUTHOR:', author.group(1))
likes = re.search(r'(\d+[\d,.]+)\s*likes', content)
comments = re.search(r'(\d+[\d,.]+)\s*comments', content)
if likes: print('LIKES:', likes.group(1))
if comments: print('COMMENTS:', comments.group(1))
"
```

### Full extraction (all metadata in one pass)

```bash
curl -sL "https://www.instagram.com/reel/REEL_ID/" \
  -H "User-Agent: Googlebot/2.1" 2>&1 | python3 -c "
import sys, re, html
content = sys.stdin.read()
for field, pattern in [
    ('TITLE', r'<meta property=\"og:title\" content=\"([^\"]+)\"'),
    ('DESC', r'<meta property=\"og:description\" content=\"([^\"]+)\"'),
]:
    m = re.search(pattern, content)
    if m: print(f'{field}: {html.unescape(m.group(1))[:500]}')
author = re.search(r'\"author\":\\\"([^\\\"]+)\\\"', content)
if author: print(f'AUTHOR: {author.group(1)}')
for label, pattern in [('LIKES', r'(\d+[\d,.]+)\s*likes'), ('COMMENTS', r'(\d+[\d,.]+)\s*comments')]:
    m = re.search(pattern, content)
    if m: print(f'{label}: {m.group(1)}')
"
```

## Limitations (text metadata only)

- **curl extracts metadata only** (caption, author, engagement). Full transcript: video pipeline below.
- **Works for public Reels only** — private accounts won't return metadata.
- **Instagram may rate-limit** aggressive crawling. Use sparingly.
- **Some sites block Googlebot** — if 403, try `Bingbot/2.0` or `Twitterbot/1.0`.

## Video extraction + transcription (Instagram Reels)

Use **Method A** (cookies + yt-dlp + faster-whisper). See `video-pipeline-global.md` for whisper/diarization rules.

**URL routing:** `/reel/` → this pipeline. `/p/` → `scripts/ig-carousel-extract.py`. Confirm when label and path disagree (`edge-cases.md`).

**Cookies required:** validate before starting (`edge-cases.md`). If missing:

```
kanban_block(reason="Instagram cookies missing — export from Chrome to /root/.hermes/cookies/ig_cookies.txt")
```

---

### Method A: Cookies + yt-dlp (reliable, requires browser auth)

The user exports their Instagram cookies from Chrome/Firefox on their desktop and copies them to the server.

#### User-side (Mac):

```bash
# Export cookies from Chrome — MUST use a Reel URL (not just instagram.com homepage,
# otherwise the sessionid cookie won't be exported)

# Step 1: Discover which Chrome profile you're on
# Run with -v to see the actual profile path:
yt-dlp --cookies-from-browser chrome --cookies /tmp/test.txt "https://www.instagram.com/" -O done -v 2>&1 | grep "Extracting cookies from"
# Example output: Extracting cookies from: "/Users/.../Chrome/Profile 5/Cookies"
# → use "chrome:Profile 5" in the export command below, NOT just "chrome"

# Step 2: Export with the correct profile
yt-dlp --cookies-from-browser "chrome:Profile 5" --cookies /tmp/ig_cookies.txt "https://www.instagram.com/reel/ANY_REEL_ID/" -O "done"

# If yt-dlp still returns N/A for all profiles, use the Chrome extension fallback:
# Install "Get cookies.txt LOCALLY" from Chrome Web Store → navigate to a Reel →
# click extension → Export → pbpaste > /tmp/ig_cookies.txt

# Step 3: Verify sessionid is present
grep -c sessionid /tmp/ig_cookies.txt  # Must be ≥1

# Step 4: Send to server
scp /tmp/ig_cookies.txt root@<tailscale-ip>:/root/.hermes/cookies/ig_cookies.txt
```

#### Server-side — safe download with rate-limiting:

```bash
# Step 1: List formats and metadata (lightweight)
yt-dlp --cookies /root/.hermes/cookies/ig_cookies.txt \
  --sleep-requests 3 --sleep-interval 5 --max-sleep-interval 15 \
  --limit-rate 2M \
  --print "%(duration)ss | %(like_count)s likes" \
  "https://www.instagram.com/reel/REEL_ID/"

# Step 2: Download (combined stream if available, otherwise best video + audio)
yt-dlp --cookies /root/.hermes/cookies/ig_cookies.txt \
  -f "bv*[height<=720]+ba/b[height<=720]" \
  --merge-output-format mp4 \
  -o "/tmp/ig_reel.mp4" \
  --sleep-requests 3 --sleep-interval 5 --max-sleep-interval 15 --limit-rate 2M \
  "https://www.instagram.com/reel/REEL_ID/"
```

**Rate-limiting is mandatory** — without it, Instagram flags the account. The settings above mimic human scrolling (~8s between requests, 15s between downloads, 2MB/s cap).

#### Then diarize + transcribe (canonical scripts, same as YouTube)

**Follow `references/video-pipeline-global.md`:** `background=true, notify_on_complete=true`
+ `process(wait, timeout=7200)` for ALL pyannote and whisper calls. No foreground. No heartbeats.

```bash
# Extract dual audio (16kHz for whisper, 8kHz WAV for pyannote)
ffmpeg -y -i /tmp/ig_reel.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/ig_audio_16k.wav
ffmpeg -y -i /tmp/ig_reel.mp4 -vn -acodec pcm_s16le -ar 8000 -ac 1 /tmp/ig_audio_8k.wav

# Diarization — canonical script
cp "$SKILL_DIR/scripts/diarize.py" /tmp/diarize.py
terminal(
    "python3 /tmp/diarize.py /tmp/ig_audio_8k.wav /tmp/ig_diarization.json",
    background=True, notify_on_complete=True
)
process(action="wait", timeout=28800)

# Transcription — canonical script
cp "$SKILL_DIR/scripts/transcribe.py" /tmp/transcribe.py
terminal(
    "python3 /tmp/transcribe.py /tmp/ig_audio_16k.wav /tmp/ig_transcript.json 60",
    background=True, notify_on_complete=True
)
process(action="wait", timeout=28800)

# Merge diarization + transcription (inline — same as YouTube pipeline)
python3 -c "
import json
with open('/tmp/ig_diarization.json') as f:
    diar = json.load(f)
with open('/tmp/ig_transcript.json') as f:
    trans = json.load(f)
for seg in trans['segments']:
    seg_mid = (seg['start'] + seg['end']) / 2
    for dia in diar['segments']:
        if dia['start'] <= seg_mid <= dia['end']:
            seg['speaker'] = dia['speaker']
            break
    if 'speaker' not in seg:
        seg['speaker'] = 'Unknown'
with open('/tmp/ig_transcript.json', 'w') as f:
    json.dump(trans, f, indent=2, ensure_ascii=False)
print(f'Merged: {len(trans[\"segments\"])} segments with speaker labels')
"

# Cleanup 8kHz audio + diarization JSON
rm /tmp/ig_audio_8k.wav /tmp/ig_diarization.json
```

---

### Method B: CDN-direct (⛔ DEPRECATED — DO NOT USE)

> **Workers: if you have no cookies, BLOCK the task. Do NOT try Method B.** It's unreliable and will waste your turn budget. This section is kept only for reference.

When Instagram exposes CDN URLs in the page's `sharedData`, you can bypass yt-dlp's download restriction by fetching the streams directly.

#### Step 1: List available formats

```bash
yt-dlp --print "%(formats)s" "https://www.instagram.com/reel/REEL_ID/" 2>/dev/null
```

yt-dlp can list CDN URLs even when download is blocked by login requirement. Look for DASH audio + DASH video streams. **If this returns empty, switch to Method A.**

#### Step 2: Download audio + video streams directly

The CDN URLs from step 1 remain valid for a few minutes. Grab the audio and a 720p video stream:

```bash
# Audio (m4a DASH stream)
curl -s -o /tmp/ig_audio.m4a -H "User-Agent: Mozilla/5.0..." -H "Referer: https://www.instagram.com/" "<AUDIO_CDN_URL>"

# Video (720x1280 DASH stream)
curl -s -o /tmp/ig_video.mp4 -H "User-Agent: Mozilla/5.0..." -H "Referer: https://www.instagram.com/" "<VIDEO_CDN_URL>"
```

#### Step 3: Merge with ffmpeg

```bash
ffmpeg -y -i /tmp/ig_video.mp4 -i /tmp/ig_audio.m4a -c copy -shortest /tmp/ig_reel.mp4
```

Then transcribe using the same whisper pipeline as Method A (above).

#### Clean up CDN-direct temp files

```bash
rm /tmp/ig_audio.m4a /tmp/ig_video.mp4 /tmp/ig_reel.mp4 /tmp/ig_audio_16k.wav
```

### Prerequisites

**Core pipeline (always needed):**
- `yt-dlp` — installed via pip
- `faster-whisper` — installed via pip (SDK, no CLI needed). Model: `large-v3` (mandatory, downloaded from HF Hub)
- `ffmpeg` — system package
- Whisper model: download with `snapshot_download('Systran/faster-whisper-large-v3', cache_dir='/root/.cache/huggingface')` (no HF_TOKEN needed for public model)

**Diarization (mandatory for ALL video content):**
- `pyannote.audio` — **Must use >=4.0** when torch >=2.5. Old API: `Pipeline().itertracks()`. New 4.x API: `Pipeline().speaker_diarization.itertracks()`. Pin: `pip install 'pyannote.audio>=4.0'`
- **HF_TOKEN** — required for pyannote model download. Source from `researcher-videos` profile: `export HF_TOKEN=$(grep -oP 'HF_TOKEN=\K[^#\n]+' /root/.hermes/profiles/researcher-videos/.env | head -1)`
- CPU-only performance: ~3x realtime (~40 min for 13-min audio), 350% CPU

**Image carousels (/p/ posts):**
- `playwright` + chromium: `pip install playwright && python -m playwright install chromium`
- Script: `scripts/ig-carousel-extract.py URL` (extracts first 2 slides via alt text)

**Book/PDF extraction:**
- `ebooklib` — ePub extraction
- `pymupdf` — PDF text extraction  
- `marker-pdf` — OCR for scanned PDFs
- `mega.py` — Mega.nz downloads
- `beautifulsoup4` — HTML parsing for curl fallbacks

**⚠️ dependency cascade pitfall:** `marker-pdf` downgrades `openai`, `anthropic`, `tenacity`, `Pillow`, `huggingface-hub`, `tokenizers` — breaking hermes-agent. After installing marker-pdf: `pip install 'openai==2.24.0' 'anthropic==0.87.0' 'tenacity==9.1.4' 'Pillow==12.2.0' 'huggingface-hub==1.16.1' 'tokenizers==0.23.1'`. Also patch `transformers/dependency_versions_table.py`: change `tokenizers>=0.22.0,<=0.23.0` to `<=0.24.0` (faster-whisper requires 0.23.1).

**Quick install all optional deps:**
```bash
pip install ebooklib pymupdf marker-pdf mega.py beautifulsoup4
pip install 'openai==2.24.0' 'anthropic==0.87.0' 'tenacity==9.1.4' 'Pillow==12.2.0' 'huggingface-hub==1.16.1' 'tokenizers==0.23.1'
pip install 'pyannote.audio>=4.0'
pip install playwright && python -m playwright install chromium
sed -i 's/tokenizers>=0.22.0,<=0.23.0/tokenizers>=0.22.0,<=0.24.0/' $(find / -path "*/transformers/dependency_versions_table.py" -not -path "*/__pycache__/*" 2>/dev/null)
```
- `pyannote.audio` — installed via pip. **Must use >=4.0** when torch >=2.5 (torch 2.5+ removed `AudioMetaData` and `list_audio_backends`, which pyannote 3.x requires). Pin: `pip install 'pyannote.audio>=4.0'` (~2 min, pulls torch 2.12+ deps). The generic `pip install pyannote.audio` resolves to 3.x which will crash on modern torch.
  
  **⚠️ pyannote 4.x API change:** `pipeline()` returns `DiarizeOutput` (not `Diarization`). The `.itertracks()` method was removed. Use:
  ```python
  diarization = pipeline('audio.wav')
  for segment in diarization:
      print(segment.start, segment.end, segment.label)
  ```
  
  **⚠️ dependency cascade pitfall:** installing `marker-pdf` may silently downgrade `openai` (2.24.0→1.x), `anthropic`, `tenacity`, `Pillow`, and `huggingface-hub` — breaking hermes-agent. After installing marker-pdf, always restore: `pip install 'openai==2.24.0' 'anthropic==0.87.0' 'tenacity==9.1.4' 'Pillow==12.2.0' 'huggingface-hub==1.16.1' 'tokenizers==0.23.1'`

- `playwright` + chromium — needed for `/p/` image carousel extraction: `pip install playwright && python -m playwright install chromium`
- **HF_TOKEN** — required for pyannote diarization model download. The main `~/.hermes/.env` may have it commented out. Source it from `researcher-videos` profile: `export HF_TOKEN=$(grep -oP 'HF_TOKEN=\K[^#\n]+' /root/.hermes/profiles/researcher-videos/.env | head -1)`
- **gh CLI path conflict** — the Python `gh` package installed by uv shadows `/usr/bin/gh` (GitHub CLI). Use `/usr/bin/gh` explicitly for git operations (clone, auth), or uninstall the Python package: `pip uninstall gh -y`

**Performance expectations (CPU-only, no GPU):**
- pyannote diarization on 13-min audio: ~3x realtime (~40 min), 350% CPU
- faster-whisper large-v3 with int8 on 13-min audio: ~2-3x realtime (~30 min), 200-300% CPU
- `pyannote.audio` — installed via pip. **Must use >=4.0** when torch >=2.5 (torch 2.5+ removed `AudioMetaData` and `list_audio_backends`, which pyannote 3.x requires). Pin: `pip install 'pyannote.audio>=4.0'` (~2 min, pulls torch 2.12+ deps). The generic `pip install pyannote.audio` resolves to 3.x which will crash on modern torch.
- `playwright` + chromium — needed for `/p/` image carousel extraction: `pip install playwright && python -m playwright install chromium`
- **HF_TOKEN** — required for pyannote diarization model download. The main `~/.hermes/.env` may have it commented out. Source it from `researcher-videos` profile: `export HF_TOKEN=$(grep -oP 'HF_TOKEN=\K[^#\n]+' /root/.hermes/profiles/researcher-videos/.env | head -1)`
- **gh CLI path conflict** — the Python `gh` package installed by uv shadows `/usr/bin/gh` (GitHub CLI). Use `/usr/bin/gh` explicitly for git operations (clone, auth), or uninstall the Python package: `pip uninstall gh -y`

## When to use this vs web_extract

| Method | When |
|--------|------|
| `web_extract` (Firecrawl) | Firecrawl available, simple extraction. Gets metadata + first 2-3 slides. No actions (Fire Engine is cloud-only). |
| **Image Posts** (Playwright + cookies) | Any Instagram `/p/` carousel — extracts first 2 slides with full alt text. Slides 3+ blocked by anti-bot. See `scripts/ig-carousel-extract.py`. |
| **Video Method A** (cookies + yt-dlp) | Any Instagram Reel — reliable, requires browser cookies from user |
| **Video Method B** (CDN-direct) | ⛔ DEPRECATED — do not use. Block task instead if cookies are missing. |
| `curl + Googlebot` | Firecrawl down, no Playwright, simple metadata extraction (caption, likes, comments). Only 1 image. |
| Browser (`browser_navigate`) | JavaScript-heavy SPA, need to click/interact, CAPTCHAs |

## Instagram Image Posts (carousels, `/p/` URLs)

### ⚠️ Pitfall: yt-dlp does NOT work for image carousels

yt-dlp works great for Reels (videos) but **fails completely for image posts**.
It returns `"No video formats found!"` because `/p/` posts don't have video
streams. Thumbnails are often empty too. Use **yt-dlp only for metadata**
(caption, likes, author, slide count via `playlist_count`), then use the
Playwright script below for the actual images.

### ⚠️ Pitfall: Firecrawl only gets 2-3 slides

Self-hosted Firecrawl lacks **Fire Engine** (cloud-only paid feature), so
actions (click, scroll) are unavailable. Even with long `waitFor` values,
it only captures the initially visible slides plus some related posts.
The results are inconsistent across runs.

### Method: Playwright + cookies (2 slides guaranteed, $0 cost)

Instagram loads only the first 1-2 carousel slides on initial page load. The remaining
slides are fetched via API calls triggered by carousel navigation — which is
**blocked by Instagram's anti-bot detection in headless browsers** (even with
`--headless=new`, touch events, mobile UA, and anti-webdriver scripts). The hard cap
is 2 slides.

The full pipeline for Instagram image carousels:

1. **yt-dlp → metadata**: caption, likes, author, slide count (`playlist_count`)
2. **Playwright script → first 2 slide images + alt text**: see `scripts/ig-carousel-extract.py`
3. **No vision_analyze needed**: Instagram puts the full slide text in each image's `alt` attribute

Total cost: **$0** (no LLM calls, no vision API).  
⚠️ Slides 3+ are NOT extractable. If critical content is in later slides,
the user must screenshot them manually.

Install Playwright once:
```bash
npx playwright install chromium
/usr/local/lib/hermes-agent/venv/bin/pip install playwright
```

Run the extraction:
```bash
/usr/local/lib/hermes-agent/venv/bin/python scripts/ig-carousel-extract.py "https://www.instagram.com/p/POST_ID/"
```

Output: JSON array with `{src, alt, width, height}` for the first 2 slides.
The `alt` field contains the full text content of the slide.

**Filtering tip**: Instagram loads recommended posts from other accounts below
the carousel. Some extracted images may be from those accounts (e.g.
`@soulrichsociety` mixed with `@sdcaw_spirituality`). Filter by checking the
alt text for the target account's handle.

### Fallback: Firecrawl (2-3 slides)

Self-hosted Firecrawl without Fire Engine: use `waitFor: 5000` to get the same
2-3 slides. Requires Firecrawl running. Use the Playwright script above instead
— simpler dependencies, same result.

### Fallback: Googlebot UA curl (no cookies, 1 slide only)

When Playwright is unavailable and you only need the first slide:

```bash
curl -sL "https://www.instagram.com/p/POST_ID/" \
  -H "User-Agent: Googlebot/2.1" | python3 -c "
import sys, re, html
content = sys.stdin.read()
og_img = re.search(r'<meta property=\"og:image\" content=\"([^\"]+)\"', content)
if og_img: print('IMAGE:', og_img.group(1))
og_title = re.search(r'<meta property=\"og:title\" content=\"([^\"]+)\"', content)
if og_title: print('TITLE:', html.unescape(og_title.group(1)))
"
```

CDN URLs expire in minutes — download immediately with a Referer header:
```bash
curl -sL -o /tmp/ig_image.jpg -H "Referer: https://www.instagram.com/" "<CDN_URL>"
```

### Vision troubleshooting

**DeepSeek V4 vision incompatibility.** DeepSeek V4 has native multimodal support but
only through the Anthropic API format (`type: "image"` blocks on
`https://api.deepseek.com/anthropic`). The OpenAI-compatible endpoint
(`https://api.deepseek.com/v1`) rejects `image_url` content with:
```
unknown variant `image_url`, expected `text`
```
Hermes `vision_analyze` sends images in OpenAI format — so DeepSeek cannot be used
as `vision.provider`. Use Anthropic with `claude-haiku-4-5-20251001` (verified
working 2026-05-24, $1/MTok input, $5/MTok output, ~$0.005/image).

**Fixing the vision model (session crash recovery):**
```bash
hermes config set auxiliary.vision.model claude-haiku-4-5-20251001
hermes config set auxiliary.vision.provider anthropic
```
Then `/reset` or restart the session — the vision tool caches the model at import time.
$5/MTok output, ~$0.005/image) or OpenRouter instead. Verify the model exists:
```bash
curl -s https://api.anthropic.com/v1/models/claude-haiku-4-5-20251001 \
  -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"
```

**Config dual-section trap.** `config.yaml` has TWO `vision:` sections:
- `auxiliary.vision:` (line ~164) — the ACTUAL config used by the vision tool
- `vision:` (bottom of file) — a duplicate, NOT used

`hermes config set vision.model XYZ` targets the WRONG section (the unused bottom
one). Always use `hermes config set auxiliary.vision.model XYZ` instead.

Verify with: `grep -A 3 'auxiliary.vision:' ~/.hermes/config.yaml`

### Cookie validation

Instagram cookies need a `sessionid` for authenticated access. `csrftoken` + `mid`
alone are NOT sufficient — yt-dlp will fail with "login required".

```bash
# Check if cookies have sessionid
grep -c 'sessionid' /root/.hermes/cookies/ig_cookies.txt  # Must be ≥1
```

**Export tip**: when exporting cookies from Chrome, you MUST use a Reel URL
(not just the Instagram homepage). Otherwise the `sessionid` cookie won't be
included:
```bash
# Correct — uses a Reel URL:
yt-dlp --cookies-from-browser chrome --cookies /root/.hermes/cookies/ig_cookies.txt "https://www.instagram.com/reel/ANY_REEL_ID/" -O "done"
# Wrong — homepage won't export sessionid:
yt-dlp --cookies-from-browser chrome --cookies /root/.hermes/cookies/ig_cookies.txt "https://www.instagram.com/" -O "done"
```

Re-export from a browser with an active Instagram session if `sessionid` is missing.

## Rate-limiting for Instagram

- yt-dlp with cookies: `--sleep-requests 3 --sleep-interval 5 --max-sleep-interval 15 --limit-rate 2M`
- 2–3 Reels per worker session
- Chain batches with `--parent` (4–5 URLs per ticket)
- Cookies persist at `/root/.hermes/cookies/ig_cookies.txt` — re-export from Chrome when session expires
