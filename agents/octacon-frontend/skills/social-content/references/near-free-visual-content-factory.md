# Near-Free Visual Content Factory for Pre-Launch Apps

## TL;DR

For high-volume social content (100+/week) on minimal budget, build a **hybrid visual stack**:
- **Free tier**: Pillow static cards + FFMPEG text animations for 80% of volume — costs £0, runs on any VPS (8GB RAM tested).
- **Premium tier**: FAL.ai API (FLUX Schnell £0.0048/image, Wan 2.2 £0.032/sec video) for hero/final content — ~£4-7/month at 400 images + 16 videos.

This is the architecture validated for Sahil Saghir's MatchdayMaestro/Plenishd/CoachOS content engine (May 2026).

## The Hybrid Stack Decision (May 2026 Validation)

After researching every option against an **8GB RAM VPS + £10/month budget**, the only viable architecture is a three-tier hybrid:

| Tier | Tool | Cost | Use For | Volume |
|---|---|---|---|---|
| **Free static** | Pillow | £0 | All routine cards (prediction, quiz, leaderboard, streak, stat mic-drop) | ~350/month |
| **Free motion** | FFMPEG drawtext | £0 | Stat reveals, countdowns, battle teasers, text animations | ~12/month |
| **Premium images** | FAL.ai FLUX-2 Klein 9B | ~£0.005/image | Hero images, launch posts, finals where quality matters | ~50/month |
| **Premium video** | FAL.ai Wan 2.2 480p/720p | ~£0.16-0.32/clip | Hero video clips, app showcases, real motion | ~4/month |

**Monthly total: £4-7** (well under £10 cap)

### Why other options were rejected

| Option | Rejected Because |
|---|---|
| ComfyUI local | Needs 12GB+ VRAM. 8GB VPS cannot run diffusion models. |
| Comfy Cloud | £20+/month subscription. Over budget. Free tier blocks `/api/prompt`. |
| Leonardo.AI free | 150 tokens/day but **UI-only, no API** for automation. |
| Hugging Face free | Tiny recurring credit. Not enough for production volume. |
| Hyperstack GPU rental | £400-1500/month H100 rentals. Content marketing article, not consumer guidance. |
| Nano Banana Pro | £0.024/image via reseller. Not cheaper than direct FAL.ai. |

### Key research sources (May 2026 pricing)

- fal.ai: FLUX-2 Klein 9B $0.006/MP, Wan 2.2 480p $0.04/sec, Wan 2.2 720p $0.08/sec
- Replicate: comparable pricing, no free tier for image/video APIs
- pricepertoken.com: cross-provider comparison verified
- Hermes Agent docs: native `image_gen` toolset wraps FAL.ai, 9 models available via Tool Gateway

### FAL.ai API client pattern

```python
import requests, os

FAL_KEY = os.getenv("FAL_KEY", "")

# Image: sync endpoint for fast models
resp = requests.post(
    "https://fal.run/fal-ai/flux-2/klein/9b",
    json={"prompt": "...", "image_size": "square_hd", "num_images": 1},
    headers={"Authorization": f"Key {FAL_KEY}"},
    timeout=60,
)
image_url = resp.json()["images"][0]["url"]

# Video: queue endpoint for slow models
queue_resp = requests.post(
    "https://queue.fal.run/fal-ai/wan/v2.2/480p",
    json={"prompt": "...", "duration": 5, "aspect_ratio": "16:9"},
    headers={"Authorization": f"Key {FAL_KEY}"},
    timeout=30,
)
# Poll queue.fal.run/{model}/requests/{request_id} until COMPLETED
```

### Hermes native image_gen tool

Hermes Agent has a built-in `image_gen` toolset that wraps FAL.ai:
- Configured via `hermes tools` → Image Generation
- Models: FLUX-2 Klein 9B (default, <1s), FLUX-2 Pro (~6s), GPT-Image (~15s), Ideogram v3 (~5s), etc.
- Two auth paths: Nous Portal subscription (no key needed) or direct FAL_KEY
- Saved to `config.yaml` under `image_gen:` block
- For automated pipelines (cron, content engine), use direct API client instead of interactive tool picker

### When to use which tier

**Always start free:** Generate with Pillow/FFMPEG first. Fall back to FAL.ai only when:
- The content is a hero/launch/milestone post
- The visual needs photorealism or complex composition
- The video needs real motion (not just text animation)
- The draft is marked as "approved" and the user explicitly wants premium treatment

**Never use FAL.ai for:** Routine stat cards, leaderboard updates, daily quiz reveals, text-only posts. The free tools are faster, more authentic, and cost nothing.

```
Hermes cron trigger
    ↓
Claude API (deepseek-v4-pro / kimi-k2.6) generates text
    ↓
Python script renders visuals (Pillow static / FFMPEG animated)
    ↓
Postiz API (DB insert or direct API) for scheduling
    ↓
Telegram bot approval digest
    ↓
User approves → Postiz publishes to X/IG/TikTok/LinkedIn
```

## Visual Types (All Code-Generated)

### 1. Prediction Result Cards (Pillow)

What: Player made X prediction, Y happened, earned Z coins
Template: Dark background, app brand colours, player name + prediction icon + result + coin count
Input: JSON from app data (prediction type, actual outcome, coins earned)

```python
# Skeleton — see templates/content-card-skeleton.py
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (1080, 1080), color='#1A1A2E')
draw = ImageDraw.Draw(img)
# ... brand text + layout logic
img.save(f'{output_dir}/prediction_card_{timestamp}.png')
```

### 2. Quiz Question Cards (Pillow)

What: Multiple-choice question as Instagram carousel slide or single tweet image
Template: Question text at top, 2-4 options as rounded buttons, correct answer reveal optional (for video)
Input: Quiz question JSON (question, options, correct_index)

### 3. Leaderboard Snippets (Pillow)

What: "Top 3 in your league this week" card
Template: Rank number + avatar circle + name + score + small trend arrow (up/down)
Input: Leaderboard array (position, player_name, score, previous_position)

### 4. Streak/Milestone Cards (Pillow)

What: "Day 14 of your prediction streak!" celebration card
Template: Large day counter, streak flame icon, reward text, subtle confetti effect
Input: Streak count, reward type, player name

### 5. PlayerCombination Puzzle (Pillow)

What: T9 grid puzzle reveal (Wordle-style for football players)
Template: 3×3 letter grid, colour-coded feedback (green/yellow/grey), player silhouette
Input: Player name, guess_attempts, feedback matrix

### 6. Stat Mic-Drop Reveals (FFMPEG)

What: Animated text reveal for weekly stats
Example: "89% predicted Liverpool" → number animates up → "89% were WRONG"
Template: Black background, bold numbers, sharp colour transitions, app brand font
Duration: 3-5 seconds
Input: Stat pairs (prediction_percentage, actual_outcome_boolean)

### 7. Quiz Answer Reveals (FFMPEG)

What: Countdown to correct answer
Template: Question text → 3-second countdown → answer slides in with highlight
Duration: 5 seconds
Input: Question, options, correct_index

### 8. Battle Challenge Teasers (FFMPEG)

What: "Your mate just challenged you" motion graphic
Template: Challenger name + avatar → coin stake amount → "Accept?" call to action
Duration: 3 seconds
Input: challenger_name, stake_amount, your_current_rank

## Tools Required (Verify on VPS)

| Tool | Install if missing | Version check |
|---|---|---|
| Pillow | `pip install Pillow` | `python -c "import PIL; print(PIL.__version__)"` |
| FFMPEG | `sudo apt-get install ffmpeg` | `ffmpeg -version` |
| imageio | `pip install imageio[ffmpeg]` | `python -c "import imageio; print(imageio.__version__)"` |

As of May 2026, Sahil's VPS had FFMPEG 6.1.1 and Pillow 12.2.0 installed.

## File Organisation

```
/home/kensei/apps/content-engine/
├── generators/
│   ├── static_cards.py          # Pillow renderers (prediction, quiz, leaderboard, streak)
│   ├── animated_reveals.py      # FFMPEG text animation generators
│   └── utils.py                 # Font loading, brand colours, image export helpers
├── templates/
│   ├── prediction_card.json       # Brand colour + layout config
│   └── quiz_card.json
├── output/                      # Generated assets, timestamped
│   ├── static/
│   └── animated/
├── cron/
│   └── generate_daily_content.py  # Main cron entry point
└── postiz_bridge.py             # Writes generated content to Postiz DB or API
```

## Postiz Integration

Postiz is already self-hosted on Sahil's VPS (port 4007). For high-volume content:

- **DO NOT use POST /api/posts** — server-side validation rejects tested shapes. Verified May 2026: 13+ payload variants all fail with "All posts must have an integration id".
- **Use direct DB insertion** into Postiz PostgreSQL. The compose exposes `postiz-postgres:5432` internally.
- If DB insertion is unavailable, generate content files + metadata JSON and upload manually to Postiz web UI.

```bash
# Direct DB insert example (from postiz-self-hosting skill)
docker exec -i postiz-postgres psql -U postiz-user -d postiz-db-local <<EOF
INSERT INTO "Post" (id,state,"publishDate","organizationId","integrationId",content,title,"group",delay)
VALUES (gen_random_uuid()::text, 'QUEUE', '2026-05-10T09:00:00.000Z', '$ORG_ID', '$INTEGRATION_ID', 'Post text', 'Title', 'mm-matchday-37', 0);
EOF
```

## Budget Summary

- Text generation: Claude API via Ollama Cloud (subscription already active) — £0 marginal
- Image generation: Pillow (already installed) — £0
- Video generation: FFMPEG (already installed) — £0
- Scheduling: Postiz (already self-hosted) — £0
- **Weekly cost for 100+ posts: approximately £0**

## Trigger
Load this reference when the user asks about:
- Low-budget visual content for social media
- Image/video generation for 100+ posts per week
- Pre-launch content engine architecture
- Template-based social content automation
