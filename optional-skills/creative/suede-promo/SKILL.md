---
name: suede-promo
description: "Music promotion and AI media production via Suede Labs AI. Two tracks: Creator jobs (social content, clipping, performance posts, contests, landing pages) and Suede services (AI music generation, stems, mastering, vocals, MIDI, lyrics, rights — 22 pay-per-call endpoints, no subscription required)."
version: 1.0.0
author: Jason Colapietro
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  env_vars: [SUEDE_WALLET_KEY]
  commands: [uv]
required_environment_variables:
  - name: SUEDE_WALLET_KEY
    prompt: Base mainnet wallet private key with USDC balance
    help: "Base mainnet wallet holding USDC for Suede pay-per-call services. Never hardcode this key; keep it in an environment variable or secrets manager."
metadata:
  hermes:
    tags: [music, promotion, ai-music, stem-separation, mastering, lyrics, creator, content, web3, usdc]
    related_skills: [kanban-video-orchestrator, meme-generation, baoyu-article-illustrator]
    category: creative
    homepage: https://suedeai.ai
---

# Suede Promo

## When to Use

- Generate AI music tracks, covers, or extensions from a text prompt
- Split stems, isolate vocals, transcribe MIDI, or master audio programmatically
- Write or sync lyrics to a track
- Look up on-chain rights or provenance for an audio asset
- Analyze guitar signal chains or get rig recommendations
- Brief human creators for social content, video clipping, contest design, or landing pages

---

Music promotion and AI media production from Suede Labs AI. Two fulfillment tracks — choose based on the task:

| Track | What it is | When to use |
|-------|-----------|-------------|
| **Suede services** | AI-powered music/media via the `suede-ai` SDK | Instant, automated output: generate tracks, split stems, master audio, write lyrics, isolate vocals |
| **Creator jobs** | Human creator marketplace | Tasks that need a human touch: social content strategy, video clipping briefs, performance posts, contests, landing pages |

---

## Prerequisites

No pip install needed — use `uv run --with suede-ai` to inject the dependency inline for any script. No venv management required.

```bash
export SUEDE_WALLET_KEY="0x..."   # Base mainnet wallet with USDC balance
```

On Windows, run the bash examples in Hermes' bundled Git Bash. In PowerShell, set the key with:

```powershell
$env:SUEDE_WALLET_KEY = "0x..."
```

Use the live app host explicitly. `suede-ai` 0.3.0 defaults to a legacy `.xyz` host, and `httpx` does not follow that manifest redirect by default:

```python
BASE_URL = "https://app.suedeai.ai"
```

Check your available endpoints and pricing:

```bash
uv run --with suede-ai python3 - <<'EOF'
from suede_ai import SuedeClient
import os

BASE_URL = "https://app.suedeai.ai"

with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"], base_url=BASE_URL) as suede:
    print(suede.manifest())   # free — lists all 22 endpoints with prices
EOF
```

---

## Track 1: Suede Services (AI-Direct)

All calls settle pay-per-use in USDC on Base. Prices below are USDC per call. Treat `manifest()` as the source of truth before spending.

### Quick reference

| Method | What it does | USDC/call |
|--------|-------------|-----------|
| `create_music(prompt, duration_seconds=30, style=None)` | Generate an original track | 0.20 |
| `agent_generate(prompt, duration_seconds=30, style=None)` | Agent-mode music generation | 0.20 |
| `agent_video(prompt, duration_seconds=8, aspect_ratio=None, resolution=None)` | AI music video | 1.50 |
| `extend(source_clip_id=None, audio_url=None, prompt=None, title=None, tags=None, continue_at_seconds=None)` | Extend an existing track | 0.40 |
| `cover(source_clip_id=None, audio_url=None, prompt=None, title=None, tags=None, style=None)` | Cover version of a track | 0.40 |
| `voice_cover(audio_url, voice_id=None, pitch_shift=None)` | Voice cover / vocal swap | 0.40 |
| `continue_track(audio_url, prompt=None, continue_at_seconds=None, duration_seconds=None)` | Continue from a clip | 0.40 |
| `stems_pro(audio_url)` | 4-stem split (drums, bass, melody, vocals) | 0.40 |
| `stems_basic(audio_url)` | 2-stem split (vocals + instrumental) | 0.20 |
| `vox(audio_url)` | Vocal isolation / acapella | 0.20 |
| `midi(audio_url)` | Audio to MIDI transcription | 0.10 |
| `wav_master(audio_url)` | Loudness + EQ mastering | 0.10 |
| `lyric_sync(audio_url, lyrics=None)` | Lyrics synchronized to audio | 0.10 |
| `lyrics(prompt, style=None)` | Generate lyrics from a prompt | 0.04 |
| `style_coach(tags, target_count=None)` | Expand style tags into a richer prompt brief | 0.02 |
| `rights_lookup(asset_hash)` | On-chain rights/provenance check | 0.005 |
| `analyze(audio_url)` | Audio analysis (BPM, key, energy, danceability) | 0.003 |
| `prompt_analyze(prompt)` | Extract genre, mood, instrumentation from a prompt | 0.003 |
| `chain_chat(question, asset_hash)` | Plain-language Q&A about on-chain rights/royalties | 0.02 |
| `rig_analyze(audio_url)` | Infer guitar signal chain from audio (pedal order, drive, FX) | 0.10 |
| `rig_oracle(goal, genre=None, budget_usd=None)` | Recommend a full guitar rig for a target tone | 0.10 |
| `rig_roast(goal, gear=None)` | Roast a gear list for laughs | 0.05 |

### Usage pattern

```bash
uv run --with suede-ai python3 - <<'EOF'
from suede_ai import SuedeClient
import os

BASE_URL = "https://app.suedeai.ai"

with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"], base_url=BASE_URL) as suede:

    # Generate a track
    track = suede.create_music(prompt="lo-fi hip hop, rainy day, 90 BPM", duration_seconds=30)
    print(track["assetUrl"])       # CDN URL to the audio file
    print(track["provenance"])     # on-chain attestation / fingerprint

    # Split stems
    stems = suede.stems_pro(audio_url=track["assetUrl"])
    # stems["drums"], stems["bass"], stems["melody"], stems["vocals"]

    # Write and sync lyrics
    lyrics = suede.lyrics(prompt="lo-fi hip hop, rainy day, introspective")
    synced = suede.lyric_sync(audio_url=track["assetUrl"], lyrics=lyrics.get("lyrics"))

    # Master the final mix
    master = suede.wav_master(audio_url=track["assetUrl"])
    print(master["assetUrl"])
EOF
```

### Direct endpoint access

For inspecting raw responses or calling endpoints by path:

```bash
uv run --with suede-ai python3 -c "
from suede_ai import SuedeClient; import os
with SuedeClient(wallet_private_key=os.environ['SUEDE_WALLET_KEY'], base_url='https://app.suedeai.ai') as suede:
    print(suede.request('POST', '/v1/style-coach', json={'tags': 'lofi, rainy'}))
"
```

### Response structure

Asset-returning endpoints (`create_music`, `agent_generate`, `agent_video`, `extend`, `cover`, `voice_cover`, `continue_track`, `stems_pro`, `stems_basic`, `vox`, `wav_master`, `lyric_sync`) return:
- `assetUrl` — CDN URL to the generated asset
- `provenance` — on-chain fingerprint/attestation via Suede's IP registry

Data endpoints (`midi`, `lyrics`, `style_coach`, `analyze`, `prompt_analyze`, `chain_chat`, `rig_analyze`, `rig_oracle`, `rig_roast`, `rights_lookup`) return structured data specific to each endpoint — no `assetUrl`.

---

## Track 2: Creator Jobs (Marketplace)

When the task needs a human creator — content strategy, video editing, contest design, web work — use this track to define and brief the job.

### Job types

| Type | What creators deliver |
|------|----------------------|
| **Social jobs** | Platform-native content (Reels, TikToks, X posts, Threads) around a release or campaign |
| **Clipping** | Short highlight clips cut from longer audio/video for promotional use |
| **Performance posts** | Staged performance content for social — live-feel, authentic |
| **Contests** | Fan engagement contest design, rules, prize structure, creative direction |
| **Website / landing page** | Artist sites, release landing pages, EPK pages |

### How to run a Creator job

1. **Scope the brief** — clarify the release, campaign goal, target platforms, and deadline.
2. **Identify the job type** — use the table above to match.
3. **Draft the job spec** — include:
   - Asset deliverables (formats, dimensions, duration)
   - Brand/style reference (existing tracks, moodboard, palette)
   - Platform requirements (aspect ratios, caption style, hashtag strategy)
   - Timeline and revision rounds
4. **Post the job** — submit via Suede Promo marketplace at [promo.suedeai.ai](https://promo.suedeai.ai).

### Example brief (Clipping job)

```
Job type: Clipping
Track: [CDN URL from create_music or upload]
Deliverables:
  - 3 × 15-second vertical clips (9:16, 1080×1920) for Reels/TikTok
  - 1 × 60-second horizontal clip (16:9) for YouTube Shorts preview
Style: energetic, fast cuts on the drop, text overlays with lyrics
Platform: Instagram Reels, TikTok, YouTube Shorts
Deadline: 72 hours
Revisions: 1 round
```

---

## Common workflows

### Release campaign (both tracks combined)

```bash
uv run --with suede-ai python3 - <<'EOF'
from suede_ai import SuedeClient
import os

BASE_URL = "https://app.suedeai.ai"

# Step 1: Generate the track (Suede service)
with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"], base_url=BASE_URL) as suede:
    track = suede.create_music(prompt="upbeat indie pop, summer, 120 BPM", duration_seconds=90)
    lyrics = suede.lyrics(prompt="upbeat indie pop, summer, optimistic")
    master = suede.wav_master(audio_url=track["assetUrl"])
    stems = suede.stems_basic(audio_url=track["assetUrl"])   # for remixers
    print(track["assetUrl"])

# Step 2: Create a social clip brief (Creator job)
# Hand the track["assetUrl"] to a clipping creator job on Suede Promo
EOF
```

### Rights verification before licensing

```bash
uv run --with suede-ai python3 - <<'EOF'
from suede_ai import SuedeClient
import hashlib, os

BASE_URL = "https://app.suedeai.ai"

with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"], base_url=BASE_URL) as suede:
    asset_hash = "0x" + hashlib.sha256(open("track.wav", "rb").read()).hexdigest()
    rights = suede.rights_lookup(asset_hash)
    print(rights)   # ownership, license type, attestation timestamp
EOF
```

### Style analysis before a contest brief

```bash
uv run --with suede-ai python3 - <<'EOF'
from suede_ai import SuedeClient
import os

BASE_URL = "https://app.suedeai.ai"

with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"], base_url=BASE_URL) as suede:
    analysis = suede.analyze(audio_url="https://cdn.example.com/track.wav")   # BPM, key, energy, genre signals
    coach = suede.style_coach(tags="lo-fi, indie, rainy")   # expand tags into a prompt brief
    print(coach)   # use output to write the contest creative direction
EOF
```

---

## Wallet setup

The SDK requires a Base mainnet wallet with USDC. Never hardcode the private key — always use an environment variable or a secrets manager.

Check balance before running expensive calls:

```bash
# Quick balance check (cast from Foundry, or any Base RPC)
cast call 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 \
  "balanceOf(address)(uint256)" YOUR_WALLET_ADDRESS \
  --rpc-url https://mainnet.base.org
```

If a call fails with a payment error, check that the wallet has sufficient USDC on Base and that the private key env var is set correctly.

---

## Pitfalls

- Never commit `SUEDE_WALLET_KEY` to source control — use `.env` + `python-dotenv` or a secrets manager.
- Bash heredocs in this document assume Hermes' terminal/Git Bash. In native PowerShell, put the Python block in a temporary `.py` file and run `uv run --with suede-ai python temp.py`.
- Pass `base_url="https://app.suedeai.ai"` until the SDK default points at the live `.ai` host without a redirect.
- `agent_video()` at 1.50 USDC is the most expensive call — confirm the prompt is ready before invoking.
- `rights_lookup` takes a hex-encoded SHA-256 hash of the raw audio bytes, not a file path.
- Creator jobs require a human review cycle — set realistic timelines (24–72 hours minimum).
- `manifest()` is free and always returns current pricing — call it first if you're unsure what's available.
