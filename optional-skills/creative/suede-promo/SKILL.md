---
name: suede-promo
description: "Music promotion and AI media production via Suede Labs AI. Two tracks: Creator jobs (social content, clipping, performance posts, contests, landing pages) and Suede services (AI music generation, stems, mastering, vocals, MIDI, lyrics, rights — 17 pay-per-call endpoints, no subscription required)."
version: 1.0.0
author: Suede Labs AI
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3]
  pip: [suede-ai]
metadata:
  hermes:
    tags: [music, promotion, ai-music, stem-separation, mastering, lyrics, creator, content, web3, usdc]
    related_skills: [kanban-video-orchestrator, meme-generation, baoyu-article-illustrator]
    category: creative
    homepage: https://suedeai.ai
---

# Suede Promo

Music promotion and AI media production from Suede Labs AI. Two fulfillment tracks — choose based on the task:

| Track | What it is | When to use |
|-------|-----------|-------------|
| **Suede services** | AI-powered music/media via the `suede-ai` SDK | Instant, automated output: generate tracks, split stems, master audio, write lyrics, isolate vocals |
| **Creator jobs** | Human creator marketplace | Tasks that need a human touch: social content strategy, video clipping briefs, performance posts, contests, landing pages |

---

## Prerequisites

```bash
pip install suede-ai          # Python 3.10+
```

Authentication uses a funded Ethereum wallet — no API keys or subscriptions. Set your private key as an environment variable:

```bash
export SUEDE_WALLET_KEY="0x..."   # Base mainnet wallet with USDC balance
```

Check your available endpoints and pricing:

```python
from suede_ai import SuedeClient
import os

with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"]) as suede:
    print(suede.manifest())   # free — lists all 17 endpoints with prices
```

---

## Track 1: Suede Services (AI-Direct)

All calls settle pay-per-use in USDC on Base. Prices shown per call.

### Quick reference

| Method | What it does | Price |
|--------|-------------|-------|
| `create_music(prompt, duration_seconds)` | Generate an original track | $0.20 |
| `agent_generate()` | Agent-mode music generation | $0.20 |
| `agent_video()` | AI music video | $1.50 |
| `extend()` | Extend an existing track | $0.40 |
| `cover()` | Cover version of a track | $0.40 |
| `voice_cover()` | Voice cover / vocal swap | $0.40 |
| `continue_track()` | Continue from a clip | $0.40 |
| `stems_pro()` | 4-stem split (drums, bass, melody, vocals) | $0.40 |
| `stems_basic()` | 2-stem split (vocals + instrumental) | $0.20 |
| `vox()` | Vocal isolation / acapella | $0.20 |
| `midi()` | Audio → MIDI transcription | $0.10 |
| `wav_master()` | Loudness + EQ mastering | $0.10 |
| `lyric_sync()` | Lyrics synchronized to audio | $0.10 |
| `lyrics()` | Generate lyrics from a prompt | $0.04 |
| `style_coach()` | Genre/style analysis and coaching | $0.02 |
| `rights_lookup(assetHash)` | On-chain rights/provenance check | $0.005 |
| `analyze()` | Audio analysis (BPM, key, energy) | $0.003 |

### Usage pattern

```python
from suede_ai import SuedeClient
import os

with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"]) as suede:

    # Generate a track
    track = suede.create_music(prompt="lo-fi hip hop, rainy day, 90 BPM", duration_seconds=30)
    print(track["assetUrl"])       # CDN URL to the audio file
    print(track["provenance"])     # on-chain attestation / fingerprint

    # Split stems
    stems = suede.stems_pro()
    # stems["drums"], stems["bass"], stems["melody"], stems["vocals"]

    # Write and sync lyrics
    lyrics = suede.lyrics()
    synced = suede.lyric_sync()

    # Master the final mix
    master = suede.wav_master()
    print(master["assetUrl"])
```

### Direct endpoint access

For endpoints not yet exposed as named methods:

```python
result = suede.request("POST", "/v1/style-coach", json={"tags": "lofi, rainy"})
```

### Response structure

Every response includes:
- `assetUrl` — CDN URL to the generated asset
- `provenance` — on-chain fingerprint/attestation via Suede's IP registry

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

```python
# Step 1: Generate the track (Suede service)
with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"]) as suede:
    track = suede.create_music(prompt="upbeat indie pop, summer, 120 BPM", duration_seconds=90)
    lyrics = suede.lyrics()
    master = suede.wav_master()
    stems = suede.stems_basic()   # for remixers

# Step 2: Create a social clip brief (Creator job)
# Hand the track["assetUrl"] to a clipping creator job on Suede Promo
```

### Rights verification before licensing

```python
import hashlib

with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"]) as suede:
    asset_hash = "0x" + hashlib.sha256(open("track.wav", "rb").read()).hexdigest()
    rights = suede.rights_lookup(asset_hash)
    print(rights)   # ownership, license type, attestation timestamp
```

### Style analysis before a contest brief

```python
with SuedeClient(wallet_private_key=os.environ["SUEDE_WALLET_KEY"]) as suede:
    analysis = suede.analyze()    # BPM, key, energy, genre signals
    coach = suede.style_coach()   # genre tags and feedback
    # Use coach output to write the contest creative direction
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
- `agent_video()` at $1.50 is the most expensive call — confirm the prompt is ready before invoking.
- `rights_lookup` takes a hex-encoded SHA-256 hash of the raw audio bytes, not a file path.
- Creator jobs require a human review cycle — set realistic timelines (24–72 hours minimum).
- `manifest()` is free and always returns current pricing — call it first if you're unsure what's available.
