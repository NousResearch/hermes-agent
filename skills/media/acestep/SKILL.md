---
name: acestep
description: Set up and use ACE-Step 1.5 for AI music generation. Guides skill installation, API configuration (cloud or local), and music creation. Use when users mention generating music, creating songs, AI music, ACE-Step, or want an open-source Suno alternative.
version: 1.0.0
metadata:
  hermes:
    tags: [music, audio, generation, ai, acestep, ace-step, lyrics, songs, suno-alternative]
    related_skills: [heartmula, audiocraft, songwriting-and-ai-music]
---

# ACE-Step 1.5 — AI Music Generation

ACE-Step 1.5 is the state-of-the-art open-source music generation model (MIT license) by StepFun. It supports text-to-music, cover generation, repainting, vocal-to-BGM, and 50+ languages.

- **Cloud API**: https://acemusic.ai (free API key, no GPU required)
- **Repo**: https://github.com/ACE-Step/ACE-Step-1.5
- **Online Demo**: https://acemusic.ai

## Step 1 — Check if ACE-Step skills are installed

```bash
ls ~/ACE-Step-1.5/.claude/skills/acestep/SKILL.md 2>/dev/null && echo "INSTALLED" || echo "NOT INSTALLED"
```

- **INSTALLED** → go to **Step 3 (Use)**.
- **NOT INSTALLED** → go to **Step 2 (Install)**.

## Step 2 — Install ACE-Step skills

The ACE-Step project ships a full skill suite (generation, songwriting, lyrics transcription, MV rendering, thumbnails, docs). Clone the repo to get them:

```bash
cd ~/
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
```

Then register the skills in Hermes by adding to `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/ACE-Step-1.5/.claude/skills
```

Restart the gateway to pick them up:

```bash
hermes gateway restart
```

This makes the following upstream skills available:

| Skill | Purpose |
|-------|---------|
| `acestep` | Core music generation — text-to-music, cover, repainting |
| `acestep-songwriting` | Songwriting guide — captions, lyrics, BPM/key/duration |
| `acestep-lyrics-transcription` | Transcribe audio → timestamped LRC/SRT |
| `acestep-simplemv` | Render music videos with waveform and synced lyrics |
| `acestep-thumbnail` | Generate album art / MV backgrounds via Gemini |
| `acestep-docs` | Documentation, setup guides, troubleshooting |

These skills are maintained by the ACE-Step project. Run `cd ~/ACE-Step-1.5 && git pull` to update them.

## Step 3 — Use ACE-Step

Once the skills are registered, use the upstream `/acestep` skill for all music generation. It supports two modes:

### Cloud API (default, recommended)

No GPU or local setup needed. Configure with a free API key from https://acemusic.ai/api-key:

```bash
cd ~/ACE-Step-1.5/.claude/skills/acestep/
./scripts/acestep.sh config --set api_url "https://api.acemusic.ai"
./scripts/acestep.sh config --set api_key "YOUR_KEY"
./scripts/acestep.sh config --set api_mode completion
```

### Local API (optional, requires GPU)

For users who want fully offline generation. Requires Python 3.11+ and `uv`:

```bash
cd ~/ACE-Step-1.5
uv sync
uv run acestep-api    # starts local API at http://localhost:8001
```

Then configure the skill to point to localhost:

```bash
cd ~/ACE-Step-1.5/.claude/skills/acestep/
./scripts/acestep.sh config --set api_url "http://127.0.0.1:8001"
./scripts/acestep.sh config --set api_key ""
```

### Generate music

```bash
cd ~/ACE-Step-1.5/.claude/skills/acestep/

# Check API health
./scripts/acestep.sh health

# Generate with lyrics (recommended)
./scripts/acestep.sh generate -c "pop, female vocal, piano" -l "[Verse] Your lyrics..." --duration 120

# Quick exploration
./scripts/acestep.sh generate -d "A cheerful song about spring"

# Cover from existing audio
./scripts/acestep.sh cover input.mp3 -c "Jazz cover" -l "[Verse] New lyrics..." --duration 120
```

Refer to the upstream `/acestep` skill for full usage details, parameters, and MV production pipeline.

## Links

- GitHub: https://github.com/ACE-Step/ACE-Step-1.5
- Cloud API: https://acemusic.ai
- Models: https://huggingface.co/ACE-Step/Ace-Step1.5
- Technical Report: https://arxiv.org/abs/2602.00744
- Discord: https://discord.gg/PeWDxrkdj7
