---
name: acestep
description: AI music generation with ACE-Step 1.5. Use when users want to generate music, create songs, or ask about ACE-Step.
version: 1.0.0
metadata:
  hermes:
    tags: [music, audio, generation, ai, acestep, ace-step, lyrics, songs]
    related_skills: [heartmula, audiocraft, songwriting-and-ai-music]
---

# ACE-Step 1.5 — AI Music Generation

## Check if installed

```bash
ls ~/ACE-Step-1.5/.claude/skills/acestep/SKILL.md 2>/dev/null && echo "INSTALLED" || echo "NOT INSTALLED"
```

## If NOT installed

Clone the project and register its skills:

```bash
git clone https://github.com/ACE-Step/ACE-Step-1.5.git ~/ACE-Step-1.5
```

Add to `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/ACE-Step-1.5/.claude/skills
```

Then `hermes gateway restart`. The project includes detailed setup instructions — refer to its own documentation for environment and API configuration.

## If installed

Use the upstream `/acestep` skill directly. It provides the full music generation workflow including songwriting, generation, lyrics transcription, MV rendering, and more. All usage details are in the project's bundled skills.
