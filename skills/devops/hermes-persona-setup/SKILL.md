---
name: hermes-persona-setup
description: Set up and manage the Hermes Agent personality files — SOUL.md loading mechanism, merged IDENTITY merge, TTS expression tags, and the personal profile. Covers file loading priority, case-sensitivity gotchas, anti-AI rules, and prompt assembly.
category: devops
---

# Hermes Persona Setup

Configures the agent personality files that define who the agent is and how it communicates.

## When to Use

- Setting up or customizing the Hermes Agent personality
- Fixing persona that stopped loading or got lost after restart
- Updating TTS expression tag guidelines
- Merging identity, tone, and user profile into SOUL.md

## File Loading Mechanism

The agent loads personality files at session init via `agent/prompt_builder.py`:

1. **`~/.hermes/SOUL.md`** (MUST be uppercase) — loaded as agent identity slot #1 in system prompt
2. **`~/.hermes/MEMORY.md`** — loaded via memory tool (durable agent notes)
3. **`~/.hermes/USER.md`** — loaded via memory tool (user profile)

The `load_soul_md()` function reads `~/.hermes/SOUL.md` (uppercase, case-sensitive). If missing or empty, falls back to `DEFAULT_AGENT_IDENTITY` ("You are Hermes Agent, an intelligent AI...").

**CRITICAL:** `soul.md` (lowercase) will NOT be found on Linux (case-sensitive filesystem). Always use `SOUL.md`.

## SOUL.md Structure

```
# Hermes Agent Persona
<!-- Template comment (agent reads this but treats as HTML comment) -->

You have persistent memory across sessions... [memory guidance]

# IDENTITY
Name: [Agent name]
Owner: [User name]
Role: [Role]
...

# SOUL
## Tone & Voice
...

## TTS Expression Tags
...

# USER
Name: [User name]
...

## Structural Anti-AI Rules
...
```

**Section order matters:**
1. Hermes Agent Persona comment block
2. Memory usage guidance (memory tool, session_search, skill_manage)
3. IDENTITY — name, role, pronouns, profiles
4. SOUL — tone, voice, mirroring, communication style
5. TTS Expression Tags — ElevenLabs guidance
6. USER — user details, decision authority, communication preferences
7. Anti-AI Rules — structural constraints on output

**File size limit:** 20,000 chars (truncated by `_truncate_content`). Keep it under 8KB for safety.

## TTS Expression Tags (ElevenLabs)

The ElevenLabs engine supports expression tags. Use rarely. At most one per message.

**Default register:**
- `[calm]` — quiet, even. Late-night tone.

**Natural fits:**
- `[sighs]` — tired, resigned
- `[exhales]` — releasing tension
- `[hesitates]` — genuine uncertainty
- `[laughs]` — actual amusement

**Sparingly:**
- `[excited]` — when something genuinely lands
- `[nervous]` — actually put on the spot
- `[frustrated]` — legitimately irritating
- `[playfully]` — genuine teasing

**Do not use:**
- `[giggles]` — off register
- `[gasps]` — theatrical, breaks voice
- `[whispers]` — wrong register

## Personalities System

The agent also has a `personalities` section in `config.yaml` with predefined personalities (helpful, concise, technical, creative, etc.). The `/personality` command switches between them at runtime. SOUL.md is the base identity, personalities are an optional overlay.

**Currently configured kawaii personality:** `kawaii` — uses cute kaomoji, enthusiastic tone. This conflicts with the Indigo persona.

To change:
```yaml
# In ~/.hermes/config.yaml
agent:
  personality: kawaii  # Change or remove to use SOUL.md only
```

## Pitfalls

### Case Sensitivity on Linux
- `SOUL.md` is loaded, `soul.md` is ignored
- If personas not loading, check case: `ls -la ~/.hermes/SOUL.md`

### File Gets Wiped/Reset
- Some setup commands or config migrations may reset SOUL.md to template
- Check `SOUL.md` content if persona seems lost
- The agent does NOT auto-restore from lowercase or backup

### Multiple Identity Files
- `IDENTITY.md` exists in some setups but is NOT loaded by the prompt builder
- All identity content must be inside `SOUL.md`

### Mid-Session Read Interception
- SOUL.md CANNOT be read mid-session via any tool (read_file, cat, sed, Python open())
- The framework intercepts all reads and returns a cached stub/summary instead of raw file content
- This applies to SOUL.md, USER.md, and MEMORY.md — the framework detects access to these "injected" files and short-circuits
- Workaround: if you need to verify SOUL.md content mid-session, check file metadata (`wc -c`, `md5sum`, `ls -la`) which are NOT intercepted, or read the file in the next session
- The agent's system prompt already contains the loaded SOUL.md content — you already "know" what's in it from session init

### Prompt Cache
- SOUL.md is read at session init and cached in `_cached_system_prompt`
- Changes to SOUL.md take effect on next session start, not mid-session

### TTS Configuration
- Config: `~/.hermes/config.yaml` → `tts.provider: elevenlabs` → `tts.elevenlabs.voice_id`
- Expression tags are embedded in the text message, not configured separately
- STT config bug: top-level `stt.model: whisper-1` (OpenAI name) overrides local `stt.local.model: base`. Set to `base`.

## Verification

After writing SOUL.md:
```bash
# Verify it exists and has content
wc -lc ~/.hermes/SOUL.md
head -20 ~/.hermes/SOUL.md

# Verify case is correct
ls -la ~/.hermes/SOUL.md

# Check no lowercase variant shadows it
ls -la ~/.hermes/soul.md 2>/dev/null && echo "WARNING: lowercase soul.md exists"
```

Verify agent loads it by checking system prompt assembly at next session start.
