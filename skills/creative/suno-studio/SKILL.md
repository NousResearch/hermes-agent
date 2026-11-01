---
name: suno-studio
description: "Suno AI music generation via browser automation: parse user intent into expert prompts with [metatags], then generate and retrieve songs on suno.com. Supports single-song and DJ queue modes."
version: 1.0.0
author: Jpalmer95
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [suno, music, generation, browser, automation, dj, radio, prompt-engineering]
    related_skills: [songwriting-and-ai-music, heartmula, spotify]
---

# Suno Studio — AI Music Generation via Browser Automation

## Overview

Suno Studio automates the full Suno AI music generation workflow through Hermes'
built-in browser tools. It combines expert Suno prompt engineering (loaded from the
`songwriting-and-ai-music` skill) with browser automation to create, manage, and
retrieve AI-generated songs — no API key needed, just a persistent browser session
logged into suno.com.

Two modes:
- **Single Song**: User describes what they want → agent engineers a detailed Suno
  prompt → fills in Custom Mode on suno.com/create → generates → retrieves audio.
- **DJ Mode**: User provides a vibe arc → agent generates a sequence of songs with
  energy progression, BPM/key transitions, and crossfade-ready structure.

## When to Use

- User asks to "make me a song about..." or "generate music for..."
- User describes a vibe, mood, or scenario and wants music for it
- User says "play me some music" or "DJ mode"
- User wants to extend, remix, or create stems from an existing Suno generation
- User mentions Suno, AI music generation, or wants Suno-quality output

## Prerequisites

1. **Persistent browser with Suno login**: Launch Brave/Chrome with CDP:
   ```bash
   brave-browser --remote-debugging-port=9222 \
     --user-data-dir=~/.hermes/chrome-debug \
     --no-first-run --no-default-browser-check &
   ```
   Config: `browser.cdp_url: http://127.0.0.1:9222` in `~/.hermes/config.yaml`
2. **One-time manual login**: The first time, navigate to suno.com in the CDP
   browser and complete Google OAuth. Login persists in `~/.hermes/chrome-debug/`.
3. **Songwriting skill**: Always load `songwriting-and-ai-music` first for
   [metatags], style formulas, phonetic tricks, and advanced Suno prompting.

## Workflow: Single Song Generation

### Step 1: Parse User Intent

Take the user's description and extract:
- **Concept/theme**: What is the song about?
- **Mood/vibe**: Emotional tone (chill, energetic, melancholic, euphoric)
- **Genre/style**: Preferred genre or genre-blend
- **Specifics**: Any details (instruments, BPM, key, vocal style, era)
- **Length preference**: Short (~1 min), standard (~3 min), long (~4 min)

If the user gives vague input ("something chill"), ask 1-2 clarifying questions
MAX. Don't over-interview. Default to sensible choices and generate.

### Step 2: Engineer the Suno Prompt

Load the `songwriting-and-ai-music` skill for reference. Build three fields:

#### Title Field
- Short, evocative, no more than 8 words
- Should capture the emotional core of the song

#### Style of Music Field (up to 1000 chars — USE them)
Follow the formula from the songwriting skill:
```
Genre + Mood + Era + Instruments + Vocal Style + Production + Dynamics
```

CRITICAL: Describe the DYNAMIC JOURNEY, not just a genre list:
```
"Cinematic orchestral electronica, brooding minor key, starts with
 sparse analog synth pads and distant thunder. Layers in tribal
 percussion building through verse 1. Chorus explodes with full
 orchestra and driving bass. Bridge strips to solo cello and whispered
 vocals. Final chorus returns with added choir and brass fanfare.
 Outro fades on detuned piano and white noise. Male vocalist with
 smoky baritone, starts restrained, builds to raw power."
```

Tips:
- NO artist names (Suno rejects them). Describe the sound instead.
- Specify BPM range: "110-120 BPM" or "slow 70 BPM"
- Build a vocal PERSONA: gender, range, texture, emotional arc
- Use Exclude Styles field for what you don't want
- Put key metatags in BOTH style field AND lyrics for reinforcement

#### Lyrics Field (up to ~3000 chars)
Always use Custom Mode. Always include structural [metatags].

Structure template:
```
[Intro]
[Instrumental]

[Verse 1]
(lyrics here)

[Pre-Chorus]
(lyrics here)

[Chorus]
(hook lyrics)

[Verse 2]
(lyrics here)

[Bridge]
[Emotional Climax]
(lyrics here)

[Final Chorus]
[Powerful]
(hook lyrics with variation)

[Outro]
[Fading]
```

Rules from songwriting skill:
- ALL CAPS = louder/more intense delivery
- Spell out numbers: "twenty four seven" not "24/7"
- Space acronyms: "A I" not "AI"
- Phonetic respelling for unusual words
- Ellipses for dramatic pauses: "I... need... you..."
- Vowel extension with hyphens: "lo-o-o-ove"
- 5-8 [metatags] per section MAX
- Don't contradict yourself ([Calm] + [Aggressive] in same section)

#### More Options (set when user specifies, otherwise use defaults)
| Option | Range | Default | Notes |
|--------|-------|---------|-------|
| Vocal Gender | M/F/None | Auto | Set from style field |
| Weirdness | 0-100 | 30 | Higher = more experimental |
| Style Influence | 0-100 | 50 | Higher = stricter to prompt |
| Lyrics Mode | Manual/Auto | Manual | Always Manual for quality |
| Exclude Styles | text | — | "autotune, mumble rap" etc |

### Step 3: Automate Suno Web UI

Use Hermes browser tools to drive suno.com/create.
For the exact JS snippets (IIFE batch-fill pattern, selector cheatsheet,
Create-button clicker, link harvester), see:
`references/browser-automation-patterns.md`.

```
1. browser_navigate → https://suno.com/create
2. browser_snapshot → Verify logged in (look for "Create" page, no login prompt)
   - If NOT logged in: tell user to log in manually in the CDP browser once
3. Switch to "Advanced" tab (the skill calls it Custom Mode — the UI tab is
   labeled "Advanced" as of v5.5). Click it from the tablist.
4. Batch-fill ALL THREE fields via a single browser_console IIFE — faster and
   more reliable than browser_type for long prompts. Target selectors:
     • Title: document.querySelector('input[placeholder*="Song Title"]')
     • Lyrics: textarea with placeholder containing "lyrics"
     • Style:  textarea with placeholder starting "classic jazz" (it's the
       recommended-styles hint, not an obvious label — find by iterating
       all textareas and matching placeholder text)
   Use the correct setter per element type:
     • HTMLInputElement.prototype for title
     • HTMLTextAreaElement.prototype for lyrics/style
   Dispatch both 'input' and 'change' events with { bubbles: true }.
5. Click "Create songs" button — locate via textContent match in a
   document.querySelectorAll('button') loop (it's often below the fold,
   so a JS click is more reliable than scrolling + browser_click).
6. Wait for generation (poll every 30-60 seconds via browser_vision):
   - Generation takes 60-120+ seconds on v5.5 (NOT 30-90s as in v4.5)
   - Suno generates 2 variations per click; they finish at different times
   - browser_vision is the best way to see cover art + timestamps;
     the snapshot alone can't tell you reliably when tracks are done.
7. When complete:
   - Collect all links: document.querySelectorAll('a[href*="/song/"]')
     returns pairs of (title, URL) for every generated track in the workspace.
   - Report song title, duration, and Suno link to user.
```

#### Setting Range Inputs (Sliders) via JavaScript
```javascript
// Use browser_console to set slider values programmatically
const slider = document.querySelector('input[type="range"]');
const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
  window.HTMLInputElement.prototype, 'value').set;
nativeInputValueSetter.call(slider, '75');
slider.dispatchEvent(new Event('input', { bubbles: true }));
slider.dispatchEvent(new Event('change', { bubbles: true }));
```

#### Handling Long Lyrics
If lyrics exceed what browser_type can handle in one shot:
```javascript
// Use browser_console to paste lyrics directly
document.querySelector('textarea[aria-label*="yrics"]').value = `LYRICS_HERE`;
document.querySelector('textarea[aria-label*="yrics"]').dispatchEvent(
  new Event('input', { bubbles: true })
);
```

### Step 4: Retrieve and Deliver Results

After generation completes:
1. browser_snapshot → find the generated song cards/players
2. Extract: song title, audio URL, Suno URL, duration
3. If user wants to download: use terminal to `curl -o` the audio URL
4. Report results with direct links

### Step 5: Offer Follow-Up Actions
- "Extend this song" (use song extend feature)
- "Generate another variation"
- "Generate stems" (separate vocals/instruments)
- "Create a sequel" (similar style, new lyrics)

## Workflow: DJ Mode

### Concept
User provides a vibe arc (e.g., "chill morning → energetic workout → cool down")
or a scenario (e.g., "a 30-minute coffee shop set"). Agent generates a sequence
of 4-8 songs with intentional energy progression.

### Energy Arc Planning
```
SONG 1 (Intro):     Energy 2-3, set the mood, ambient/intro feel
SONG 2 (Warm-up):   Energy 4-5, establish groove
SONG 3 (Build):     Energy 6-7, add layers, build momentum
SONG 4 (Peak 1):    Energy 8-9, first climax
SONG 5 (Breather):  Energy 5-6, slight pullback
SONG 6 (Peak 2):    Energy 9-10, main climax
SONG 7 (Wind-down): Energy 4-5, start descending
SONG 8 (Outro):     Energy 2-3, ambient, fade to silence
```

### BPM Key Matching (for transitions)
- Keep adjacent songs within ±10 BPM for smooth transitions
- Use relative keys (e.g., Am → C or Am → Em) for harmonic mixing
- Specify BPM and key in style field for each song

### DJ Mode Execution
1. Plan the full arc first, present to user for approval
2. **Parallel-prep technique (critical efficiency win)**: Do NOT wait for
   track N to finish generating before filling track N+1's fields. The
   pattern is:
     - Click Create on track N
     - IMMEDIATELY fill title/lyrics/style for track N+1 via browser_console
     - Click Create on track N+1 (can be fired while N is still rendering)
     - Sleep/poll — by the time N+1 is done, prep N+2 or collect results
   This cut a 4-track set's wall-clock time roughly in half vs sequential
   wait-generate-wait-generate. Suno doesn't throttle overlapping creates
   (on Pro plan at least); the "rate limit" warning in older docs refers
   to per-day credit caps, not per-minute throughput.
3. Track all audio URLs as they complete (use the /song/ link selector)
4. After all songs generated, present the full setlist with **both** versions
   of each track — duration varies wildly (3-min and 8-min takes from same
   prompt are both possible) so let user pick. Mention that longer versions
   suit DJ mixing, shorter ones suit standalone listening.

## Suno Web UI Reference (as of v4.5/v5)

### Page Layout (suno.com/create)
- **Top**: Navigation bar with Home, Create, Library, profile
- **Main area**: Song description input (simple mode) or Custom mode fields
- **Custom mode toggle**: Usually a switch/toggle near the input area
- **Custom mode fields**:
  - Lyrics textarea (largest field)
  - Style of Music text input
  - Title text input
  - "More Options" collapsible section
  - "Create" button (prominent, usually colored)
- **Results area**: Below or side panel, shows generated songs as cards

### Common UI Identifiers
- Look for elements with aria-labels or role attributes
- Song cards typically have play buttons and download options
- Generated songs show title, duration, and waveform visualization

## Prompt Engineering Patterns (Quick Reference)

### Genre Combos That Work Well
- "lo-fi hip hop jazz" — study beats
- "Appalachian folk gothic" — dark country
- "synthwave darkwave retro" — 80s electronic
- "bossa nova neo-soul" — smooth fusion
- "orchestral dubstep cinematic" — epic drops
- "bluegrass trap" — unexpected but great
- "chamber pop art rock" — sophisticated indie

### Dynamic Arc Starters (put in Style field)
- "Begins whisper-quiet with just piano, gradually builds to full orchestra"
- "Starts aggressive and in-your-face, softens in the bridge, returns harder"
- "Sparse and atmospheric throughout, with sudden explosive chorus"
- "Upbeat from bar one, pure joy, wall of sound production"

### Vocal Persona Templates
- Male: "Weathered baritone with gravel and warmth, speaks-sings verses,
  opens up to full chest voice in chorus"
- Female: "Crystal clear soprano with breathy intimacy in verses,
  powerful belted chorus with controlled vibrato"
- Androgynous: "Ethereal countertenor floating between registers,
  layered with harmonies creating a chorus effect"

## Error Handling

### Not Logged In
If browser_snapshot shows a login page or "Sign In" button:
→ Tell user: "Suno isn't logged in. Open your CDP browser
(brave on port 9222) and log into suno.com manually once.
Your login will be saved permanently."

### CAPTCHA Challenge
Suno sometimes triggers hCaptcha. The browser tools can attempt it,
but CAPTCHAs are hard for AI. If stuck:
→ Suggest user solves it manually in the CDP browser, then retry.

### Rate Limits
Suno free tier: ~10 songs/day. Pro: 50-200/day depending on plan.
If hit rate limit:
→ Report credits remaining, suggest waiting or switching to HeartMuLa
for unlimited local generation as fallback.

### Generation Timeout
If page doesn't show results after 3 minutes of polling:
→ browser_vision to see current state, check for error messages,
retry if needed.

## Common Pitfalls

1. **Forgetting to load songwriting skill first.** The [metatags] and
   prompt formulas are complex — always load it before generating prompts.

2. **Not using Custom Mode.** Default mode gives Suno a single text prompt
   and lets it decide everything. Custom Mode with separate Title/Lyrics/Style
   gives dramatically better results.

3. **Contradictory metatags.** [Calm] and [Aggressive] in the same section
   confuses the model. Pick a direction per section.

4. **Too many metatags.** More than 8 tags per section dilutes their effect.
   Fewer, well-chosen tags work better.

5. **Artist names in Style field.** Suno actively blocks these. Always describe
   the sound: "90s grunge" not "Nirvana style".

6. **Numbers and acronyms in lyrics.** "24/7" becomes gibberish. Always spell
   out: "twenty four seven". "AI" → "A I".

7. **Browser session not persistent.** Without CDP + user-data-dir, the Suno
   login won't survive between sessions. The whole point of the CDP approach
   is persistent auth.

8. **Ignoring the dynamic arc in Style field.** "Sad rock song" is weak.
   "Begins with lone acoustic guitar, layers in drums, builds to full band
   crescendo, strips to whispered vocal over ringing chord" gives Suno a
   performance map.

9. **Not generating multiple takes.** Even with perfect prompts, Suno produces
   2 variations per generation. Present both to the user and let them pick.

10. **Trying to set sliders with browser_type.** Range inputs need JavaScript
    via browser_console, not keyboard text input.

11. **Thinking 'Custom Mode' means a visible Custom toggle.** As of v5.5,
    the Suno create page has three tabs: "Simple | Advanced | Sounds".
    What older docs call "Custom Mode" is the **Advanced tab**. Click
    the Advanced tab to get Title/Lyrics/Style fields. Do not look for
    a separate Custom/Default toggle.

12. **Using browser_type for lyrics/style on a busy page.** The Advanced
    tab has multiple textareas and the snapshot is cluttered. Batch-fill
    them all in a single `browser_console` IIFE using placeholder-based
    matching — much faster and less brittle than three sequential
    browser_type calls, and avoids ref-ID drift after other interactions.

13. **Assuming both generated variations finish at the same time.** They
    often stagger by 30-90 seconds. Poll with browser_vision and look
    for timestamps (complete) vs loading spinners (in progress).

14. **Assuming track durations are predictable.** Same prompt can yield a
    3:03 take and a 7:59 take. Duration is emergent from v5.5 — factor
    this into set planning (offer both options to the listener).

## Verification Checklist

- [ ] Songwriting skill loaded for [metatag] reference
- [ ] User intent parsed into concept, mood, genre, specifics
- [ ] Title is evocative and ≤8 words
- [ ] Style field describes dynamic journey, not just genre list
- [ ] Lyrics include structural [metatags] in every section
- [ ] No artist names in Style field
- [ ] Numbers/acronyms spelled out in lyrics
- [ ] Advanced tab (not a "Custom" toggle) active on suno.com/create
- [ ] All three fields (title/lyrics/style) batch-filled via one IIFE
- [ ] Create button clicked via JS querySelector loop (below the fold)
- [ ] Generation polled with browser_vision every 30-60s (v5.5 = 60-120s)
- [ ] Both variations per track collected with a[href*="/song/"] selector
- [ ] Follow-up options offered (extend, stems, variations, Part 2 set)

## Integration with Other Skills

- **songwriting-and-ai-music**: Core reference for all prompt engineering
- **heartmula**: Local fallback when Suno is down or rate-limited
- **spotify**: Add generated songs to playlists, or find inspiration tracks
- **comfyui**: Generate cover art for Suno tracks