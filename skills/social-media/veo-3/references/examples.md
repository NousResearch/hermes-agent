# Examples — prompts in practice

Worked Veo prompts showing weak→strong, audio, image-to-video, and a vertical hook — plus honest
scope. Brand context: a SaaS social scheduler (clean, calm, terminal-green accent). Your output pulls
the real style from `brand-profile.md`.

---

## Weak → strong

**Weak:** "a person using a laptop, cinematic, 4k, amazing quality"
**Strong:** "Medium shot of a calm solo founder closing a laptop on a sunlit desk, then leaning back
with a small relieved smile. Camera slowly pushes in. Warm late-afternoon light, soft shadows. Clean,
minimal, documentary style. Audio: a quiet room tone, a single soft notification chime as the laptop
closes. 9:16 vertical, subject centered. 8 seconds."
*(subject, action, camera move, lighting, style, explicit audio, vertical framing, length — no quality
spam)*

## Audio-rich (the superpower)

"Close-up, over-the-shoulder of hands batching social posts on a laptop, a calendar filling with
scheduled slots. The founder says, to camera, "I do a week in twenty minutes — once.", in a dry,
matter-of-fact tone. Audio: soft keyboard clicks, gentle ambient room tone, no music. Shallow depth of
field, clean daylight. 9:16, 8 seconds."
*(dialogue in quotes + tone + SFX + ambient; treat the VO as a guide track for branded final; verify
lip-sync)*

## Image-to-video (Nano Banana → Veo)

**Input:** a `nano-banana` still — a clean flat-illustration title card, "Post less. Reach more.",
brand palette.
**Veo prompt:** "Bring this title card to life: the text settles in with a subtle scale, a soft
terminal-green underline draws left-to-right beneath the last line, gentle floating particles. Calm,
minimal motion. Audio: a soft whoosh as the underline draws, light ambient pad. 9:16, 6 seconds."
*(image gives exact text/brand control; Veo adds motion + sound)*

## Vertical hook clip (for a Reel)

"Fast 1.5-second opener: a phone on a desk lighting up as seven posts publish in a rapid burst, screen
glow flicking across the desk. Quick snap zoom. Audio: a satisfying rapid series of soft pops. Crisp,
high-contrast, 9:16." (→ `reels-script` writes the rest of the script around it.)

---

## Honest scope (say this)

- **8-second clips** — plan short; stitch/scene-extend for longer; hook in the first second.
- **Async + costly** — iterate at low-res/Fast, finalize the keeper at 4K; off-prompt gens still cost.
- **Audio = draft** — for branded work, record real VO / license music for the final; verify lip-sync.
- **SynthID + disclosure** — every clip is watermarked; disclose AI video (EU AI Act, TikTok
  auto-disclosure); never pass it off as real footage.
- **No real people / no IP** — no real identifiable individuals, copyrighted characters, or brand
  styles you don't own.
- **Verify** every clip for artifacts before publishing.
- **Prompt craft only** — API/model IDs/async flow + the *generate → upload to WoopSocial Media →
  attach* path live in `tools/integrations/veo.md`; WoopSocial doesn't generate video.

---

## What the examples share

- **Direct the shot** (subject/action/camera/lighting/style) in natural language — no quality spam.
- **Describe the audio explicitly** (dialogue in quotes, SFX, ambient) — it's the differentiator.
- **Vertical framing + length** set up front; **brand-grounded.**
- **Verified, disclosed, and IP-safe** before publishing.
