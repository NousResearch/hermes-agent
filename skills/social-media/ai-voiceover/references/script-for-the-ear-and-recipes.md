# Scripting for the ear + two worked VO scripts

## Write for the ear (the 80%)
- Contractions, short sentences, spoken rhythm. "You'll" not "you will."
- One idea per breath; vary sentence length for a natural cadence.
- Spell tricky names/jargon phonetically in a pronunciation note.
- **Read it aloud first.** If you stumble, rewrite the line.
- Don't paste eye-copy (blog paragraphs, bullet lists) — rewrite it as speech.

## Direct the delivery
- **Model:** Eleven v3 (expressive + Audio Tags) for performed VO; Multilingual v2 for long-form
  finals; Flash for drafts.
- **Audio Tags (v3):** sparing emotion/pacing cues, e.g. `[warm]`, `[whispers]`, a beat `...`.
- **Settings:** Stability ~0.3–0.5 (expressive) vs ~0.7–1.0 (consistent); Similarity ~0.75–0.85.
- Generate 2–3 takes; pick the most natural; regenerate weak lines rather than the whole script.

## Model-by-job, fast
| Job | Model | Why |
|---|---|---|
| Reel/Short VO (expressive) | Eleven v3 | Audio Tags, energy |
| Explainer / course / audiobook | Multilingual v2 | stable over long-form |
| Draft / internal / real-time | Flash v2.5 | ~half credits, ~75ms |
| Localized versions | Dubbing | preserves voice, 70+ langs |

## Worked example 1 — Reel VO (blunt indie-founder voice, ~18s, Eleven v3)
```
VOICE: designed "dry, low-key founder"   MODEL: eleven_v3   STABILITY: 0.4
SCRIPT (for the ear, tags sparing):
"Almost missed four grand in tax this year. [beat] Not proud of it.
I was tracking income. I wasn't tracking the set-aside.
So now? Thirty percent of every invoice, separate account, same day it lands.
[warm] Boring. Works. Saved my April."
CAPTIONS: burned in (mute viewing). DISCLOSURE: "AI voiceover" in caption.
```

## Worked example 2 — explainer narration (warm studio voice, ~30s, Multilingual v2)
```
VOICE: consented founder clone (PVC, own voice)   MODEL: eleven_multilingual_v2   STABILITY: 0.6
SCRIPT:
"If tax season makes your stomach drop -- this'll help.
We do one thing, and the whole panic just... goes.
Open a second account. Call it Tax.
Then move thirty percent the moment a client pays you.
You never see it, so you never spend it -- and April stops hurting."
LOCALIZE: Dubbing into ES/FR/DE, founder voice preserved; review each.
CAPTIONS: burned in. DISCLOSURE: "AI voiceover (our founder's voice)".
```

Both: read brand-profile + voice-builder first; written for the ear; directed (model/tags/settings);
captioned for mute viewing; consent + disclosure handled. ElevenLabs generates the audio; a human
mixes it into the video (it has no built-in editor). B-roll → veo-3 / ai-video; avatars → heygen;
captions/clipping → captions-and-clipping; publish → scheduling-and-queue → WoopSocial.
