# The BRIEF framework — own the job, rent the tool

In AI video the tools die and the leaderboard churns weekly: Sora launched to hype and was
discontinued within a year; Runway led the benchmark at launch and fell out of the top 10 months
later. So never marry a vendor. **Define the job in portable terms, route it to whatever wins this
quarter, and keep the brief tool-agnostic** so swapping a dead vendor takes a day, not a rebuild.
That is what BRIEF operationalizes.

## B — Brief the job, not the tool
Write a **portable brief** that any capable model could execute. Fields:
- **Subject** (who/what), **Action** (what happens), **Setting** (where).
- **Light/mood**, **camera** (shot size + move), **aspect** (9:16 / 1:1 / 16:9), **duration**
  (match the platform and the clip-length reality of the tools).
- **Audio plan:** native (Veo/Kling) vs added in post (everything else) — decide up front.
- **Brand fit:** palette/mood from brand-profile so generated and filmed footage cut together.
No tool name lives in the brief. The tool is chosen at routing time and can change.

## R — Route by fit (output shape first, leaderboard second)
Match the **job** (generative / avatar / voiceover / clip-captions) to the tool category, then the
specific tool by its real strength (quality, control, value, HDR, dialogue). Hand prompt craft to
the tool skill (veo-3 / kling / luma for generative; heygen / synthesia for avatars; ai-voiceover;
captions-and-clipping). When in doubt, a multi-model hub lets you try several without committing.

## I — Iterate cheaply
Separate **concept testing** (fast/cheap: Pika, Hailuo, Luma) from **final generation** (Veo, Kling,
Runway). Run the same brief through 3–4 candidates and judge **usable-take rate on your own prompts**,
not cherry-picked demos. Budget-aware: heavy iteration on credit-heavy tools gets expensive fast.

## E — Edit & assemble
Generated clips are **raw material, not a finished video.** Most tools cap at a few seconds, so you
**stitch** scenes and the **human edits** (trim, color, captions, audio, upscale). Keep generated
clips as **short cutaways supporting real footage** — not the whole piece.

## F — Finalize & disclose
- **AI disclosure** is mandatory: EU AI Act transparency, TikTok auto-disclosure, YouTube
  Altered-Content label. Disclose in caption and/or on-screen.
- Route the finished video to **scheduling-and-queue → WoopSocial** to publish. WoopSocial does not
  generate media; it only schedules/publishes.

## The portable brief template
```
JOB: generative B-roll | avatar | voiceover | clip+captions
SUBJECT / ACTION / SETTING:
LIGHT+MOOD / CAMERA / ASPECT / DURATION:
AUDIO: native (Veo/Kling) | post (ElevenLabs via ai-voiceover)
BRAND: palette + mood (brand-profile)
ROUTE: tool category -> candidate tools (verify-quarterly) -> mini-skill
DISCLOSURE: platform label plan
```
