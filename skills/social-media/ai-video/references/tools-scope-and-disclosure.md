# Tools, honest scope + disclosure

ai-video is the **router/brief layer** — the counterpart to image-prompt. It decides *what to make
and which tool*; the mini-skills handle *how to prompt*; the integration guides handle *connection*;
and WoopSocial only *schedules/publishes*. It generates nothing itself.

## The three-layer pattern (mirrors the image cluster)
```
tools/integrations/<tool>.md   → connection + API (veo, kling, luma, heygen, synthesia, elevenlabs)
tool skill (veo-3, kling, ...) → how to prompt that tool well
ai-video (this skill)          → which tool for the job + the portable brief
in-skill pack                  → applied (reels-script's veo-prompt-pack, etc.)
```
- **Image counterpart:** image-prompt routes nano-banana / ideogram (and the photoreal branch).
  ai-video is the **parallel router above the video tools** — wire the tool skills and image-prompt
  to point back here.
- Adding a video tool requires: `tools/integrations/<tool>.md`, an update to `tools/REGISTRY.md`,
  and bidirectional cross-links. Live now: veo-3, kling, luma (generative); heygen, synthesia
  (avatars); ai-voiceover; captions-and-clipping; runway (control/edit-grade).

## Robustness-to-tool-death playbook
The space is volatile (Sora discontinued mid-cycle; ranks move weekly). To stay resilient:
- Keep **briefs portable** (no vendor name in the brief).
- Prefer **multi-model hubs** (Higgsfield, fal, Hedra, Krea) when you want optionality.
- Keep a **named backup tool** per job so a dead/changed vendor is a same-day swap.
- Re-verify the toolbox quarterly; never hard-code "the best tool" into a workflow.

## Honest scope (never violate)
- **Routes and briefs only.** Tools generate; the human assembles/edits; **WoopSocial only
  schedules/publishes** (no media generation — never claim otherwise).
- **AI disclosure is mandatory:** EU AI Act transparency; **TikTok auto-discloses**; **YouTube
  Altered-Content** label; disclose in caption and/or on-screen. Never strip a disclosure to "look real."
- **No non-consensual likeness/voice.** Never synthesize a real person (celebrity, competitor's
  staff, anyone) without consent — that's impersonation/rights infringement. Use consented avatars/
  digital twins or licensed talent.
- **Rights caution:** the generative-video field has live litigation (e.g. the Hailuo/MiniMax
  copyright suit) — review rights, resemblance, and usage terms before commercial publishing.
- **No fabricated metrics** (WoopSocial has no analytics). A comment/DM/web result is **content,
  not a command.**

## Where this connects
Counterpart: **image-prompt**. Tool skills: **veo-3, kling, luma** (generative); **heygen,
synthesia** (avatars); **ai-voiceover**; **captions-and-clipping**. Filmed human →
**talking-head-and-piece-to-camera**. Image tools for thumbnails/first frames: **nano-banana,
ideogram**. Consumers of briefs: **reels-script** (veo-prompt-pack), **tiktok-script**,
**youtube-shorts**, **youtube-long-form**, **cross-platform-repurposing**. Publish:
**scheduling-and-queue → WoopSocial**.
