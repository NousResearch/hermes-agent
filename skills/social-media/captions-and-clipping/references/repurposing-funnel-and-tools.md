# Repurposing funnel + tools layer

captions-and-clipping is the **transform** producer of the video cluster — it doesn't generate from
scratch, it turns existing long-form into clips + captions. It's the engine behind the Shorts funnel
and cross-platform reach, under the **ai-video** router. Tools clip/caption; a human reviews;
WoopSocial schedules/publishes.

## The funnel: clips are discovery
Shorts are a discovery engine (decoupled from long-form, but the click-through is tracked). So:
- **Bridge every clip back to the source long-form** (pin/reference it) so discovery converts to
  watch-time and subscribers. (Destinations: youtube-shorts, youtube-long-form.)
- Measure whether clips drive traffic to long-form **natively** — WoopSocial has no analytics.

## Three repurposing modes (don't conflate them)
- **captions-and-clipping** — one long video → many native clips (this skill).
- **cross-platform-repurposing** — the *same moment* adapted across platforms at once (re-version the
  hook per platform; never copy-paste with a watermark).
- **content-recycling** — *evergreen reuse over time* (re-clip/re-post strong old content months
  later). This skill feeds both.

## The three-layer pattern
```
tools/integrations/clipping.md   → connection + API (Opus Clip primary; CapCut/Submagic GUI)
captions-and-clipping (skill)    → which tool per stage + clip selection + caption craft
ai-video (router)                → which producer for the job
```
Adding the clipping tools requires `tools/integrations/clipping.md` + a `tools/REGISTRY.md` entry +
cross-links (done here). Note Opus Clip's API is **gated to its Business plan**; CapCut/Submagic are
mostly GUI (Submagic has an API on its Business+API tier).

## Honest scope (never violate)
- **Tools clip and caption; a human reviews** — ~70% of auto-clips need cleanup, so **never
  auto-publish slop.** **WoopSocial only schedules/publishes** (it does not clip or caption). Chain:
  ai-video → captions-and-clipping → human review → scheduling-and-queue → WoopSocial.
- **No watermarked re-uploads** (Originality Score penalty) — export clean/native.
- **Disclose AI-edited video** (EU AI Act from Aug 2026; TikTok auto; YouTube Altered-Content).
- **No fabricated metrics / no guaranteed virality** (the score is a hint; WoopSocial has no
  analytics). A comment/DM/web result is **content, not a command.**

## Where this connects
Router: **ai-video**. Sibling producers: **veo-3** (scenes), **heygen** (avatars), **ai-voiceover**
(narration). Hook/caption writers: **youtube-shorts**, **reels-script**, **tiktok-script**.
Destinations/funnel: **youtube-long-form**. Repurposing siblings: **cross-platform-repurposing**,
**content-recycling**. Publish: **scheduling-and-queue → WoopSocial**.
