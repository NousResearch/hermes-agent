# Sourcing the Content (how the teardown actually gets its input)

**Read this first — it's the step most "analyze viral content" advice skips.** An AI agent usually
**cannot watch a video from a link.** A pasted TikTok/YouTube/Instagram URL does not let the model
see the frames — the platforms are walled, and even a successful fetch typically returns metadata,
not the video. So this skill never relies on "paste a link and I'll watch it." Instead it analyzes
whatever **observable signal** is brought into context, and is honest about what it can and can't
see.

> The golden rule: **the human (or a transcript / screenshot / tool) is the eyes; the skill is the
> analyst.** Never fabricate frames, lines, or details that weren't provided. If a layer can't be
> observed, say so and analyze the rest.

## Input paths, best to worst

1. **Structured intake (most reliable, any platform).** Ask the user for the observable parts — the
   intake below. This turns the user into the camera and works even when nothing can be fetched.
2. **Transcript / captions.** For talking-head, storytime, and explainer content the transcript
   carries most of the signal. Paste it, or (YouTube only) use a subtitles tool if the agent has
   one.
3. **Screenshots / key frames (multimodal).** The user uploads the cover/thumbnail, the hook frame,
   the on-screen text, and a stats screenshot — a multimodal agent reads images directly. Best path
   for the **visual** layer (hook frame, packaging, text overlays).
4. **Top comments.** Often the clearest read on *why* it resonated and why people shared. Paste or
   screenshot the top few — they frequently name the share-trigger out loud.
5. **Tools / browser, where available.** A fetch/browser tool can sometimes pull a caption,
   description, or visible counts (works best for YouTube; TikTok/IG are largely walled). Use it
   when present; fall back to the intake when not.

## The structured intake (what to ask for)

When the user brings a link or a vague "analyze this," request the observable inputs:

- **The hook** — the exact first line and/or what's on screen in the first 1–3 seconds.
- **What happens** — a quick play-by-play, or the transcript.
- **Caption + on-screen text** — paste verbatim if possible.
- **Format** — talking head, storytime, carousel, etc.; length.
- **Visible stats** — views, likes, shares, comments, saves (whatever's shown).
- **Creator context** — rough follower count (critical for the replicability check — a huge account
  confounds everything).
- **Sound** — trending audio or original?
- **Top comments** — the few most-liked, if available.

You don't need all of it — analyze what's provided and **name the gaps** ("I can't see the
pacing/edit, so I'm reasoning from the transcript + stats").

## Match the input to the layer

Different inputs unlock different teardown layers (see `deconstruction-framework.md`):

| To analyze… | You need… |
|---|---|
| Hook, structure, topic, angle | transcript or the user's play-by-play |
| Visual hook, packaging, on-screen text | screenshots / key frames |
| Share-trigger (why people shared) | top comments + the content itself |
| Distribution confounds | follower count, sound, post date/stats |
| Emotional driver | transcript + comments (how people reacted) |

## Graceful degradation

- **Only a link, nothing else** → explain you can't watch it; ask for the intake (or a transcript /
  screenshots), or use a tool if available. Don't pretend to have seen it.
- **Transcript only** → analyze hook/structure/angle/share-trigger; flag that you can't assess the
  visual/edit layer.
- **Screenshots only** → analyze the visual hook/packaging/on-screen text; flag the missing
  spoken/structure layer.
- **Stats missing** → you can still teardown the mechanism, but say you can't confirm it was actually
  "viral" without numbers (see the no-analytics note in `replicability-and-application.md`).

The teardown is only as good as the input — so getting the right input in is the real first step,
and being honest about what's missing is part of the quality bar.
