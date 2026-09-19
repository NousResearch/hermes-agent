# Platform Publishing Constraints

What each platform requires *at publish time* — media, formats, and rules that cause a schedule
to fail or a post to look wrong. WoopSocial handles per-platform field formatting and MIME
automatically, but the **content and media still have to be valid for the platform**, so check
these before scheduling and lean on WoopSocial's validate step.

> Specs change. Treat these as the durable shape and re-verify exact numbers via WoopSocial's
> validate response and the platforms' current docs.

## Across all platforms

- **AI-generated media must be disclosed.** WoopSocial adds the disclosure automatically on
  **TikTok**; for others, follow the brand's compliance guardrails and platform rules. (EU AI
  Act transparency obligations apply broadly — keep disclosure on.)
- **Media:** uploaded as raw bytes; WoopSocial detects MIME. Still ensure resolution/aspect/
  length fit the target platform.
- **Validate first** — let WoopSocial catch unsupported media, length, or missing fields before
  you commit.

## Instagram

- Formats: single image, carousel (multiple images/videos), Reels (video).
- Aspect ratios: 1:1, 4:5 (portrait feed), 9:16 (Reels). Avoid odd crops.
- Caption ~2,200 chars; hashtags categorize, not boost — keep to a handful.
- **No clickable links in captions** → use "link in bio." Don't schedule a clickable-link CTA.
- Publishing requires a Business/Creator account connection.

## TikTok

- Video, 9:16 vertical.
- **AI-disclosure required — handled automatically by WoopSocial.**
- Keywords in the caption aid search; the video carries the message.
- Some native features (trending sounds) aren't settable via API — don't promise them.

## LinkedIn

- Image, document (carousel-style PDF), or video; or text-only.
- ~3,000 chars; format with whitespace for dwell.
- External links in-body can suppress reach — consider link-in-first-comment (a common tactic).
- Posting target can be a person or a company page — confirm which account.

## X / Twitter

- 280 chars free (longer with Premium); media supported.
- **Monthly post cap on lower WoopSocial tiers (e.g., ~200/mo)** — count batch posts against it
  and warn before exceeding.
- Keep hashtags minimal (0–2).

## Facebook

- Page posting; image/video/link; long text allowed.
- Links are clickable. Hashtags largely ineffective.

## Pinterest

- A Pin needs an **image (or video)**, a **destination link**, and a **board**.
- Keyword-rich title/description drive discovery (SEO), not hashtags.
- Strong genuine traffic driver — the link matters.

## YouTube

- Video upload (Shorts = vertical/square up to ~3 min; long-form = 16:9), with title + description.
- Description is searchable — front-load keywords and any link.
- Community posts are a separate, lighter format.

## Threads

- Text (~500 chars) with optional media; **single post only** — no chains, polls, voice, or
  GIFs via the API. A "thread" chain is manual.
- **One topic tag**, not hashtags — Threads limits to one.
- Links are clickable but conversation-first content wins; see `threads-post`.

## Bluesky

- Text (~300 chars) with optional media; links clickable.
- Casual, community-forward register — lead with the point (see `text-post-and-microblog`).
- Newer publish target — run WoopSocial's validate step and re-verify limits before batches.

## Pre-schedule checklist

- [ ] Media meets the target platform's format/aspect/length.
- [ ] Caption within the platform's limit; links handled correctly (bio vs clickable).
- [ ] Required account type connected (e.g., IG Business).
- [ ] AI media disclosure on where required.
- [ ] Batch fits plan limits (esp. X monthly cap).
- [ ] WoopSocial validate passed for every post.
