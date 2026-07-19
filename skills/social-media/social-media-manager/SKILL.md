---
name: social-media-manager
description: "Plan, create, review, publish, verify, and roll back cross-platform social-media campaigns. Use for content calendars, posts, threads, captions, images, videos, accessibility text, scheduling, publishing, engagement review, or coordinated account operations."
license: MIT
metadata:
  hermes:
    tags: [social-media, publishing, content, image, video, accessibility]
---

# Social Media Manager

Run social work as an auditable content pipeline. Keep research and drafting autonomous; keep account writes explicit, scoped, and verifiable.

## Workflow

1. Build the brief.
   - Record the goal, audience, platforms, accounts, locale, voice, source material, deadline, and success measure.
   - Separate known facts from assumptions. Verify drift-prone platform limits or policies against current official documentation before relying on them.
2. Discover the available route.
   - Prefer an official connector, API, or CLI such as `xurl` over browser automation.
   - Use the in-app browser only when the platform lacks a suitable tool or when a visible review/login step is required.
   - Check authentication with a status command. Never inspect authentication material or ask the user to paste it into chat.
3. Draft a campaign manifest. For every item, track:
   - `platform`, `account`, `format`, `copy`, `alt_text`, `media`, `publish_at`, `approval`, `status`, and the final platform ID/URL.
   - Use these states precisely: `draft`, `reviewed`, `approved`, `submitted`, `published`, `verified`, `failed`, `rolled_back`.
4. Create media.
   - Route still images through `image_generate` and video through `video_generate` when those tools are available.
   - Preserve identity and consent. Reject deceptive impersonation, non-consensual face/voice use, or fabricated endorsements.
   - Verify aspect ratio, duration, text safe areas, captions, spelling, audio, and alt text before approval.
5. Review the complete batch.
   - Show the exact copy, target account/platform, media, timing, and irreversible effects.
   - Treat a request to publish already-specified content to specified targets as approval for that exact batch. Otherwise obtain explicit approval before posting, replying, messaging, following, unfollowing, deleting, spending, or changing account settings.
6. Publish with bounded writes.
   - Execute only the approved batch. Respect rate limits and platform terms.
   - Stop for login, MFA, CAPTCHA, legal/rights uncertainty, unexpected cost, or a target mismatch.
7. Verify and reconcile.
   - Capture returned IDs/URLs and re-read the published item when possible.
   - Distinguish API acceptance from visible publication. Never report `published` until the platform confirms it; use `verified` only after a read-back.
   - On partial failure, report the exact completed and failed items. Retry only idempotent operations.
8. Roll back when requested and supported.
   - Confirm the exact item before deletion. Record whether deletion was accepted and whether the item is no longer visible.

## Content quality gates

- Match the platform without inventing algorithm claims or guaranteed reach.
- Preserve factual attribution and source links when claims need evidence.
- Add useful alt text for images and captions/transcripts for spoken video.
- Keep hashtags, mentions, and calls to action relevant; avoid spam patterns.
- Check copy and media together so the caption does not contradict the asset.
- Use platform-native previews or a dry run when available.

## Safety and integrity

- Never expose authentication material, recovery codes, private messages, or raw browser storage.
- Never bypass CAPTCHA, anti-bot controls, rate limits, moderation, or account restrictions.
- Never automate fake engagement, purchased followers, coordinated inauthentic behavior, harassment, or unsolicited mass messaging.
- Never scrape or enrich personal data for targeting. Use public business/contact data only when lawful and requested.
- Never create or fund ads, subscriptions, paid boosts, or other cost-bearing actions without explicit approval.
- Do not infer rights to music, images, logos, likenesses, or third-party clips; stop when ownership or permission is unclear.

## Delivery

Return a compact ledger: target, final state, platform ID/URL, verification result, and any remaining human gate. Avoid claiming reach, delivery, or engagement that has not been measured.
