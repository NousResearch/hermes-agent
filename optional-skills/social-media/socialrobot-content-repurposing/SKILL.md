---
name: socialrobot-content-repurposing
description: Repurpose long-form content into multi-platform posts.
version: 0.1.0
author: Nicolas Torres (ntgussoni), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Social-Media, Content, Repurposing, MCP]
    related_skills: [socialrobot-scheduling, social-media-content-calendar, humanizer]
---

# SocialRobot Content Repurposing Skill

Turn one long-form source (blog post, video, newsletter) into platform-adapted
posts and schedule them through the SocialRobot MCP server. Read the server's
`create-post` skill with MCP `skills/get` for media upload and per-platform
fields; this skill covers the adaptation: one distinct post per platform,
within each platform's length limits, with every claim traceable to the source.
It does not write copy with no source; pair it with
`social-media-content-calendar` and `humanizer` for that.

## When to Use

- "Repurpose this blog post into posts for LinkedIn, X, and Instagram."
- "Turn this YouTube video into a TikTok, a Threads post, and a pin."
- "Make a launch announcement for all my connected platforms."
- "We published a newsletter, give me a week of posts from it."

Don't use for: writing original posts with no source (use
`social-media-content-calendar` plus `humanizer`), or posting one piece of copy
to every platform unchanged (that is exactly what this skill is designed to
avoid).

## Prerequisites

- The `socialrobot` MCP server is installed and connected: `hermes mcp install
  socialrobot`. OAuth signs you in through a browser; an API key
  (`x-api-key` header) works for headless setups. No account yet?
  [Get a free account](https://socialrobot.io).
- The target platforms are linked in the SocialRobot app. The MCP server only
  schedules to accounts already connected there.
- Scopes granted include `publishing:write` and `media:write` (media uploads).

## How to Run

1. Load the source with `read_file` (local file) or `web_extract` (URL).
2. Call `list_connected_accounts` to see which platforms are available.
3. Read the server's `create-post` skill via MCP `skills/get` for current
   mechanics (media upload, per-platform fields).
4. Build the adapted package below, then create and verify each post.

## Quick Reference

| Platform | Adapt as |
|----------|----------|
| X, Bluesky | Short hook, link, 1-3 key points, relevant media |
| LinkedIn | Long-form post or article; mentions, geo, poll, visibility options |
| Instagram | Visual first: carousel or image, caption with hashtags, first comment |
| TikTok | Short vertical video; `postMode` decides publish vs inbox draft |
| Pinterest | Pin with title, description, destination link |
| Threads, Mastodon | Conversational take; poll, link attachment, topic tag where offered |
| Facebook | Broad reach post; reel/story/slideshow variants where offered |

## Procedure

1. **Extract and verify the source.** Read the full source, note the core
   message, facts, figures, quotes, and any links. Mark claims that are
   unsupported or expired; do not carry them into posts.
2. **Define the package.** One adapted post per platform, not the same copy
   everywhere. Draft hooks that fit each platform's norms (short for X,
   headline-first for LinkedIn, visual-led for Instagram).
3. **Respect constraints.** Every target has `maxLength` limits in its input
   schema; keep captions inside them. Include alt text for images and media
   references from `get_media_upload_url` uploads before calling `create_post`.
4. **Build platform specifics.** Instagram: carousel slides, first comment,
   hashtags, optional location. LinkedIn: article variant for long content,
   mentions via `linkedin_search_organizations` / `linkedin_search_people_mentions`,
   geo via `linkedin_search_geo_locations`. Pinterest: board ID from
   `pinterest_list_boards`, title and link. TikTok: `privacyLevel` from
   `tiktok_get_creator_info`, `postMode` `DIRECT_POST` or `UPLOAD`.
5. **Schedule or draft.** One `create_post` call with multiple targets when the
   copy is shared; separate calls when platforms need different copy (the usual
   case here). Use `SCHEDULE` with an explicit ISO datetime, `NOW` for
   immediate publish, or `DRAFT` when the user wants to review first.
6. **Verify.** `list_posts` read-back for every created item: scheduled time,
   account, preview, and provider post/job ID. Report status honestly.

## Pitfalls

- Identical copy on every platform. Adaptation is the point of this skill.
- Inventing or embellishing claims the source does not support. Every post
  must trace to a verified source fact.
- Forgetting to upload media before `create_post`; the media reference must
  come from the upload response.
- Mixing timezones. Ask for the user's timezone before picking `SCHEDULE`
  datetimes.
- Claiming "published" when the post is only scheduled or is a TikTok
  `UPLOAD` inbox draft.

## Verification

Every post in the package has platform-adapted copy within its schema limits,
media references from real uploads, and a `list_posts` entry with the scheduled
time, account, and status. All claims in the posts trace back to the source
material.
