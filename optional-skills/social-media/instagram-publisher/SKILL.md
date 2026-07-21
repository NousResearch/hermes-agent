---
name: instagram-publisher
description: "Publish Instagram media through MyBrandMetrics."
version: 1.0.0
author: Clawbus; Hermes Agent
license: MIT
platforms: [linux, macos, windows]
required_environment_variables:
  - name: INSTAGRAM_API_KEY
    description: MyBrandMetrics API key for Instagram publishing
  - name: INSTAGRAM_CONNECTION_ID
    description: Instagram connection ID in MyBrandMetrics
  - name: INSTAGRAM_ACCOUNT_ID
    description: MyBrandMetrics Instagram account ID
metadata:
  hermes:
    tags: [Instagram, Social Media, Publishing, MyBrandMetrics]
    homepage: https://www.clawbus.com/skills/instagram-publisher
---

# Instagram Publisher Skill

Use this skill to publish Instagram images, reels, and carousels through a
preconfigured MyBrandMetrics API connection. It supports hosted media URLs,
local media files, mixed carousel items, and status checks for existing publish
jobs.

## When to Use

- The user asks to publish, upload, or post media to Instagram.
- The user wants to publish an image, reel, or carousel.
- The user provides hosted media URLs or local media file paths.
- The user asks to check the status of an existing Instagram publish job.

## Prerequisites

- A direct public media URL or a readable local media file path.
- `INSTAGRAM_API_KEY` or `INSTAGRAM_AUTHORIZATION_TOKEN`.
- `INSTAGRAM_CONNECTION_ID`.
- `INSTAGRAM_ACCOUNT_ID`.
- Network access to `https://api.mybrandmetrics.com`.

The script also supports command-line credentials and a config file; see
`references/configuration.md`.

## References

- `references/configuration.md` — command-line, environment, and config file
  credential sources.
- `references/publishing-examples.md` — concrete examples for images, reels,
  carousels, and status checks.

## How to Run

Use the `terminal` tool with the publishing script:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/social-media/instagram-publisher/scripts/publish_instagram.py" \
  --type IMAGE \
  --url "https://example.com/image.jpg" \
  --caption "Hello World!"
```

Check an existing publish job:

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/skills/social-media/instagram-publisher/scripts/publish_instagram.py" \
  --check-id "v_pub_file~123"
```

## Quick Reference

Publish an image URL:

```bash
python3 "$HERMES_HOME/skills/social-media/instagram-publisher/scripts/publish_instagram.py" \
  --type IMAGE \
  --url "https://example.com/image.jpg" \
  --caption "Hello World!"
```

Publish a local reel:

```bash
python3 "$HERMES_HOME/skills/social-media/instagram-publisher/scripts/publish_instagram.py" \
  --type REELS \
  --path "/path/to/video.mp4" \
  --caption "Check this out!" \
  --thumb-offset 1000
```

Publish a mixed carousel:

```bash
python3 "$HERMES_HOME/skills/social-media/instagram-publisher/scripts/publish_instagram.py" \
  --type CAROUSEL \
  --items "/path/to/img1.jpg" "https://example.com/vid2.mp4" \
  --caption "My Carousel"
```

Check status:

```bash
python3 "$HERMES_HOME/skills/social-media/instagram-publisher/scripts/publish_instagram.py" \
  --check-id "v_pub_file~123"
```

## Procedure

1. Determine whether the user wants a new `IMAGE`, `REELS`, or `CAROUSEL` post,
   or a status check.
2. For single media posts, confirm exactly one media source: `--url` or `--path`.
3. For carousels, confirm every item in `--items` and count them back to the
   user before publishing.
4. For local paths, verify files exist before calling the API.
5. Load credentials from command-line arguments, environment variables, or the
   config file described in `references/configuration.md`.
6. Run `scripts/publish_instagram.py`.
7. Return the real JSON response from the API. Do not fabricate publish IDs,
   container IDs, URLs, or final statuses.

For public-facing posts, confirm the caption and media list with the user before
publishing.

## Pitfalls

- Never commit or echo real Instagram/MyBrandMetrics credentials.
- Carousels require uploading each item first; the script handles child media
  creation and then publishes the final carousel.
- Mixed carousels may take longer because video children need extra processing.
- Hosted URLs must be directly reachable by the API.
- Non-200 API responses are printed to stderr and the script exits non-zero.

## Verification

```bash
python3 "$HERMES_HOME/skills/social-media/instagram-publisher/scripts/publish_instagram.py" \
  --check-id "v_pub_file~123"
```
