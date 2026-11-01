# WoopSocial publish spec — per-platform matrices

*Ground truth = the WoopSocial OpenAPI contract (`POST /v1/posts`). Re-verify against the live spec /
`tools/integrations/woopsocial.md` quarterly; enums/limits change.*

## Platforms (the `SocialPlatform` enum)
`FACEBOOK · INSTAGRAM · LINKEDIN · LINKEDIN_PAGES · PINTEREST · THREADS · TIKTOK · X · YOUTUBE · BLUESKY`
(+ `WOOPTEST` sandbox). A post has **one content item** (text + optional media — the array is `maxItems: 1`,
so **no native multi-post thread/chain**) and **1+ social-account targets**, all in the **same project**.

## Required per-platform fields (the `*Fields` schemas)
| Platform | Required | Optional / notable |
|---|---|---|
| **Facebook** | `postType` (TEXT_ONLY/IMAGE/VIDEO/REEL/STORY) | `link`, `locationId` |
| **Instagram** | `postType` (POST/REEL/STORY) | `cover` (REEL only) |
| **LinkedIn** / **LinkedIn Pages** | — (text + optional media) | `link` |
| **Pinterest** | `pinterestBoardId` | `title` (≤100), `link` (≤2048); **single image OR video** |
| **Threads** | — (text + optional media) | — (no postType/link/poll fields) |
| **TikTok** | `postType` (VIDEO/PHOTO), `privacyLevel`, `allowComment`, `allowDuet`, `allowStitch`, `isYourBrand`, `isBrandedContent`, `autoAddMusic` | `cover` (VIDEO), `postMode` (DIRECT_POST/MEDIA_UPLOAD) |
| **X** | — (text + optional media) | credit-metered after free allotment |
| **YouTube** | `title`, `privacy` (public/private/unlisted) | `category`, `tags[]`, `madeForKids` |
| **Bluesky** | — (text + optional media) | ~300-char limit; verify exact fields against the live spec |

## Media rules (the gotchas validation catches)
- **Facebook:** cannot mix images + videos in one post.
- **Instagram:** Stories require media (no text-only Story); needs a Business/Creator account.
- **Pinterest:** **single image OR single video** per Pin (no multi-image carousel).
- **TikTok:** `cover` applies to **VIDEO** only; `autoAddMusic` applies to **PHOTO** only;
  `allowDuet`/`allowStitch` apply to **VIDEO** (required by contract but ignored for PHOTO).
- **Media** = raw-bytes upload, server-side MIME detection (`tools/integrations/woopsocial.md`). `MediaType`
  is `IMAGE` or `VIDEO`.

## Field-applicability quirks (set-but-ignored)
TikTok PHOTO ignores `cover`/`allowDuet`/`allowStitch`; TikTok VIDEO ignores `autoAddMusic` — all still
**required by the contract**, so set them, but know which the platform actually uses. Instagram `cover` is
REEL-only.

## Platform-discovery endpoint
`GET /social-accounts/{id}/platform-inputs` returns per-account valid values:
- **Pinterest:** the available **boards** (use a board `id` as `pinterestBoardId`).
- **TikTok:** `privacyLevelOptions`, `commentAvailable`, `duetAvailable`, `stitchAvailable`,
  `maxVideoPostDurationSec`.
**Always resolve `pinterestBoardId` and TikTok `privacyLevel` from this endpoint** — don't guess.

## Lifecycle facts that constrain a post
- **Atomic create:** `POST /posts` validates all targets together; **if any target fails, nothing is
  created.**
- **No update endpoint:** edit = **delete + recreate** (delete only while every delivery is `NOT_STARTED`).
- **Delivery status:** `NOT_STARTED → SENDING → PUBLISHED | FAILED` (read via webhooks / list).
- **Schedule:** `DRAFT` / `PUBLISH_NOW` / `SCHEDULE_FOR_LATER` (`scheduledFor`, UTC).
