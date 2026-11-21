# Validation reference + two worked examples

## The validate-then-create loop
```
1. Build the post (content + schedule + socialAccounts[] with each target's fields)
2. POST /posts/validate  -> { isValid, errors[], warnings[] }   # same rules as create, nothing published
3. Fix every error; review warnings (non-blocking but usually worth fixing)
4. POST /posts  -> creates atomically (any failing target => nothing created => 422)
5. No update endpoint: to change, DELETE (while NOT_STARTED) then recreate
```

## `ValidationField` codes → what to fix
| Code | Means | Fix |
|---|---|---|
| `CONTENT` / `DESCRIPTION` | post text problem | provide/trim text per platform |
| `MEDIA` | media missing/invalid/mismatched | add/replace; obey FB no-mix, IG-Story-needs-media, Pinterest-single |
| `LINK` | bad/over-long link | valid URL (Pinterest link ≤2048) |
| `TITLE` | title issue | YouTube needs a title; Pinterest title ≤100 |
| `PINTEREST_BOARD` | board missing/invalid | set `pinterestBoardId` from platform-inputs |
| `TIKTOK_PRIVACY_LEVEL` | privacy not allowed for the account | pick from `privacyLevelOptions` |
| `TIKTOK_CONTENT_DISCLOSURE` | disclosure flags invalid/required | set `isBrandedContent`/`isYourBrand` **truthfully** (branded ≠ private) |
| `YOUTUBE_PRIVACY` / `YOUTUBE_CATEGORY` / `YOUTUBE_TAGS` | YouTube field invalid | valid privacy/category/tags |
| `FIRST_COMMENT` | first-comment issue | fix/withdraw the first-comment content |
| `SCHEDULE` | bad schedule | valid `scheduledFor` (UTC, future) |

## `CreatePostErrorCode` (top-level failures)
`INVALID_SOCIAL_ACCOUNT_ID` · `SOCIAL_ACCOUNT_NOT_CONNECTED` ·
`SOCIAL_ACCOUNTS_MUST_BELONG_TO_SAME_PROJECT` (returns the two conflicting IDs) ·
`DUPLICATE_SOCIAL_ACCOUNT_ID` · `INVALID_MEDIA_ID` · `INVALID_PLATFORM_SPECIFIC_DATA` ·
`VALIDATION_FAILED` · `INTERNAL_ERROR`.

## Worked example 1 — fan-out FB + IG + Pinterest (blunt indie-founder voice)
```
TARGETS: FB, IG, Pinterest (same project). FB postType=IMAGE (single image - no video mix). IG postType=POST.
Pinterest: pinterestBoardId from platform-inputs + title<=100 + single image. One image fans out cleanly.
VALIDATE -> isValid=true, 0 errors -> create. (If MEDIA error on Pinterest: you sent 2 images -> drop to 1.)
```

## Worked example 2 — TikTok video, fixing a rejection (warm studio voice)
```
SYMPTOM: 422 VALIDATION_FAILED, field TIKTOK_PRIVACY_LEVEL. CAUSE: sent PUBLIC_TO_EVERYONE but the account
only allows FOLLOWER_OF_CREATOR right now (unaudited). FIX: GET platform-inputs -> pick an allowed privacyLevel;
ensure all 8 TikTok fields present + disclosure flags truthful. RE-VALIDATE -> isValid -> create.
```

Both: every target's required fields hydrated; account values from platform-inputs; media rules satisfied;
declarations truthful; validate → fix → create atomically; no update → delete+recreate; metrics native
(never fabricated).
