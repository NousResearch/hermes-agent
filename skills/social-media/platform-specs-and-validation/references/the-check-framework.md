# The CHECK framework — make a post publish-ready on every target

A fan-out post lands correctly only when **every target's required fields, media rules, and limits are
satisfied** — because `POST /posts` validates **atomically** (one bad target fails the whole post) and
there's **no update** to fix it after. CHECK gets a post publish-ready, then validates. This is the
**keystone the format skills point to** for the validate-before-publish step.

## C — Collect the targets + project
- List the **social-account targets** and confirm they're in the **same project** (cross-project = error).
  Note each platform so you know which field set applies.

## H — Hydrate per-platform required fields
- Fill each target's **required fields** (see the matrices): FB/IG `postType`; TikTok's eight; YouTube
  `title`+`privacy`; Pinterest `pinterestBoardId`. **Resolve account-specific values** (Pinterest board,
  TikTok `privacyLevel`) from **`GET /social-accounts/{id}/platform-inputs`** — don't guess.

## E — Enforce media + content rules
- Apply the **media gotchas**: FB no image+video mix; IG Stories need media; **Pinterest single media**;
  TikTok PHOTO vs VIDEO field applicability. Where one content item can't fan out cleanly, use
  **`contentOverride`** per target or **separate posts.** Respect text limits (e.g. Threads ~500).

## C — Confirm truthful declarations (human-judgment)
- Disclosure/compliance flags are **truthful declarations, not validation knobs**: TikTok `isBrandedContent`/
  `isYourBrand`, YouTube `madeForKids`. **Never flip them to clear an error** — fix the real field instead.
  These stay with the **person's judgment.**

## K — Knock it through validation
- Run **`POST /posts/validate`** → `isValid` + `errors` + `warnings` (same rules as create). **Resolve all
  errors, review warnings**, then create. **No update** → if you must change it, **delete + recreate**
  (only while `NOT_STARTED`).

## The publish-readiness brief a request should fill
```
TARGETS: the social accounts (same project) + each platform
REQUIRED FIELDS: per-target required set hydrated; account values from platform-inputs (board, privacyLevel)
MEDIA/CONTENT: media rules satisfied; contentOverride/separate posts where fan-out conflicts; limits respected
DECLARATIONS: disclosure/madeForKids truthful (human call) - never flipped to pass validation
VALIDATE: POST /posts/validate -> fix errors/warnings -> create (atomic); no update -> delete+recreate; metrics native
```
