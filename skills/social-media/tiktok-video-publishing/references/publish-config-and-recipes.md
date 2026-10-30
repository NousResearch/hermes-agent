# Caption SEO, publish config + two worked examples

## Caption-SEO formula (first ~100 chars do the work)
```
[PRIMARY KEYWORD PHRASE - the actual topic] - [value-prop: what they get in one line].
[secondary keyword / dual-target] #nichehashtag #secondary
```
Example: "Beginner home workout no equipment - 15 min, no jumping, no gym. #homeworkout #noequipment".
Lead with the search phrase (not "this saved me"); mirror it in **on-screen text** + the first spoken line.

## Reach-settings defaults (and the brand-safety tradeoff)
| Field | Default for reach | Restrict when |
|---|---|---|
| allowComment | ON (engagement engine) | active harassment / sensitive topic |
| allowDuet | ON (free distribution) | brand-safety / legal / sensitive |
| allowStitch | ON (free distribution) | brand-safety / legal / sensitive |
| privacyLevel | PUBLIC_TO_EVERYONE | testing (SELF_ONLY) — not for branded |

## WoopSocial VIDEO publish-config block (required fields)
```
postType: VIDEO
privacyLevel: from platform-inputs (PUBLIC_TO_EVERYONE for reach; branded can't be private)
allowComment: true            # engagement engine
allowDuet: true               # APPLIES to VIDEO - free distribution
allowStitch: true             # APPLIES to VIDEO - free distribution
cover: <image media>          # the video's cover/thumbnail -> the title-card look (1080x1920)
isYourBrand: <true if promoting own products>        # disclosure - truthful
isBrandedContent: <true if a third party paid>        # disclosure - truthful; #ad up front
postMode: DIRECT_POST         # or MEDIA_UPLOAD = inbox-to-finish (add a trending sound natively)
autoAddMusic: required by the API but IGNORED for VIDEO (it applies to PHOTO)
```
Then **validate** (`POST /posts/validate`) — watch `TIKTOK_PRIVACY_LEVEL`, `TIKTOK_CONTENT_DISCLOSURE`,
`MEDIA`, `CONTENT`. **No update** → edit = delete + recreate (only while `NOT_STARTED`). Media specs +
raw-bytes upload: `tools/integrations/woopsocial.md`.

## DIRECT_POST vs MEDIA_UPLOAD
**DIRECT_POST** = publishes straight to TikTok (everything set via the API). **MEDIA_UPLOAD** = lands in
the creator's TikTok **inbox** to finish in-app — choose it when you want a **licensed trending sound** (the
API can't attach Commercial Music) or last-touch edits.

## Worked example 1 — educational, own brand (blunt indie-founder voice)
```
CAPTION: "Notion running slow? 5 settings that fix it - do #2 first. #notiontips #productivity"
CONFIG: VIDEO, PUBLIC, comment/duet/stitch ON, cover = "5 settings that fix slow Notion" title frame, isYourBrand TRUE, isBrandedContent false, DIRECT_POST. 1080x1920, ~45s. Validate -> schedule.
```

## Worked example 2 — paid partnership + trending sound (warm studio voice)
```
CAPTION: "Bookkeeping for freelancers - 3 steps before tax season. #ad #smallbusiness #bookkeeping"
CONFIG: VIDEO, PUBLIC, comment/duet/stitch ON, cover = step-1 title frame, isBrandedContent TRUE (paid), isYourBrand false. Wants the trending sound -> postMode MEDIA_UPLOAD (creator adds it in the inbox). Validate -> publish.
```

Both: keyword-led caption; reach settings open; title-card cover; truthful disclosure (#ad up front); specs
right; video file external; WoopSocial publishes postType VIDEO + auto-AI-disclose; metrics native (never fabricated).
