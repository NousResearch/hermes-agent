# Content types, publish config + two worked examples

## Swipeable content-type menu (pick by niche + goal)
| Type | Why it completes | Goal it serves |
|---|---|---|
| Listicle ("7 X that…") | each item = one swipe; strong save impulse | saves, reach |
| Before / after | the reveal pulls swipes; deeply shareable | shares, reach |
| Step tutorial | one step per slide mirrors how people follow | saves, follows |
| Hot take / unpopular opinion | splits the room → comments | comments, reach |
| Photo / aesthetic dump | native, low-friction, visual | reach, brand affinity |
| Story arc ("how I went from X→Y") | narrative tension to the payoff | follows, comments |
| Myth vs fact | disagreement drives shares/comments | comments, shares |

## Slide blueprint (5–7 slides)
```
Slide 1  HOOK: visual + overlay (question/claim) + payoff promise
Slide 2  set up the problem / first item (one idea)
Slide 3-5 progressive reveal, one idea each, cliffhanger to the next
Last     REWARD (best insight/punchline) + ONE CTA (save/comment/follow); deliver the promise
Caption  question + context; 1-2 hashtags; overlay keywords mirror search terms (TikTok SEO)
```

## WoopSocial PHOTO publish-config block (required fields)
```
postType: PHOTO
privacyLevel: PUBLIC_TO_EVERYONE | SELF_ONLY | MUTUAL_FOLLOW_FRIENDS | FOLLOWER_OF_CREATOR
             (discover allowed values via GET /social-accounts/{id}/platform-inputs)
allowComment: true            # comments are the engagement engine on TikTok
autoAddMusic: true            # APPLIES to PHOTO - a track lifts distribution
isYourBrand: <true if promoting your own brand>      # TikTok commercial-content disclosure
isBrandedContent: <true if a third party paid/gifted> # paid-partnership disclosure - set TRUTHFULLY
postMode: DIRECT_POST         # or MEDIA_UPLOAD = send to TikTok inbox for the creator to finish
allowDuet / allowStitch: required by the API contract but IGNORED for PHOTO (set either)
cover: NOT used for PHOTO (video-only)
```
Then **validate** (`POST /posts/validate`) — watch for `TIKTOK_PRIVACY_LEVEL`, `TIKTOK_CONTENT_DISCLOSURE`,
`MEDIA`, `CONTENT`. **No update endpoint** → edit = delete + recreate (only while `NOT_STARTED`). Media
specs + raw-bytes upload: see `tools/integrations/woopsocial.md`.

## Worked example 1 — listicle (blunt indie-founder voice)
```
ANGLE: "5 settings that fix slow Notion" (SaaS niche). SLIDE 1: "your Notion is slow. 4 of these are why."
2-5: one setting each, "the last one is the big one." LAST: the #1 fix + "save this / comment your load time."
PUBLISH: PHOTO, PUBLIC, allowComment true, autoAddMusic true, isYourBrand false, isBrandedContent false, DIRECT_POST. 1 hashtag.
```

## Worked example 2 — before/after, paid partnership (warm studio voice)
```
ANGLE: "the receipts shoebox -> clean books in 3 steps" (bookkeeping). SLIDE 1: the messy before + "this took 20 min."
2-4: each step. LAST: the after + "save this for tax season." PAID by a software brand -> isBrandedContent: TRUE (disclosed).
PUBLISH: PHOTO, PUBLIC, allowComment true, autoAddMusic true, isYourBrand false, isBrandedContent TRUE. Validate -> publish.
```

Both: swipeable type matched to niche; hook + progressive reveal + reward/CTA for completion; 9:16
1080×1920; truthful disclosure flags; images made externally; WoopSocial publishes native PHOTO; metrics
from native analytics (never fabricated).
