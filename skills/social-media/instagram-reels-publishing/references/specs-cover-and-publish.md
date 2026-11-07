# Specs, cover, publish + two worked examples

## Spec sheet + pre-publish checklist
```
[ ] 1080 x 1920 px, 9:16 (off 0.5625 -> crops/letterboxes)
[ ] MP4 or MOV, H.264, 30 or 60 fps, >=1080p, <=4GB, ~15-20 Mbps VBR
[ ] No other-platform watermark (TikTok logo -> remove; IG deprioritizes)
[ ] Hook text / face / CTA centered (survives 4:5 feed + 3:4 grid crops)
[ ] Has audio (with-audio beats silent)
[ ] Custom cover ready (3:4-grid-safe center; can't edit after upload)
[ ] Caption ready (keyword-first; from instagram-seo)
```

## Safe-zone map (1080 x 1920 canvas)
```
Top ~250 px      -> username/UI (atmospheric only)
Center ~1320 px  -> SAFE: hook text, face, captions, CTA. Also survives the 4:5 feed + 3:4 grid (1080x1440) crops.
Bottom ~350 px   -> caption + action buttons (NO burned-in text -- it vanishes behind the comment icon)
```

## Cover brief (-> ideogram / nano-banana; or a frame)
```
Custom upload (beats a video frame). Key subject + bold text >=60px, high contrast, in the central 3:4 (1080x1440) crop.
Design so it reads on the grid AND in the 9:16 feed. CAN'T be edited after upload -- get it right once.
```

## Publishable vs native
| Element | Where |
|---|---|
| postType: REEL | **WoopSocial** |
| The video file (9:16) | **WoopSocial** (made by a video tool/creator) |
| Caption (keyword-first) | **WoopSocial** (from instagram-seo) |
| Cover (REEL-only field) | **WoopSocial** (designed by an image tool) |
| Schedule | **WoopSocial** |
| Trending audio | native / in-app (IG licensed library -- not via API) |
| Share-to-Feed toggle | native / in-app |
| Collaborator / product tags | native / in-app |
| Media-quality setting | native / in-app |
| Reel Insights | native |

## WoopSocial publish block
```
platform: INSTAGRAM | postType: REEL | media: the 9:16 video | content.text: the keyword caption | cover: custom image (REEL-only)
Schedule it. Validate (9:16, <=4GB, MP4/MOV) via platform-specs-and-validation. Add trending audio + Share-to-Feed in-app.
```

## Worked example 1 - educational creator (blunt indie-founder voice)
```
Asset: a 45-sec 1080x1920 H.264 Reel, no watermark. Hook text top-center (clear of the 250px), CTA mid-lower (above the
350px). Custom cover in ideogram: bold 5-word title in the 3:4-safe center. Caption keyword-first (from instagram-seo).
WoopSocial publishes postType REEL + video + caption + cover, scheduled. I add the trending sound + Share-to-Feed in-app.
```

## Worked example 2 - product brand (warm studio voice)
```
A 9:16 product Reel -- we double-check it's 1080x1920 and the TikTok watermark is gone before anything else. Key shots stay
centered so the 3:4 grid crop stays clean. A custom cover (nano-banana) reads on the grid. WoopSocial publishes the postType REEL,
video, keyword caption + cover on schedule (validated via platform-specs-and-validation); we add the trending audio + product tags natively.
```

Both: 9:16 spec-clean + safe-zone-aware; a custom grid-safe cover (can't edit after); keyword caption; WoopSocial
publishes postType REEL + video + caption + cover, while trending audio + toggles + tags stay native/in-app.
