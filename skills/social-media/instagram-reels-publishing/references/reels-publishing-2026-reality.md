# Instagram Reels publishing 2026 — verified

*Volatile. Re-verify quarterly (Meta updates specs via Help Center edits before announcing).*

## The POV: the Reel is made elsewhere — this gets it OUT the way IG displays it
Great Reels get squashed/cropped/blurred at the publish step — a **sizing/settings problem, not a content
problem.** Design for **how Instagram actually displays a Reel**, not how it looks in your editor.

## The spec
- **9:16, 1080×1920 px** (the math: 1080/1920 = **0.5625** — anything off this **crops or letterboxes**).
  **MP4 or MOV**, **H.264** codec, **30 fps (or 60)**, **≥720p (1080 ideal)**, **≤4GB.** Export ~**15–20 Mbps
  H.264 VBR (2-pass)** to survive compression; higher source res → better after compression; clean filenames
  (letters/numbers/dashes). **Length:** up to ~90s historically; now **up to 3 min in-app**; **up to ~15 min
  via third-party tools** (verify).

## The crop trap (3 display contexts)
- **Reels tab = full 9:16** (full frame). **Main feed preview = 4:5** crop (cuts top/bottom). **Profile grid =
  3:4 (1080×1440)** — the **2025 rectangular-grid change** (rolled out Jan 2025; removes the top ~240px + bottom ~240px; was 1:1).
  **Link/DM preview ≈ 1.91:1.** → **Design for 9:16 but keep critical content (hook text, face, CTA) centered**
  so it survives every crop.

## Safe zones (UI overlays)
- Reserve the **top ~250px** (username) and **bottom ~350px** (caption + action buttons — bigger than Stories'
  ~340px because Reels has more buttons). Keep burned-in captions/text in the **central ~1320px-tall safe
  zone** (text in the bottom 350px vanishes behind the comment icon). **March 2026: Meta unified FB + IG Reels
  safe zones** (~top 250 clear) so one asset posts cleanly to both.

## Cover / thumbnail
- Choose the cover **right before publishing** — a **video frame** or a **custom uploaded image**. **You can't
  edit the cover after uploading** (choose wisely). It renders **9:16 in the Reels feed** but is **cropped to
  3:4 (1080×1440) on the profile grid** — so put key elements + **bold text (≥60px)** in the **3:4-grid-safe
  center.** Best practice: **a custom cover beats a video frame** (frames are rarely composed well as static
  thumbnails).

## Publish-adjacent best practices
- **Audio:** Reels **with audio outperform silent**; **trending audio is pushed harder** (a track under ~10k
  uses but rising) — *selecting it is native/in-app.* **Hook in the first 3s** (→ `reels-script`). **Burn
  captions in** (most watch sound-off) — keep in the safe zone (→ `captions-and-clipping`). **Remove
  other-platform watermarks** (a TikTok watermark signals lazy repurposing → IG deprioritizes it). **Blurry
  after upload?** low bitrate, IG still processing (~30 min), re-exporting repeatedly, or color-space mismatch
  (use **Rec. 709**; slightly desaturate saturated reds/darks). The **"upload at highest quality"** media
  setting is native/in-app.
