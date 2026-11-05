# Scope, distinctions + connections

youtube-publishing-and-metadata is a **format-execution skill** — it writes the YouTube publish metadata
(title, description+chapters, tags) and sets category/privacy/madeForKids. The agent **writes the metadata
+ config**; the **video FILE + script are inputs**; **WoopSocial publishes via the YouTube fields**; the
**thumbnail + A/B testing are native Studio**.

## Honest scope (never violate)
- **The agent writes the metadata** (title/description/chapters/tags) + picks **category/privacy** + sets
  **madeForKids**. The **video FILE + script are inputs** (`youtube-long-form`/`youtube-shorts` +
  filming/editing) — **WoopSocial doesn't film/edit/generate video.**
- **The THUMBNAIL is a native YouTube Studio step** — **WoopSocial has no thumbnail field** (and the
  thumbnail is a major CTR lever, so brief it via `thumbnail-design` and have a human upload it).
  **A/B title testing** is also native Studio.
- **WoopSocial publishes via the YouTube fields** (`title`/`privacy`/`category`/`tags`/`madeForKids`;
  **description = `content.text`**; raw-bytes video upload). **No update** → delete+recreate before publish;
  **metadata edits happen in Studio** after publish.
- **`madeForKids` set truthfully** (COPPA/FTC — child-directed content; mislabeling is illegal, not a reach
  lever). **Affiliate + AI disclosure** where relevant. **No keyword-stuffing** (penalized). **Never promise
  a ranking; never fabricate metrics** — native analytics only. **Verify-quarterly.**

## Distinct from its siblings
- **youtube-publishing-and-metadata (this)** — the **publish METADATA + config** (title/description/tags/
  category/privacy/madeForKids).
- **youtube-long-form** / **youtube-shorts** — write the **SCRIPT/content**; this skill **publishes** it.
- **thumbnail-design** — the **thumbnail** craft (uploaded natively — no API field; this skill briefs it).
- **caption-writer** / **hook-writer** — generic copy; this skill owns the **YouTube metadata + config.**

## Where this connects
- **Reads first:** `brand-profile` (niche/voice), `goals-and-kpis` (search reach / subs / watch time), and
  the **script/video** (often from `youtube-long-form`/`youtube-shorts`).
- **Content inputs (external to WoopSocial):** `youtube-long-form`/`youtube-shorts` (the script), the
  **video** cluster (`veo-3`/`luma`/`kling`) + editing, `thumbnail-design` (the thumbnail, native upload),
  `hook-writer`/`caption-writer` (copy lines).
- **Publish:** `scheduling-and-queue → WoopSocial` (YouTube fields), `platform-specs-and-validation`
  (validate + field rules), `content-calendar` (cadence).
- **Reuse/measure:** `content-recycling` (→ Shorts/clips), `analytics-and-reporting` (search/watch-time/
  CTR readout), `experimentation-and-ab-testing` (title/thumbnail A/B — run natively in Studio). **Video/thumbnail/A/B stay
  external/native.**
