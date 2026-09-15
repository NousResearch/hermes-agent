# Scope, distinctions + connections

threads-post is a **format-execution skill** — it writes a Threads-native conversation post (built to earn
replies) **and** carries the WoopSocial Threads publish facts. The agent writes the **post + topic tag +
media direction + reply angles**; **WoopSocial publishes/schedules the single post**; the **replying that
drives reach is human**.

## Honest scope (never violate)
- **WoopSocial publishes ONE Threads post** (`content.text` ≤500 chars + optional MEDIA_LIBRARY image/
  video, raw-bytes upload). **No `postType`, no `link` field** (links go inline), **no polls / voice notes
  / GIFs** (native-app only), and **no native multi-post chain** (the content array is a single item).
- **No reply/engagement surface** → the **replying that actually drives Threads reach is a human job**
  (`community-management`). The agent can draft reply angles; a person sends them.
- **No update endpoint** → edit = delete + recreate (only while `NOT_STARTED`).
- **No engagement bait** (algorithm-penalized + dishonest); **no press-release/promo tone** (suppressed).
  **Never promise a reach multiple** (organic is tightening as ads roll out) and **never fabricate metrics**
  — native analytics only. **Verify-quarterly** (`references/threads-2026-reality.md`).

## Distinct from its siblings
- **threads-post (this)** — the **Threads-native conversation post** + the WoopSocial Threads publish facts.
- **thread-writer** — the **multi-post chain**; a user who says "write a thread" means that, not this.
  This skill produces ONE Threads post (and WoopSocial can't chain — single content item).
- **x-growth** — the **sibling text platform with a different culture**: X is real-time/credit-metered with
  **links pushed to the reply**; Threads is conversation/reply-driven with **links rewarded inline.** Don't
  cross-post the same text.
- **caption-writer** / **hook-writer** — supply the **post line** this skill builds into a Threads post.
- **community-management** — the **replies** (the half of Threads that wins reach).
- per-platform **strategy/growth** skills — the broader channel strategy; this is the post-level execution.

## Where this connects
- **Reads first:** `brand-profile` (niche/voice), `goals-and-kpis` (replies / profile visits / follower growth).
- **Copy:** `hook-writer` (the take/question), `caption-writer` (the line + reply prompt), `brand-profile` (voice).
- **Media (external to WoopSocial):** `image-prompt`/`ideogram`/`nano-banana` (a supporting photo), the `video`
  cluster (short clip).
- **Publish:** `scheduling-and-queue → WoopSocial` (single Threads post to a velocity window),
  `platform-specs-and-validation` (validate + the field rules), `content-calendar` (cadence).
- **Engage/measure:** `community-management` (the reply work), `analytics-and-reporting` (replies/reach/
  profile-visits readout), `experimentation-and-ab-testing` (A/B hooks/post times). **Replies + polls/voice stay native.**
