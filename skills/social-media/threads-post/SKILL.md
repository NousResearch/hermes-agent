---
name: threads-post
description: >-
  Write and publish single posts on Meta's Threads — the Threads format-execution skill. Use when
  someone wants a "Threads post," to "post to Threads," grow on Meta's Threads, or publish a Threads
  post via WoopSocial. Writes a Threads-native conversation post (text <=500 + optional media) built
  to earn replies — Threads is conversation-first: replies outweigh likes and engagement velocity
  drives distribution. Uses the REPLY framework. Reads brand-profile + goals-and-kpis first.
  WoopSocial publishes ONE Threads post (no native multi-post chain, no polls/voice/GIF, no reply
  surface); the replying that drives reach is a human/community-management job; metrics never
  fabricated; no engagement bait. Distinct from thread-writer (a user saying "write a thread" wants
  that multi-post X chain, not this), x-growth (sibling text platform, different culture) and
  caption-writer (text only).
version: 1.0.0
---

# threads-post

A **format-execution skill** — write a Threads-native conversation post (built to earn replies) **and**
carry the WoopSocial Threads publish facts. The agent writes the **post + topic tag + media direction +
reply angles**; **WoopSocial publishes/schedules the single post**; the **replying that drives reach is
human.**

## The POV: conversation-first — reply more than you post
Threads rewards what most platforms only claim to: **genuine conversation.** **Replies outweigh likes**,
**conversation depth** and **velocity** (first 30–90 min) drive distribution, and the **For You feed
surfaces new/smaller accounts** — an open (but tightening, post-ads) window. Mosseri's rule is blunt:
**"reply much more than you post."** So write a **conversation starter** (question / hot take / opinion),
keep a **consistent niche** with **one topic tag** (topic-neighbourhoods), put **links inline** (the old
penalty was reversed), and treat the post as the *smaller half* — the replies win the reach, and that part
is human.

## Read these first
1. **brand-profile** — niche + voice (Threads punishes press-release tone).
2. **goals-and-kpis** — the goal (replies / profile visits / follower growth, not vanity likes).

## The framework: REPLY
(Depth: `references/the-reply-framework.md`.)
- **R — Reason to reply:** a conversation starter (question/hot take/opinion), not a broadcast; no bait.
- **E — Engagement velocity first:** the first 30–90 min decide reach — post at a peak window, reply fast.
- **P — Personality + niche:** authentic/original (not promo); consistent niche + **one topic tag.**
- **L — Light media + link if it fits:** photos out-engage text-only (~+60%); **links inline** (no
  first-comment tax); stay ≤500 chars.
- **Y — You reply more than you post:** the post is the smaller half; **WoopSocial publishes it, a human
  (→ community-management) does the replying.**

## The reality (verify-quarterly)
Platform stats (text-first, ~400M MAU, passed X in daily mobile, ~6.25% ER, ads global Jan 2026 →
tightening); conversation-first + Mosseri "reply more than you post"; ranking (replies>likes, depth,
velocity 30–90 min, For You surfaces new accounts); Mosseri specifics (links rewarded, bait crackdown,
credibility/originality, Account Status, Dear Algo); topic tags + niche-neighbourhoods; formats (500 chars,
photos +60%, video 5min, polls/voice/GIF native-only); avoid-list: `references/threads-2026-reality.md`.
Post types, anatomy, the WoopSocial Threads publish block + worked examples:
`references/post-recipes-and-publish.md`.

## Honest scope (never violate)
- **WoopSocial publishes ONE Threads post** (`content.text` ≤500 + optional media). **No `postType`, no
  `link` field** (links inline), **no polls/voice/GIF** (native-only), **no multi-post chain** (single
  content item). **No reply surface** → the **replying that drives reach is human** (`community-management`).
- **No update endpoint** → edit = delete + recreate (only while `NOT_STARTED`).
- **No engagement bait** (penalized + dishonest); **no press-release tone** (suppressed). **Never promise a
  reach multiple; never fabricate metrics** (native analytics only). (Scope, distinctions + connections:
  `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**threads-post (this)** = the Threads-native conversation post + the WoopSocial Threads publish facts ·
**thread-writer** = the multi-post chain ("write a thread" means that; this is ONE post — WoopSocial can't
chain) · **x-growth** = the sibling text platform, different culture (X real-time/credit-metered,
links-in-reply; Threads conversation/reply-driven, links-inline) · **caption-writer**/**hook-writer** =
the post line · **community-management** = the replies that win reach.

## Where this connects
Reads first: **brand-profile**, **goals-and-kpis**. Copy: **hook-writer**, **caption-writer**. Media
(external to WoopSocial): **image-prompt**/**ideogram**/**nano-banana**, the **video** cluster. Publish:
**scheduling-and-queue → WoopSocial** (single Threads post to a velocity window),
**platform-specs-and-validation** (validate + field rules), **content-calendar** (cadence). Engage/measure:
**community-management** (the reply work), **analytics-and-reporting** (replies/reach/profile-visits),
**experimentation-and-ab-testing** (A/B hooks/times). Replies + polls/voice stay native.

## Definition of done
A conversation-starting post (question/hot take/opinion) in the brand's voice, ≤500 chars, with one genuine
reply prompt (not bait), one niche topic tag, optional supporting media and an inline link if useful;
scheduled to a velocity window with a reply plan for the first 30–90 min; published by WoopSocial as a
single Threads post (no chain/poll/voice/GIF), validated first, edits via delete+recreate; the reply work
handed to a human/community-management, nothing fabricated and no promised reach multiple; correctly
distinguished from x-growth and caption-writer.
