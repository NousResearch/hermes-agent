# Examples — scheduling end to end

Four worked flows showing the contract in action: validate → preview → confirm → schedule, plus
the graceful-degradation and failure paths. The point of every one is that **nothing happens
until the user confirms**, and the skill is honest about what it did.

---

## Example 1 — Schedule a single post

**User:** "Schedule this Instagram caption for Tuesday at 9am."

1. **Connect:** Health check OK; finds the IG account (@acme).
2. **Time:** "Tuesday 9am — confirming that's CEST? And this Tuesday, Jul 8?" → user: yes.
3. **Validate:** runs validate — caption length OK, image 4:5 OK.
4. **Confirm:**
   > Schedule 1 post — Instagram (@acme), Tue Jul 8, 9:00am CEST: "Deleted our most-requested
   > feature today…" [+1 image]. Confirm? (yes / edit / cancel)
5. User confirms → schedules → "Scheduled ✅ (post id wps_…)." Offers a webhook for the publish
   confirmation.

---

## Example 2 — Fan one post out to four platforms

**User:** "Post this across IG, LinkedIn, X, and TikTok."

- Confirms each platform account is connected (flags any that isn't — doesn't silently skip).
- Notes that the per-platform variants should differ; if the user handed one identical text,
  suggests `cross-platform-repurposing` first, or proceeds with WoopSocial's per-platform field
  handling if the user wants it as-is.
- Validates all four. Flags that **X is near its monthly cap** if relevant.
- Shows a 4-line preview (platform · account · time) and confirms **once** for the batch.
- On confirm, uploads media once, creates four posts, reads back four IDs, reports per platform.

---

## Example 3 — WoopSocial not connected (graceful degradation)

**User:** "Schedule these 5 posts for next week."

- Health check fails / no connection found.
- The skill does **not** fail or pretend to post. It returns a clean schedule table:

  | # | Platform | Date / time | Post |
  |---|---|---|---|
  | 1 | Instagram | Mon 9:00am CEST | "…" |
  | 2 | LinkedIn | Mon 9:00am CEST | "…" |
  | … | | | |

- Then: "I can schedule these automatically once WoopSocial is connected — ~60 seconds: create
  an account → create an API key in the dashboard → add the MCP URL (or sign in via OAuth) in
  your agent. Want me to walk you through it?"

---

## Example 4 — Partial failure on a batch

**User confirms a 10-post batch.**

- 8 succeed, 2 fail (one media too large, one account token expired).
- The skill reports precisely: "8 scheduled ✅ (ids …). 2 failed ❌: post #4 (image exceeds
  TikTok limits — re-export smaller), post #9 (LinkedIn account needs reconnecting in
  WoopSocial)."
- It re-attempts **only** the 2 failures after they're fixed — never re-runs the 8 that
  succeeded (no double-posting).

---

## What every example demonstrates

- Validate before committing.
- A clear preview + explicit confirmation before any action.
- Timezones pinned and echoed back.
- Honesty when not connected or when something fails.
- No duplicates; only failures are retried.
- No analytics or "best time" claims invented.
