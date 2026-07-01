---
name: crwd-reminders-followups
description: "Set reminders and schedule follow-up check-ins for a CRWD member — before a gig deadline, or to circle back later ('check with me tomorrow about my submission'). Use when a member asks to be reminded, mentions a deadline, or when a proactive follow-up would keep them from losing a payout."
version: 1.0.0
metadata:
  hermes:
    tags: [crwd, reminder, deadline, follow-up, followup, check-in, schedule, cron]
    related_skills: [crwd-gig-execution, crwd-gig-discovery]
    requires_toolsets: [cronjob]
---

# CRWD Reminders & Follow-ups

Two jobs: **deadline reminders** (don't let a payout die to a clock) and **follow-up
check-ins** (circle back so the member doesn't get dropped).

## When to Use

- "Remind me before this is due."
- "Can you check back with me tomorrow?"
- A gig deadline is approaching and the member hasn't finished.
- After helping with something that isn't resolved yet — offer to follow up.

## Procedure

1. **Get the real timing.** For a deadline reminder, look up the gig's `end_date` with
   `crwd_db` (`get_gig_details` / `get_user_gigs`) — don't guess when something is due.
2. **Schedule it with the `cronjob` toolset.** Create a one-off job for the reminder or
   follow-up. Include enough context in the job so the future run is useful (which gig, what
   you're following up on, the member).
3. **Deadline reminders:** offer proactively when a window is closing — *"Want me to remind
   you the day before this is due?"* — then schedule it.
4. **Follow-ups:** when something is left open ("I'll submit tonight", a pending review),
   offer to check back — *"I'll follow up tomorrow to make sure your submission went
   through."* — and schedule a check-in, not just a passive reminder.
5. Confirm what you set, in one line: *"Done — I'll ping you tomorrow morning about the Pul
   Tool submission."*

## Pitfalls

- Don't schedule against a made-up time — pull the real `end_date` first.
- Don't over-promise cadence; one useful reminder beats several nags.
- Keep the confirmation short and specific (what + when).

## Verification

- The reminder/follow-up time is based on real gig data or the member's stated time.
- A `cronjob` was actually created (not just promised), and you confirmed it in one line.
