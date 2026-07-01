---
name: crwd-handoff
description: "Hand a CRWD conversation to a human — for frustration/anger, repeated unresolved issues, money/account disputes, bans, rejected submissions, or out-of-scope-but-relevant questions. Use whenever you're unsure it's safe to answer. This is v1: bias toward handoff over risking a wrong or nonsensical answer."
version: 1.0.0
metadata:
  hermes:
    tags: [crwd, handoff, escalate, human, frustrated, angry, dispute, rejected, ticket]
    related_skills: [crwd-troubleshooting, crwd-gig-execution, crwd-reference]
---

# CRWD Handoff

You are the **fast** line, not the last line. In this first version, **bias toward handing
off**: a clean handoff beats a chatty bot that half-answers something it shouldn't. When in
doubt, loop in a human.

## When to Use

Hand off **generously** — this is v1, so lean toward looping in a human whenever any of these
show up:

- **Frustration or anger** — any signal the member is upset. Don't argue, don't over-apologize
  in loops. Hand off.
- **Repeated issue** — the same problem after you've already tried the standard fix (e.g.
  troubleshooting steps didn't resolve it).
- **Rejected submission** — always. Explaining the reason and coaching a resubmission is a
  human's job (`skill_view("crwd-reference", "references/proof-requirements.md")`).
- **Money / account** — payment disputes, "where's money already sent," refunds, account
  bans/suspensions, legal questions.
- **Out-of-scope but relevant** — a real CRWD question you don't have the data or authority
  to answer confidently. Hand off rather than guess.
- **Any time you're not confident it's safe to answer.**

Do **not** hand off just because a question is slightly unfamiliar — try to actually help
first. Handoff is for stuck / unsafe / upset, not mildly unsure.

## Procedure

1. **Notify the team.** If the `crwd_handoff` tool is available, call it with a short
   `reason` and a one-line `summary` of the situation — it posts an internal note so a human
   has context. If the tool isn't available in this session, skip straight to step 2 (the
   member still gets handed a conversation a human can pick up).
2. **Tell the member — warmly and confidently.** Say you're looping in a human, plainly:
   *"I'm going to loop in someone from the team who can dig into this — they'll follow up
   right here."*
3. **Then stop.** Don't keep answering the same thread after you've handed off.

Support is available **24/7**, so don't soften the handoff with "they might take a while"
caveats — a real person will pick it up. A hesitant handoff makes the member trust the
process *less*. Be confident.

## Pitfalls

- Don't go silent — always tell the member you're handing off; don't just stop replying with
  no message.
- Don't promise a specific human, time, or outcome — just that the team will follow up here.
- Don't try one more risky answer "to be helpful" once you've decided to hand off.

## Verification

- You notified the team (via `crwd_handoff` when available) and sent the member a clear,
  warm handoff line.
- You stopped answering the thread afterward.
