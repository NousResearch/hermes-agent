---
name: adversarial-ux-test
description: Roleplay the most difficult, tech-resistant user for your product. Browse the app as that persona, find every UX pain point, then filter complaints through a pragmatism layer to separate real problems from noise. Creates actionable tickets from genuine issues only.
version: 1.0.0
author: Omni @ Comelse
license: MIT
metadata:
  hermes:
    tags: [qa, ux, testing, adversarial, dogfood, personas, user-testing]
    related_skills: [dogfood]
---

# Adversarial UX Test

Roleplay the worst-case user for your product — the person who hates technology, doesn't want your software, and will find every reason to complain. Then filter their feedback through a pragmatism layer to separate real UX problems from "I hate computers" noise.

Think of it as an automated "mom test" — but angry.

## Why This Works

Most QA finds bugs. This finds **friction**. A technically correct app can still be unusable for real humans. The adversarial persona catches:
- Confusing terminology that makes sense to developers but not users
- Too many steps to accomplish basic tasks
- Missing onboarding or "aha moments"
- Accessibility issues (font size, contrast, click targets)
- Cold-start problems (empty states, no demo content)
- Paywall/signup friction that kills conversion

The **pragmatism filter** (Phase 3) is what makes this useful instead of just entertaining. Without it, you'd add a "print this page" button to every screen because Grandpa can't figure out PDFs.

## How to Use

Tell the agent:
```
"Run an adversarial UX test on [URL]"
"Be a grumpy [persona type] and test [app name]"
"Do an asshole user test on my staging site"
```

You can provide a persona or let the agent generate one based on your product's target audience.

## Step 1: Define the Persona

If no persona is provided, generate one by answering:

1. **Who is the HARDEST user for this product?** (age 50+, non-technical role, decades of experience doing it "the old way")
2. **What is their tech comfort level?** (the lower the better — WhatsApp-only, paper notebooks, wife set up their email)
3. **What is the ONE thing they need to accomplish?** (their core job, not your feature list)
4. **What would make them give up?** (too many clicks, jargon, slow, confusing)
5. **How do they talk when frustrated?** (blunt, sweary, dismissive, sighing)

### Good Persona Example
> **"Big Mick" McAllister** — 58-year-old S&C coach. Uses WhatsApp and that's it. His "spreadsheet" is a paper notebook. "If I can't figure it out in 10 seconds I'm going back to my notebook." Needs to log session results for 25 players. Hates small text, jargon, and passwords.

### Bad Persona Example
> "A user who doesn't like the app" — too vague, no constraints, no voice.

The persona must be **specific enough to stay in character** for 20 minutes of testing.

## Step 2: Become the Asshole (Browse as the Persona)

1. Read any available project docs for app context and URLs
2. **Fully inhabit the persona** — their frustrations, limitations, goals
3. Navigate to the app using browser tools
4. **Attempt the persona's ACTUAL TASKS** (not a feature tour):
   - Can they do what they came to do?
   - How many clicks/screens to accomplish it?
   - What confuses them?
   - What makes them angry?
   - Where do they get lost?
   - What would make them give up and go back to their old way?

5. Test these friction categories:
   - **First impression** — would they even bother past the landing page?
   - **Core workflow** — the ONE thing they need to do most often
   - **Error recovery** — what happens when they do something wrong?
   - **Readability** — text size, contrast, information density
   - **Speed** — does it feel faster than their current method?
   - **Terminology** — any jargon they wouldn't understand?
   - **Navigation** — can they find their way back? do they know where they are?

6. Take screenshots of every pain point
7. Check browser console for JS errors on every page

## Step 3: The Rant (Write Feedback in Character)

Write the feedback AS THE PERSONA — in their voice, with their frustrations. This is not a bug report. This is a real human venting.

```
[PERSONA NAME]'s Review of [PRODUCT]

Overall: [Would they keep using it? Yes/No/Maybe with conditions]

THE GOOD (grudging admission):
- [things even they have to admit work]

THE BAD (legitimate UX issues):
- [real problems that would stop them from using the product]

THE UGLY (showstoppers):
- [things that would make them uninstall/cancel immediately]

SPECIFIC COMPLAINTS:
1. [Page/feature]: "[quote in persona voice]" — [what happened, expected]
2. ...

VERDICT: "[one-line persona quote summarizing their experience]"
```

## Step 4: The Pragmatism Filter (Critical — Do Not Skip)

Step OUT of the persona. Evaluate each complaint as a product person:

- **RED: REAL UX BUG** — Any user would have this problem, not just grumpy ones. Fix it.
- **YELLOW: VALID BUT LOW PRIORITY** — Real issue but only for extreme users. Note it.
- **WHITE: PERSONA NOISE** — "I hate computers" talking, not a product problem. Skip it.
- **GREEN: FEATURE REQUEST** — Good idea hidden in the complaint. Consider it.

### Filter Criteria
1. Would a 35-year-old competent-but-busy user have the same complaint? → RED
2. Is this a genuine accessibility issue (font size, contrast, click targets)? → RED
3. Is this "I want it to work like paper" resistance to digital? → WHITE
4. Is this a real workflow inefficiency the persona stumbled on? → YELLOW or RED
5. Would fixing this add complexity for the 80% who are fine? → WHITE
6. Does the complaint reveal a missing onboarding moment? → GREEN

**This filter is MANDATORY.** Never ship raw persona complaints as tickets.

## Step 5: Create Tickets

For **RED** and **GREEN** items only:
- Clear, actionable title
- Include the persona's verbatim quote (entertaining + memorable)
- The real UX issue underneath (objective)
- A suggested fix (actionable)
- Tag/label: "ux-review"

For **YELLOW** items: one catch-all ticket with all notes.

**WHITE** items appear in the report only. No tickets.

**Max 10 tickets per session** — focus on the worst issues.

## Step 6: Report

Deliver:
1. The persona rant (Step 3) — entertaining and visceral
2. The filtered assessment (Step 4) — pragmatic and actionable
3. Tickets created (Step 5) — with links
4. Screenshots of key issues

## Tips

- **One persona per session.** Don't mix perspectives.
- **Stay in character during Steps 2-3.** Break character only at Step 4.
- **Test the CORE WORKFLOW first.** Don't get distracted by settings pages.
- **Empty states are gold.** New user experience reveals the most friction.
- **The best findings are RED items the persona found accidentally** while trying to do something else.
- **If the persona has zero complaints, your persona is too tech-savvy.** Make them older, less patient, more set in their ways.
- **Run this before demos, launches, or after shipping a batch of features.**
- **Register as a NEW user when possible.** Don't use pre-seeded admin accounts — the cold start experience is where most friction lives.
- **Zero WHITE items is a signal, not a failure.** If the pragmatism filter finds no noise, your product has real UX problems, not just a grumpy persona.
- **Check known issues in project docs AFTER the test.** If the persona found a bug that's already in the known issues list, that's actually the most damning finding — it means the team knew about it but never felt the user's pain.
- **Subscription/paywall testing is critical.** Test with expired accounts, not just active ones. The "what happens when you can't pay" experience reveals whether the product respects users or holds their data hostage.
- **Count the clicks to accomplish the persona's ONE task.** If it's more than 5, that's almost always a RED finding regardless of persona tech level.

## Example Personas by Industry

These are starting points — customize for your specific product:

| Product Type | Persona | Age | Key Trait |
|-------------|---------|-----|-----------|
| CRM | Retirement home director | 68 | Filing cabinet is the current CRM |
| Photography SaaS | Rural wedding photographer | 62 | Books clients by phone, invoices on paper |
| AI/ML Tool | Department store buyer | 55 | Burned by 3 failed tech startups |
| Fitness App | Old-school gym coach | 58 | Paper notebook, thick fingers, bad eyes |
| Accounting | Family bakery owner | 64 | Shoebox of receipts, hates subscriptions |
| E-commerce | Market stall vendor | 60 | Cash only, smartphone is for calls |
| Healthcare | Senior GP | 63 | Dictates notes, nurse handles the computer |
| Education | Veteran teacher | 57 | Chalk and talk, worksheets in ring binders |

## Rules

- Stay in character during Steps 2-3
- Be genuinely mean but fair — find real problems, not manufactured ones
- The pragmatism filter (Step 4) is **MANDATORY**
- Screenshots required for every complaint
- Max 10 tickets per session
- Test on staging/deployed app, not local dev
- One persona, one session, one report
