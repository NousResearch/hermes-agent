---
name: career-ops-job-search
description: "Run a filtered AI job search with career-ops: scan boards, score a posting against the candidate's CV, tailor a résumé + cover letter, and track the application."
platforms: [linux, macos, windows]
version: 1.0.0
author: Fighter90
author_url: https://github.com/Fighter90/career-ops-ui
license: MIT
category: productivity
metadata:
  hermes:
    tags: [job-search, resume, cv, ats, cover-letter, career, applications, hiring]
---

# career-ops Job Search

Drive **career-ops** — an agentic, *filter-first* job-search pipeline — end to end:
find the few postings worth applying to, score each against the candidate's **real
CV** (by reasoning about fit, not keyword matching), tailor a résumé + cover letter
per listing, and keep an honest application tracker.

career-ops is a **filter, not a spray-and-pray tool**: it recommends AGAINST applying
to anything scoring below **4.0 / 5**. The candidate's time — and the recruiter's — is
valuable. Every application is reviewed by a human before it is submitted.

- Pipeline: <https://github.com/santifer/career-ops>
- Optional local web UI: <https://github.com/Fighter90/career-ops-ui>

## When to Use

- The user asks to **find jobs**, **evaluate a specific posting**, **tailor a résumé /
  cover letter** for a role, or **check on their applications**.
- A **job-posting URL** is shared and the user wants a fit assessment or an application
  prepared.
- The user wants a **recurring scan** of job boards for new matches.

Do NOT use this to mass-apply, and never invent experience the candidate doesn't have.

## Prerequisites

- A career-ops checkout whose `cv.md` and `config/profile.yml` are filled in with the
  candidate's real details. Verify with `npm run doctor`. If they are incomplete, STOP
  and ask the user to complete them — career-ops must never fabricate a CV.
- Node ≥ 18. Optional: the career-ops-ui web app (`npm start` → `http://127.0.0.1:4317`)
  for a visual layer over the same files.

## Procedure

1. **Confirm setup.** Run `npm run doctor`. If the CV or profile is incomplete, stop and
   ask the user to complete them first.
2. **Scan for matches.** Run the scan mode (`npm run or:scan`, or tell the driving agent
   "Run the career-ops scan mode"). It collects postings from the configured boards into
   the pipeline. Summarize the *new* matches — don't evaluate everything blindly.
3. **Evaluate a posting.** For a chosen posting or a shared URL, run the auto-pipeline
   (`npm run or:eval`, or "Evaluate this JD with career-ops auto-pipeline: <url>"). It
   reasons about the CV vs the job description and writes a scored report (out of 5) under
   `reports/`. **Only advance postings scoring ≥ 4.0.** Report the score and the reasoning
   honestly, gaps included.
4. **Tailor + prepare.** For a strong match, run the apply flow (`npm run or:apply`): it
   tailors the résumé to the listing — grounded ONLY in the real CV/profile, never
   inventing experience — and drafts a cover letter, and can render a PDF (`npm run pdf`).
   Show the user the tailored bullets and the cover letter for review BEFORE anything is
   submitted. career-ops' fact-check gate (`npm run cv:verify-facts`) blocks unsupported
   claims — respect it.
5. **Track the outcome.** Applications live in `data/applications.md`. When the user
   applies, record it; when a result lands (rejected / interview / offer / hired), record
   the outcome (`node outcome.mjs <report#|company> <type>`). Surface follow-up timing
   (`node followup-cadence.mjs`) so nothing goes stale.
6. **Research on demand.** If the user asks about a company, the role, or a posting's
   legitimacy, research it (the posting page, the company's careers site, funding/news)
   and answer plainly — flag anything that looks like a fake or expired listing.

## Verification

- [ ] `npm run doctor` is green (real CV + profile present) before any evaluation.
- [ ] **No fabricated** experience, metrics, dates, or employers in any tailored CV or
      cover letter — everything traces to the real CV/profile.
- [ ] Postings scoring **< 4.0** are reported but NOT advanced to an application without
      an explicit user override.
- [ ] The user **reviewed** the tailored résumé and cover letter before submission.
- [ ] Every submitted application is **recorded** in the tracker with its status.

## Example

> **User:** "Any good backend roles this week? And take a look at this one:
> `https://boards.greenhouse.io/acme/jobs/123`"
>
> 1. `npm run doctor` → green.
> 2. `npm run or:scan` → "7 new postings; 2 look on-target: Acme Senior Backend, Globex
>    Platform Eng."
> 3. "Evaluate this JD with career-ops auto-pipeline: `…/acme/jobs/123`" → report scores
>    **4.3 / 5**: "Strong Go + distributed-systems match; gap: no formal Kubernetes cert,
>    but hands-on evidence is in the CV."
> 4. Since ≥ 4.0, run the apply flow → tailored résumé (Go / microservices bullets
>    surfaced) + a cover letter grounded in the CV. Show both to the user for review.
> 5. User approves and applies → record in `data/applications.md` as **Applied**; set a
>    follow-up reminder.
> 6. Globex scored **3.6** → reported, but NOT advanced (below the 4.0 filter) unless the
>    user insists.
