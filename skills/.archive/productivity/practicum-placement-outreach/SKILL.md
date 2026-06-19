---
name: practicum-placement-outreach
description: "Build and operate urgent practicum/internship placement searches: approved placement portals first, public web lead research second, contact extraction, outreach drafting, tracking, and follow-up."
---

# Practicum Placement Outreach

## Core strategy

When a student needs a placement by a hard deadline, run a structured outreach pipeline:

1. Capture school/program requirements and deadline.
2. Search approved placement portals first.
3. Research outside leads only on public websites.
4. Rank leads by fit, approval speed, supervision likelihood, and direct contact availability.
5. Draft concise outreach and call top leads quickly.
6. Track all statuses and follow-up dates.

## Data model for the tracker

Use columns: status, priority, source, approved_status, agency, fit_area, population, location, website, contact_name, email, phone, next_action, last_contact_date, follow_up_date, notes.

## Lead qualification rubric

High-priority leads have strong practice/population fit, approved or likely fast approval, direct contact, supervision evidence, and realistic timeline.

## Public web/Crawl4AI workflow

Use crawling only for public pages: homepage, contact, careers/internship, team, clinical leadership. Extract emails, phone numbers, contact forms, director/coordinator names, and supervision clues.

### Crawl4AI setup pitfall

Use a current Python version and run setup/doctor before relying on browser crawling. Do not crawl protected student portals unless the user explicitly provides a logged-in workflow and privacy boundary.

## Outreach drafting

Keep messages concise and forwardable: identity, school/program, placement start, relevant fit, direct ask, offer resume/requirements.

## Email sending safety

Do not send without explicit user approval. Draft first, verify recipients, and avoid exposing private student data unnecessarily.

## Support files

- `references/uky-msw-long-beach-practicum.md` — session-specific UKY MSW Long Beach practicum context.
