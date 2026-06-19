---
name: practicum-placement-search
description: "Build and run a time-sensitive practicum/internship placement search system using approved-school portals first, targeted outside research second, lead tracking, web/contact extraction, and daily outreach/follow-up cadences. Use when a student needs to secure a practicum, field placement, internship, externship, or supervised training site by a deadline."
---

# Practicum Placement Search

## When to use

Use this skill when the user needs to find, qualify, and contact practicum/internship/field-placement sites, especially when there is a hard placement deadline, the school has an approved-site portal, outside agencies can be considered, and approval/onboarding risk matters.

## Core principle

Treat the placement search like a short, high-urgency sales pipeline:

1. Approved-school sites first because they usually have the shortest approval path.
2. Outside sites second, filtered for speed and fit.
3. Daily outreach quota with follow-ups and phone calls.
4. Immediate tracking of every lead, contact, status, next action, and deadline.
5. Fast advisor escalation when a promising outside site needs school approval.

## Intake checklist

Capture program/school/degree, deadline, start date, required hours, schedule/location/remote constraints, supervisor credentials, approved portal access, target populations/practice areas, relevant experience, and advisor guidance.

## Search strategy

### Approved portal first

Use the school platform as source of truth. Search by practice area, population, modality, location, supervisor/credential, and remote/hybrid terms. Capture site, approved status, contact, application steps, supervision, fit score, next action, and follow-up date.

### Outside research second

Use web search and crawling only for public websites and public contact information. Prioritize small/private group practices, outpatient programs, IOP/PHP/treatment centers, nonprofit counseling agencies, and agencies with internship/practicum language.

### Qualification rubric

High = strong fit, approved/fast-moving, direct contact, qualified supervision likely, realistic deadline path. Medium = good fit but contact/approval/supervision unclear. Low = poor fit, bureaucratic, no supervision evidence, or no realistic path.

## Outreach cadence

For urgent searches: identify leads daily, send tailored concise emails, call high-fit sites, follow up after 48 hours, and escalate promising non-approved sites to the advisor the same day.

## Contact extraction workflow

For public sites capture homepage, contact, careers/internship, team, and clinical leadership pages; extract emails, phones, forms, clinical director/coordinator names, and supervision clues. Prefer direct contacts over generic forms.

Privacy rule: do not scrape password-protected school portals or student-account pages unless the user explicitly asks and is logged in. Treat these as sensitive student data.

## Tracker schema

```csv
status,priority,source,approved_status,agency,fit_area,population,location,website,contact_name,email,phone,next_action,last_contact_date,follow_up_date,notes
```

## Outreach message structure

Identity, deadline/urgency, agency fit, relevant experience, clear ask, offer resume/school requirements, and ask for the correct coordinator if needed. Keep messages concise and easy to forward.

## Support files

- `references/uky-msw-long-beach-2026.md` — session-specific UKY MSW practicum context and lead filters from the June 2026 placement sprint.
- `templates/practicum-outreach-emails.md` — reusable email and phone scripts for practicum placement outreach.
