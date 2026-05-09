---
name: job-search
description: "Career research and job monitoring for Gordon Rouse — Ventur area tech jobs, daily reports, company deep-dives."
version: 1.0.0
author: Gordon Rouse
license: MIT
metadata:
  hermes:
    tags: [jobs, career, ventura, northrop-grumman, amgen, teledyne, research]
    category: productivity
---

# job-search

Career research, employer analysis, and automated job monitoring for Gordon's relocation from Milpitas to Ventura CA.

## Gordon's Profile
- **Current:** Director of Engineering at KLA (semiconductor capital equipment), software + systems engineering + program management background
- **Plan:** Relocate to Ventura (1920 E Linda Vista Ave) within 1 year; open to IC or management roles
- **Preferences:** IC track preferred, but will consider management. No Verilog/VHDL/firmware/hardware-description. No adtech (The Trade Desk excluded).
- **Commute threshold:** ~30 min max
- **Work style:** Concise. Wants direct answers, not explanations. Check memory for full preferences.

## Key Employers (Ventura Area)

| Company | Location | Drive from home | Notes |
|---|---|---|---|
| Northrop Grumman | Camarillo | ~26 min | Defense/aero, software + systems + PM |
| Amgen | Thousand Oaks | ~32 min | Biotech/pharma, manufacturing systems |
| Teledyne | Thousand Oaks | ~37 min | Aerospace/defense electronics, imaging sensors |

**Excluded:** The Trade Desk (adtech, boring). Goleta/Raytheon (commute too far).

## Daily Job Report (Cron)

Set up with `cronjob` tool. Cron job ID: `041c2bde2ba9` (Ventura Area Tech Jobs Daily).
Schedule: `0 14 * * 1-5` (2 PM UTC = 7 AM PT, Mon–Fri). Delivery: `origin` (delivers output to this Telegram chat automatically — do NOT set `deliver: telegram`; the gateway origin IS the Telegram delivery path).

Prompt template (attach as cron prompt, not as a separate sub-agent skill):
```
Every weekday morning (Mon-Fri), check careers pages at three companies near Ventura CA for senior-level roles matching Gordon's background: software engineering management, systems engineering management, program management, senior IC (staff/principal). Gordon has 20+ years in tech including Director of Engineering at KLA (semiconductor capital equipment). Prefers IC track but will consider management.

Search the following companies directly:

1. Northrop Grumman Camarillo (jobs.northropgrumman.com) — search for: software engineer OR systems engineer OR program manager OR manager programs in Camarillo CA
2. Amgen Thousand Oaks (careers.amgen.com) — search for: senior engineer OR staff engineer OR manager OR associate director in Thousand Oaks engineering
3. Teledyne Thousand Oaks/Camarillo — search for: software engineer OR systems engineer OR engineering manager at teledyne.com careers

Filter criteria:
- Include: senior software engineer, staff software engineer, principal engineer, engineering manager, program manager, systems engineer, software engineering manager
- Exclude: The Trade Desk, Verilog/VHDL/firmware/hardware-description roles, embedded hardware/firmware bringup
- Exclude jobs requiring active security clearance (note if clearance may be needed but not required)
- Focus on roles that would fit a Director-level candidate willing to go IC or manage

For each job found, report:
- Company
- Title
- Location (city)
- Salary if available (often shown on Indeed, Glassdoor, ZipRecruiter, or the company's own site)
- Direct URL to the job posting

Format the report as:

**Ventura Area Jobs Report — [DAY, DATE]**

**Northrop Grumman (Camarillo ~26 min from home)**
- [Title] | [City] | $[salary] if available
  [1-line description]
  [link]
- No new relevant postings

**Amgen (Thousand Oaks ~32 min from home)**
- [Title] | [City] | $[salary] if available
  [1-line description]
  [link]
- No new relevant postings

**Teledyne (Thousand Oaks ~37 min from home)**
- [Title] | [City] | $[salary] if available
  [1-line description]
  [link]
- No new relevant postings

If no relevant jobs at a company, say "No new relevant postings today."
Keep total under 500 words. Deliver to this Telegram chat.
```

**Cron output location:** `/opt/data/cron/output/041c2bde2ba9/` (latest run: `YYYY-MM-DD_HH-MM-SS.md`). Parse the "## Response" section for the actual job report. The "## Prompt" section is the skill attachment and can be skipped.

## Company Deep-Dive Workflow

1. Identify target companies (commute <30 min from home)
2. Check careers pages directly (Indeed, LinkedIn, company job boards)
3. Cross-reference salary via ZipRecruiter, Glassdoor, Indeed
4. Filter by role type (IC vs management), domain (exclude hardware-description, adtech)
5. Assess fit against Gordon's background: software engineering management, systems engineering management, program management, senior IC
6. Present: title, company, location, salary, link, 1-line fit assessment

## Known Good Search Patterns

```bash
# Northrop Grumman Camarillo
site:jobs.northropgrumman.com OR site:linkedin.com "software engineer" OR "systems engineer" OR "program manager" Camarillo

# Amgen Thousand Oaks
site:careers.amgen.com OR site:linkedin.com "senior engineer" OR "staff engineer" OR "manager" Thousand Oaks engineering

# Teledyne Thousand Oaks/Camarillo
site:linkedin.com OR site:indeed.com "software development engineer" OR "systems engineer" OR "program manager" Teledyne Thousand Oaks OR Camarillo

# Salary ranges
site:ziprecruiter.com OR site:glassdoor.com [company] [role] Thousand Oaks OR Camarillo
```

## Reference Data

See `references/ventura-employers.md` for compiled employer details, roles, salary ranges, ATS quirks, and search patterns from verified research.