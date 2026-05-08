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

Set up with `cronjob` tool. Prompt template:

```
Every weekday morning (Mon-Fri), search for senior-level roles at [Northrop Grumman, Amgen, Teledyne] near Ventura CA.

Include: senior software engineer, staff software engineer, principal engineer, engineering manager, program manager, systems engineer, software engineering manager, hardware systems engineer.

Exclude: The Trade Desk (adtech), Verilog/VHDL/FPGA/firmware/hardware-description roles, embedded hardware bringup. Note if clearance may be required.

For each job found, report: Company | Title | City | Salary if available | Direct URL.

Format:
**Ventura Area Jobs Report — [DAY, DATE]**

**Northrop Grumman (Camarillo ~26 min)**
- ...

**Amgen (Thousand Oaks ~32 min)**
- ...

**Teledyne (Thousand Oaks ~37 min)**
- ...

If no new relevant postings, say "No new relevant postings today."
Keep under 500 words. Deliver to Telegram.
```

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

See `references/ventura-employers.md` for compiled employer details, roles, salary ranges, and commute data from current session research.