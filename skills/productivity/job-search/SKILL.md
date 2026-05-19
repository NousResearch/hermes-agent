---
name: job-search
description: "Career research and job monitoring for Gordon Rouse — Ventura area tech jobs, daily reports, company deep-dives."
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
- **Commute threshold:** ~30 min max. Exclude in-office/hybrid roles in Agoura Hills; Ventura ↔ Agoura Hills is too long for a regular commute. Remote roles based there are acceptable.
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
Schedule: `0 14 * * 1-5` (2 PM UTC = 7 AM PT, Mon–Fri). Delivery: `origin`.

**Important — cron job prompt style:**
- Attach prompt TEXT directly, not a skill reference
- Do NOT attach `autonomous-ai-agents/hermes-agent` or any skill to cron jobs that just do web search — skills are heavy and unnecessary for simple search tasks
- Set `skills: []` explicitly when creating or updating job crons

Prompt template (attach as cron prompt, not as a separate sub-agent skill):
```
Every weekday morning (Mon-Fri), search for senior-level roles at Northrop Grumman, Amgen, and Teledyne near Ventura CA for Gordon Rouse. Gordon is a Director of Engineering at KLA with 20+ years in software/systems engineering and program management. Prefers IC track (staff/principal) but will consider management. No firmware/Verilog/VHDL, no adtech.

**Companies & commute:**
- Northrop Grumman | Camarillo | ~26 min
- Amgen | Thousand Oaks | ~32 min
- Teledyne | Thousand Oaks/Camarillo | ~37 min

**Search sources:**
- Northrop Grumman: jobs.northropgrumman.com + LinkedIn
- Amgen: careers.amgen.com + LinkedIn
- Teledyne: flir.wd1.myworkdayjobs.com (Teledyne FLIR uses this ATS) + LinkedIn site:linkedin.com Teledyne Thousand Oaks OR Camarillo software engineer OR systems engineer OR manager

**Include:** Senior Software Engineer, Staff Engineer, Principal Engineer, Engineering Manager, Program Manager, Systems Engineer, Software Engineering Manager, Director-level roles.

**Exclude:** The Trade Desk, Verilog/VHDL/FPGA/firmware, embedded hardware bringup, jobs requiring active clearance (note if clearance may be needed later), and in-office/hybrid Agoura Hills roles. Agoura Hills may only appear if the role is truly remote.

**For each job:** Company | Title | City | Salary if available | Direct URL | 1-line fit assessment.

**Format:**
**Ventura Area Jobs — [DAY, DATE]**

**Northrop Grumman (Camarillo ~26 min)**
- ...

**Amgen (Thousand Oaks ~32 min)**
- ...

**Teledyne (Thousand Oaks/Camarillo ~37 min)**
- ...

If no new relevant postings, say "No new relevant postings today."
Keep under 500 words.
```

**Cron output location:** `/opt/data/cron/output/041c2bde2ba9/` (latest run: `YYYY-MM-DD_HH-MM-SS.md`). Parse the "## Response" section for the actual job report. The "## Prompt" section is the skill attachment and can be skipped.

**Wiki/page update requirement:** The active public career page is `/opt/data/hermes-pages/career-opportunities.html` (root public page, not the password-protected `/wiki/` copy). Daily job reports should update that static HTML file directly and deploy `/opt/data/hermes-pages` to Cloudflare Pages project `hermes-pages` with wrangler. Preserve public links `/career` and `/business-opportunities`. Keep the page high-contrast and readable. If the system Node is too old, use:
`export PATH=/opt/data/.nvm/bin:$PATH && /opt/data/.npm-global/bin/wrangler pages deploy /opt/data/hermes-pages --project-name hermes-pages --commit-dirty=true` or `npx -y -p node@22 -p wrangler wrangler pages deploy /opt/data/hermes-pages --project-name hermes-pages --commit-dirty=true`.
Live page: `https://hermes-pages-d55.pages.dev/career-opportunities`.

## Company Deep-Dive Workflow

1. Identify target companies (commute <30 min from home). Treat in-office/hybrid Agoura Hills as out-of-bounds despite being KLA-adjacent; Ventura ↔ Agoura Hills is too long for Gordon's daily commute target. Truly remote roles based there are acceptable.
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

## Landing Page

Gordon has a personal landing page built from his LinkedIn data. It's a dark GitHub-style resume page used for recruiter outreach and job applications.

**Live URL pattern:** `https://hermes-pages.rouse-gordon.workers.dev/<hash>-gordon-rouse-landing.html`

**Current version:** published with full LinkedIn work history (KLA 31 years, Syntex Labs, Santa Clara MS, UCSB BS).

**To update:** Edit `/opt/data/repo/landing.html`, then use `publish_html` to republish. Gordon can share the URL directly with recruiters. The landing page includes his full career timeline, education, target employers, and open-to section.

---

## Reference Data

See `references/ventura-employers.md` for compiled employer details, roles, salary ranges, ATS quirks, and search patterns from verified research.