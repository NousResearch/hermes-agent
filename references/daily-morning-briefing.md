# Daily Morning Briefing Format

## Data Sources
- signal_monitor SQLite database (pricing diffs, RSS feeds, page checks)
- Hermes agent outputs/ directory
- Built LMS Obsidian vault (wiki + raw)
- Linear for product intent
- GitHub for implementation evidence

## Briefing Sections
1. **Date & Context**
2. **Competitive Signals** — pricing changes, product launches, M&A, hiring
3. **Product & Release Ops** — merged PRs, shipped features, Linear status
4. **GTM / Content** — drafts, approvals, posting status, content mix
5. **Discovery / Outreach** — prospect pipeline, conversations, signals
6. **Knowledge Base Health** — lint issues, stale articles, orphaned notes
7. **Next Actions** — per-lane priorities

## Cronjob Config
Runs daily. Output to `outputs/daily-morning-briefing/YYYY-MM-DD.md`.

## Known Friction Points
- Signal monitor daemon may have stale data if Cloudflare credentials are invalid
- prospect_discovery_cf.py may hang during Cloudflare-dependent lanes
- Null-safety bug in generate_weekly_report() when signal_summary is NULL