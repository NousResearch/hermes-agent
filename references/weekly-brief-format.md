# Weekly Brief Output Format

prospect_discovery_cf.py output format:
Output: ~/built-lms-agent-system/trackers/weekly_brief_YYYYMMDD.txt

Sections:
1. Top new prospects (top 10, status='new', sorted by score DESC)
2. Recent competitor changes (last 7 days)
3. Lane health (monitor_runs, last 20 entries)
4. Issues encountered

## Known Issues
- Cloudflare-dependent lanes hang if CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are not in ~/.env
- generate_weekly_report() null-safety bug on signal_summary