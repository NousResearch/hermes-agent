# Pipeline Diagnostics

Operational runbook for diagnosing and fixing biotech screener pipeline outages, stale Hermes agents, CI failures, and cron issues in the Warrenpoobear/biotech-screener repo.

---

## Diagnostic Approach

When the user reports pipeline issues, follow this sequence:

1. **Confirm symptoms from email signals** — search for Herald, Bellringer, Intraday Mover, PDUFA, and Morning Briefing emails to establish what's delivering and what's dark
2. **Check GitHub CI status** — look at recent commits, open PRs, and CI check results on main
3. **Cross-reference the stale agent checklist** — compare against known agent inventory
4. **Build a priority-ordered fix list** — hermes-mail bridge first (everything depends on it), then highest-value signals

## Agent Inventory (30 total, 27 active)

### Lane A: Deterministic (5 agents)
- `aact_trial_ingest` — CT.gov clinical trial data
- `ctgov_poller` — CT.gov polling
- `earnings_calendar_sync` — earnings date sync
- `herald` — daily press release digest (HIGHEST PRIORITY when dark)
- `universe_maintenance` — ticker universe upkeep

### Lane B: Monitoring + Escalation (18 agents)
- `bellringer` — biotech earnings preview + results
- `biotech_news_digest`, `calibration`, `catalyst_delta`, `crt_resolution_watcher`
- `data_auditor`, `event_analyst`, `grok_biotech_watch`, `ic_health_monitor`
- `intraday_mover_watch`, `options_watch`, `policy_shadow_watch`, `postmortem`
- `price_action_watch`, `shadow_monitor`, and others

### Lane C: Manual Engineering (7 agents)
- `fleet_steward` — fleet health monitoring
- `ops`, `ops_supervisor`, `production_qa`, `qa`, `review_queue_steward`, `sentinel`

### Deprecated/Suppressed (3)
- `company_news_ingest` (retired, replaced by herald)
- `bioshort_watch` (suppressed)
- `shadow_watch` (placeholder)

## Triage Priority Order

When multiple agents are down, fix in this order:

1. **hermes-mail bridge** — if bridge is down, no agent can deliver email. Fix SMTP first.
2. **Herald Digest** — highest-value daily signal. Dark Herald = blind on press releases.
3. **Bellringer** — earnings preview/results. Check preview vs. results separately (different cron jobs).
4. **fleet_steward** — restores health monitoring for everything else.
5. **Intraday Mover Watch** — real-time price alerts during market hours.
6. **grok_biotech_watch** — depends on XAI API key, often the longest-running outage.
7. **Evening forward-shadow** — watchdog pattern, lower priority.
8. **postmortem memory-write** — cosmetic mtime bug, lowest priority.

## Common Failure Patterns

### Herald Dark (No Digest Email)

**Symptoms:** No Herald Digest email in inbox for 1+ days. Morning Briefings may still work (those are Town routines, not Hermes).

**Diagnostic steps:**
```bash
# Check cron entry
crontab -l | grep herald

# Check logs
ls -la ~/.hermes/logs/ | grep herald
tail -50 logs/daily_production_*.log | grep -i herald

# Manual test
python3 run_agent_direct.py --agent herald
# If preflight blocks:
python3 run_agent_direct.py --agent herald --skip-preflight
```

**Common root causes:**
- **Timeout budget exceeded** — sequential IR fetch loop over 341 tickers takes 23+ min in sleeps alone. Fix: PR #266 parallelized fetcher (ThreadPoolExecutor, 12 workers, ~3-5 min). If not merged, merge it.
- **hermes-mail bridge down** — Herald runs but email never arrives. Test bridge separately.
- **Cron entry missing** — WSL2 restart can lose crontab entries.
- **Preflight gate blocking** — dirty git state, missing snapshot, or governance hold.

### Bellringer Results Not Delivering

**Symptoms:** Bellringer preview emails arrive (morning, pre-market), but results emails (post-market with EPS surprise, price moves) do not.

**Key distinction:** Previews and results are different cron jobs. Previews working + results dark = results job failing silently.

```bash
# Check results cron (should fire ~4:30-5:00 PM ET)
crontab -l | grep -i bellringer

# Check error logs
cat logs/bellringer_results_*.log | tail -50

# Manual test with known earnings date
python3 run_agent_direct.py --agent bellringer_results --date 2026-05-14
```

### CI Failures

**Common CI failure patterns in this repo:**

| Failure | Root Cause | Fix |
|---|---|---|
| pytest version CVE | Security advisory on pytest | Upgrade in requirements.txt (e.g., 8.3.4 to 9.0.3) |
| Agent registry enum validation | Invalid status values (e.g., `suppressed`, `retired`) | Change to valid enum (`deprecated`) in AGENT_REGISTRY.json |
| Ruleset drift test | New governance source not in allowed list | Add source to `test_decision_ruleset.py` allowed set |
| Critical code errors (F821/F811) | Undefined variables, unused imports | Fix in source, run flake8 locally |
| Universe loading: 1 ticker | Stale `ipo_dates.json` with old `last_price_date` | Update all tickers' `last_price_date` to latest trading day |
| Intraday mover NO_DATA | Poll runs before production snapshot exists | Shift first poll to after production run completes (~10:30 ET) |
| dep-audit failures | Outdated dependencies with known CVEs | Merge dependabot PRs |
| type-check failures | mypy version drift | Bump mypy, fix new type errors |

### hermes-mail Bridge Down

**Symptoms:** Agents run successfully (logs show completion) but no email arrives.

```bash
# Smoke test
python3 hermes_mail_smoke_test.py

# Check SMTP credentials
cat ~/.hermes/.env | grep SMTP
# or
cat .env | grep -i mail

# If auth error: regenerate Gmail app password
# https://myaccount.google.com/apppasswords
# Update in .env
```

**Success criteria:** Smoke test email arrives in djschulz@gmail.com inbox.

### grok_biotech_watch Dark (XAI API Key)

```bash
# Check API key exists
grep XAI_API_KEY ~/.hermes/.env

# If missing/expired: regenerate at console.x.ai
# Add to .env: XAI_API_KEY=xai-xxxxx

# Test
python3 run_agent_direct.py --agent grok_biotech_watch
# 401/403 = key invalid, regenerate
# Rate limit = check xAI account billing
```

### WSL2 Cron Issues

WSL2 cron is fragile — entries can be lost on restart, and sleep/wake cycles can cause missed runs.

```bash
# Verify cron is running
sudo service cron status
# If stopped: sudo service cron start

# Verify all entries
crontab -l
# Should include entries for: herald, fleet_steward, hermes_mail, bellringer, intraday_mover, evening forward-shadow

# PR #269 (open) adds WSL2-sleep-resilient catchup for evidence builds
```

## What Town AI Can Do vs. What Requires Terminal

### Town AI can:
- Search emails to confirm which pipelines are delivering and which are dark
- Check GitHub CI status, recent commits, open PRs
- Read commit diffs to understand what was fixed
- Create GitHub issues tracking remaining fixes
- Send runbook emails to work address
- Update the stale agent diagnostic checklist

### Terminal required (Town AI cannot):
- Run `crontab -l` or modify cron entries
- Run `python3 run_agent_direct.py` or any local scripts
- Check `~/.hermes/.env` for API keys
- Read local log files
- Run hermes-mail smoke tests
- Merge PRs that have failing CI on their branches

When the user asks to "fix the pipeline" and fixes require terminal access, build a priority-ordered runbook and send it to dschulz@wakerobin.co for morning execution.

## Email Signal Verification Queries

Use these Gmail searches to confirm pipeline status:

```
# Herald Digest (should be daily on trading days)
subject:"Herald" after:2026/05/12

# Bellringer previews (morning, pre-market)
from:djschulz@gmail.com subject:"Bellringer" "biotech earnings" after:2026/05/12

# Bellringer results (post-market, EPS surprise data)
from:djschulz@gmail.com subject:"Bellringer" "results" after:2026/05/12

# Intraday Mover Alerts
subject:"Intraday Mover" OR subject:"HIGH alert" after:2026/05/12

# Morning Briefings (Town routine, not Hermes)
subject:"Morning Briefing" after:2026/05/12

# Catalyst updates (Town routine)
subject:"Catalyst Update" after:2026/05/12
```

## Reference Docs

- [Hermes Stale Agent Diagnostic Checklist](https://www.town.com/content/document/nx7aacpkbvyxj32fh79hwn4kf586vkd1)
- [Optimal Agent Setup](https://www.town.com/content/document/nx7bhdet5bfd6727qgyxejd6fh870zek)
- [CCFT-Aware Routing Policy](https://www.town.com/content/file/sh75yt8qsbdtep7ja2f8r6cpfs86vwzt)
- GitHub: [Warrenpoobear/biotech-screener](https://github.com/Warrenpoobear/biotech-screener)
