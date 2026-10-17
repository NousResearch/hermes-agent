---
name: boss-zhipin-scraper
description: Search and analyze BOSS直聘 jobs with plaintext salaries.
version: 2.2.0
author: eatmoreduck, Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [jobs, career, scraper, chrome, cdp, zhipin, boss直聘]
    category: productivity
    related_skills: [blocked-page-recovery, product-price-monitor]
---

# BOSS直聘 Scraper Skill

Collect public job listings from BOSS直聘 (zhipin.com) via the Chrome
DevTools Protocol attached to a dedicated, isolated Chrome profile, and save
them as structured JSON/CSV with plaintext salaries such as `30-60K·15薪`,
optional job-description details, and a market digest. This is a research
tool for a personal job search — passive, rate-limited capture only, never a
mass crawler, and it never bypasses captchas or anti-bot checks.

Upstream project: https://github.com/eatmoreduck/boss-zhipin-scraper

## When to Use

- The user wants to search BOSS直聘 jobs by keyword, city, and filters
  (salary range, experience, degree, company scale) and keep structured
  results.
- Comparing salary levels, skill demands, or company mix across keywords or
  cities to inform a job search or offer negotiation.
- Summarizing a finished scrape into a digest the agent can present or build
  on (`scripts/job_summary.py`).
- Not for: bulk harvesting, resume delivery, or job boards other than
  zhipin.com.

## Prerequisites

- macOS or Linux. The Chrome-process matching relies on POSIX `ps` and
  `os.kill`; Windows is not covered.
- Google Chrome or Chromium installed.
- Python 3.10+ with `requests` and `websocket-client`
  (`pip install requests websocket-client`).
- A BOSS直聘 account: the first `--setup-chrome` run opens a dedicated
  Chrome window for the user to log in by hand. The session persists in
  `~/.boss-zhipin-scraper/chrome-profile` across restarts.
- Run everything through the `terminal` tool; inspect outputs with
  `read_file` and the JSON files under `~/.boss-zhipin-scraper/job-result/`.

## How to Run

Install once (user side; the skill then ships its own scripts):

```text
hermes skills install official/productivity/boss-zhipin-scraper
```

Paths below are relative to this skill's directory. The scripts also work
standalone from the upstream repository. Minimal flow via `terminal`:

```bash
python3 scripts/boss_cdp_raw.py --check                     # deps + CDP + login probe
python3 scripts/boss_cdp_raw.py --setup-chrome              # only if the CDP check fails
python3 scripts/boss_cdp_raw.py --keyword "AI Agent" --city 上海 --pages 3 --no-detail
python3 scripts/job_summary.py --top 15                     # digest of the latest scrape
```

The city table ships in `data/city_codes.json` (374 cities, offline); the
script also syncs fresher codes from BOSS直聘 at runtime.

## Quick Reference

| Flag | Meaning |
|---|---|
| `--keyword`, `--city` | Search keyword and city (Chinese name or 9-digit code; default `AI Agent` / `上海`). |
| `--pages N` | Pages to capture, hard cap 10. |
| `--no-detail` / `--detail` | Detail scraping is on by default; disable for list-only runs. |
| `--max-details N` | Cap the number of detail pages fetched. |
| `--analysis` | Print a salary/skill analysis report after scraping. |
| `--format csv` | Also write CSV (UTF-8 BOM) beside the JSON outputs. |
| `--output`, `--detail-output` | Override output paths; defaults live under `~/.boss-zhipin-scraper/job-result/`. |
| `--merge FILE` | Merge and dedupe with a previous JSON result (by `job_id`). |
| `--input FILE` | Analyze an existing JSON instead of scraping. |
| `--check` / `--smoke-test` | Environment diagnosis / one real search round-trip, no files written. |
| `--setup-chrome` / `--stop-chrome` / `--close-chrome` | Start, stop, or auto-close the dedicated CDP Chrome. |
| `--browser edge` / `--setup-edge` / `--stop-edge` | Same flow through Microsoft Edge instead of Chrome (`--browser edge` is a `--setup-chrome` compatibility form). |
| `--reset-chrome-profile` | Rebuild the dedicated profile (wipes its login). |
| `--list-cities [kw]` | Print supported cities from the code table. |

Filter codes: `--salary` 402=3K以下 403=3-5K 404=5-10K 405=10-20K 406=20-50K
407=50K以上; `--experience` 102=应届生 104=1-3年 105=3-5年 106=5-10年;
`--degree` 203=本科 204=硕士 205=博士; `--scale` 303=100-499人 305=1000-9999人.
City codes: 全国 100010000, 北京 101010100, 上海 101020100, 广州 101280100,
深圳 101280600, 杭州 101210100, 成都 101270100.

## Procedure

1. Run `--check`. It verifies Python dependencies, CDP reachability, and the
   login state, in that order. All green → step 3.
2. If the CDP check fails, run `--setup-chrome`. It creates or reuses the
   isolated profile `~/.boss-zhipin-scraper/chrome-profile` (never the main
   Chrome profile), starts Chrome with `--remote-debugging-port=9222`, opens
   the login page, and waits until the search API returns plaintext
   salaries. Tell the user to log in in that window; then re-run `--check`.
   Copying login state from the main Chrome only happens when the user
   explicitly passes `--copy-login-state`.
3. Capture the list: `--keyword ... --city ... --pages N [--no-detail]`.
   The script navigates the real search page once, then passively captures
   the joblist responses the page itself emits while it scrolls — nothing is
   injected. Results are written after each page, so an interrupted run
   keeps partial data.
4. For details, keep `--detail` (default) and consider `--max-details` —
   each detail page costs 10–25 s. `--analysis` prints a report combining
   list and detail data.
5. Summarize: `scripts/job_summary.py --top 15` reads the latest
   `boss_jobs_*.json` and `boss_details_*.json` under the result directory;
   `--input`/`--details` pin exact files. It reads no local resume files.
6. Merge repeat runs over the same query with `--merge earlier.json` to
   accumulate a deduplicated corpus.
7. Clean up with `--stop-chrome`, or pass `--close-chrome` to close the
   dedicated Chrome after a successful scrape. Matching is strict on the
   profile's `--user-data-dir`, so the user's main Chrome is never touched.

## Pitfalls

- Font obfuscation: salaries rendered on the page use a scrambled font.
  Only the `salaryDesc` field from the captured API traffic is plaintext and
  trustworthy. `--allow-dom-fallback` is off by default for this reason;
  DOM-extracted salaries must not be reported as fact.
- Blank salaries or HTTP 401 usually mean the session expired — re-run
  `--setup-chrome` and have the user log in again. Do not treat this as a
  parsing bug.
- Risk control: aggressive repeated navigation triggers the
  `_security_check` interstitial or API codes 31/37 (rate limited). Ask the
  user to clear the check manually in the dedicated window; never automate
  or bypass it. Built-in pacing (12–22 s between pages, 10–25 s per detail,
  caps of 10 pages / 500 API requests per run) exists to avoid this — do not
  loop tighter than the script already does.
- SPA quirks: the search page has no pagination controls and ignores
  `&page=N` URLs; capture works by driving the page's own infinite scroll in
  a background tab with focus emulation. Extra manual navigation is what
  triggers the security check.
- An unrecognized city name aborts the run by design (it would otherwise
  look like "0 jobs"). Use `--list-cities` or pass a 9-digit city code.

## Verification

- Before scraping, `--check` reports dependencies OK, CDP OK, and login OK.
- The output JSON has `total > 0`, and each job carries a non-empty `salary`
  with `salary_source: "api"`.
- `--smoke-test` completes one real search round-trip through the live
  Chrome and writes no files.
- `scripts/job_summary.py --top 15` prints a digest naming the files it
  read.
- After `--stop-chrome`, a follow-up `--check` reports CDP unreachable —
  expected, not a failure.
