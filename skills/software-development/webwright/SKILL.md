---
name: webwright
description: Use when automating browser tasks that should leave behind a reusable, auditable Playwright program rather than a one-shot browser session — search/filter/form-fill flows, probing browser-gated or WAF-sensitive sites, dynamic extraction, regression harnesses for JurisNet legal-source adapters. Drives a local headless Playwright (Firefox preferred), writes `final_script.py` + `final_runs/run_<id>_<tag>/` with screenshots, action log, and `results.json`, and self-verifies against a `plan.md` critical-points checklist.
version: 1.0.0
author: Hermes Agent (adapted from Microsoft Webwright)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [browser-automation, playwright, webwright, scraping, qa, browser-gated, jurisnet]
    homepage: https://github.com/microsoft/Webwright
    related_skills: [systematic-debugging, spike, requesting-code-review]
---

# Webwright — Terminal-Native Browser Automation

## Overview

Webwright is the preferred Hermes workflow for browser tasks that should leave behind a reusable, auditable program instead of a fragile manual click path. The pattern is: write Playwright code, run it from the terminal, save screenshots / action log / JSON results into a numbered run folder, inspect failures, iterate, then preserve a `final_script.py` that can be rerun from a clean workspace.

This skill is adapted from Microsoft Webwright (https://github.com/microsoft/Webwright). The upstream loop relied on `webwright.tools.image_qa` / `self_reflection` for OpenAI-backed visual QA; here those are replaced by the agent's own `Read` on PNG files plus reasoning against `plan.md`. No `OPENAI_API_KEY` or other external model API is required — the workflow is terminal + Playwright + this agent.

Treat Webwright as a **workflow / tooling skill**, not as a separate runtime service. Hermes can drive it directly with `Bash` + file tools, and can delegate the heavier scripting loop to Claude Code when the task is non-trivial (see `Delegating to Claude Code`).

## When to Use

Use Webwright when the user asks to:

- Automate a live website task: search, filter, form-fill, checkout-like flow, upload/download, compare results.
- Probe browser-gated or WAF-sensitive sources where plain HTTP is insufficient (typical for JurisNet legal-source adapter spikes — SAIJ, CSJN, JUBA, provincial portals).
- Extract data from dynamic pages that require JS, scrolling, waiting, or human-like navigation.
- Produce durable evidence: screenshots, action log, `results.json`, and a rerunnable script.
- Turn a successful one-shot flow into a reusable parameterized CLI tool (see `references/cli_tool_mode.md`).
- Build a regression harness for a flaky UI path or a site whose markup drifts.

Do **not** use it for:

- Static HTTP fetches where `curl` / `httpx` / an official API works. Use the API.
- Tasks that would require bypassing access controls or violating ToS. Stay on public/authorized surfaces, keep rate low, and surface the legal/ToS concern back to Hermes if in doubt.
- Quick visual inspection where a single browser snapshot is enough and no reusable artifact is needed.

## Prerequisites

```bash
python3 -m pip show playwright >/dev/null 2>&1 || python3 -m pip install playwright
python3 -m playwright install firefox chromium
```

If the project provides a venv or `uv`, prefer the project runtime:

```bash
uv run python -c "from playwright.async_api import async_playwright; print('playwright ok')"
uv run playwright install firefox chromium
```

Firefox is the default engine — some sites (cars.com, Akamai-protected portals, parts of SAIJ/JUBA) reject Playwright Chromium with `ERR_HTTP2_PROTOCOL_ERROR` due to TLS/H2 fingerprinting but load cleanly under Firefox. Keep Chromium installed for CDP / compatibility comparisons.

## Workspace Contract

Pick a bounded workspace per task. Typical layouts:

```text
spikes/<task-id>/webwright/
outputs/<task-id>/webwright/
```

Layout inside the workspace:

```text
plan.md
final_script.py
final_runs/run_<id>_<tag>/
  final_script.py            # snapshot of the script that produced this run
  final_script_log.txt
  results.json
  screenshots/final_execution_<step>_<action>.png
scratch/                      # exploration scripts and PNGs, separate from final_runs/
```

Rules:

- `plan.md` lists every critical point that must be proven by a screenshot or a log line.
- `final_script.py` must be rerunnable from scratch — no hidden state, no shell-side mutations required.
- Every clean execution creates a **new** `final_runs/run_<id>_<tag>/` (next integer above any existing `run_*`).
- Each constraint-relevant interaction writes `step <n> action: ...` to `final_script_log.txt`. The first line after reset is `step 0 params: ...` when running in CLI tool mode.
- One screenshot per critical point, named `final_execution_<step>_<action>.png`. **Never** use `full_page=True` — for exploration, debugging, or final runs alike.
- Viewport is `{"width": 1280, "height": 1800}`. Don't change it without a reason recorded in `plan.md`.
- Persist concise snippets and structured facts, not giant raw HTML bodies.
- **Never** persist local IPs, proxy credentials, cookies, tokens, session storage, or anything that could leak the runner's identity / network.

## Workflow

1. **Plan critical points.** Write `plan.md`:

   ```markdown
   # Task
   <verbatim task description>

   # Critical Points
   - [ ] CP1: <exact constraint / target URL / expected result>
   - [ ] CP2: <filter, sort, selection, or required datum that must be proven>
   ```

   One CP per independently verifiable requirement. Numeric/date/quantity/unit CPs must be exact. Ranking CPs ("cheapest", "most cited", "latest") must reference the site's actual sort/filter control, not your own ordering of the results. If the task asks for a final datum, make it its own CP.

2. **Explore with scratch scripts.** Short Playwright heredocs (see `references/playwright_patterns.md`) to inspect titles, URLs, response status, visible labels, `aria_snapshot()`, selectors, network events, and screenshots. Save scratch PNGs under `scratch/`, not `final_runs/`.

3. **Author `final_script.py`.** Deterministic, parameterized where useful. Include:
   - `argparse` flags for task inputs, proxy, headless mode, engine, output tag.
   - A run-dir allocator (`allocate_run_dir`) so each invocation goes to a fresh `final_runs/run_<id>_<tag>/`.
   - Structured `log(step, msg)` writing to `final_script_log.txt`.
   - `results.json` output with status, URLs, captured data, timestamps.
   - A screenshot at each critical point, mapped 1-to-1 to a CP in `plan.md`.

   The skeleton is in this file (`Playwright Skeleton`); detailed patterns (role/name selectors, interactive form filling, ARIA snapshots, paired-field modals) are in `references/playwright_patterns.md`.

4. **Run cleanly.** Execute once direct, then with alternate engine or proxy if relevant:

   ```bash
   python final_script.py --tag direct
   python final_script.py --engine chromium --tag chromium
   JURISNET_BROWSER_PROXY=socks5://127.0.0.1:18080 python final_script.py --tag socks
   ```

5. **Self-verify.** For each CP in `plan.md`: identify the screenshot(s) and/or log line that prove it, `Read` each cited PNG, confirm the evidence is unambiguous (filter chip visible, value exact, sort applied via the site control, final datum legible). Be harsh with partial, occluded, or ambiguous states. If a CP fails, diagnose the *specific* issue, fix `final_script.py`, re-run inside `run_<id+1>_<tag>/`, and re-verify. Empty result sets are acceptable when the correct filters were demonstrably applied.

6. **Report and preserve.** Report the final datum verbatim and the artifact paths. Commit the script + results only when the repo expects spike evidence (e.g. `spikes/<task-id>/`); otherwise keep generated outputs local and `.gitignore`'d.

Full step-by-step guidance — including CP rules, exploration heuristics, and the completion checklist — lives in `references/workflow.md`. Load it for non-trivial tasks; the summary above is the minimum.

## Playwright Skeleton

```python
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright

VIEWPORT = {"width": 1280, "height": 1800}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def allocate_run_dir(workspace: Path, tag: str) -> Path:
    runs = workspace / "final_runs"
    runs.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in runs.glob("run_*"):
        try:
            nums.append(int(p.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    run_id = max(nums, default=0) + 1
    out = runs / f"run_{run_id}_{tag}"
    (out / "screenshots").mkdir(parents=True, exist_ok=True)
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--tag", default="direct")
    parser.add_argument("--proxy", default=os.getenv("JURISNET_BROWSER_PROXY"))
    parser.add_argument("--engine", choices=["firefox", "chromium"], default="firefox")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    run_dir = allocate_run_dir(workspace, args.tag)
    log_path = run_dir / "final_script_log.txt"
    log_path.write_text("")

    def log(step: int, msg: str) -> None:
        line = f"step {step} action: {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async with async_playwright() as pw:
        browser_type = getattr(pw, args.engine)
        launch_kwargs = {"headless": True}
        if args.proxy:
            launch_kwargs["proxy"] = {"server": args.proxy}
        browser = await browser_type.launch(**launch_kwargs)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()

        log(1, f"goto {args.url}")
        response = await page.goto(args.url, wait_until="domcontentloaded", timeout=45_000)
        await page.screenshot(path=str(run_dir / "screenshots" / "final_execution_01_loaded.png"))

        result = {
            "url": page.url,
            "status": response.status if response else None,
            "title": await page.title(),
            "captured_at": now(),
        }
        (run_dir / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        log(2, f"result: {json.dumps(result, ensure_ascii=False)}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
```

For role/name selectors, ARIA snapshots, interactive form filling, paired-field modal patterns, and final-script instrumentation, see `references/playwright_patterns.md`.

## Browser-Gated Source Pattern (JurisNet)

For legal-source probes and adapter spikes (SAIJ, CSJN, JUBA, provincial portals, FalloBot data sources):

- **Try the official API / static endpoint first.** If it works, build the production adapter on the API and skip the browser.
- **Use Webwright as a spike or regression harness** when HTTP returns 403/WAF/JS-only pages, or when a site exposes data only through interactive controls.
- Keep rate low: one browser, one page, bounded target list. Do not loop scrape until legality / robots / ToS are clear; surface concerns to Hermes.
- Comparison ladder: direct Firefox → direct Chromium → Firefox with `JURISNET_BROWSER_PROXY` (SOCKS/HTTP proxy) → Chromium with proxy. Tag each run accordingly so screenshots and logs are comparable.
- Preserve evidence (`results.json` + screenshots), not full HTML dumps.
- If browser probes still hit WAF/403 while an official open-data API works, design the production adapter around the API and keep the Webwright spike only as a diagnostic harness in the repo.

## CLI Tool Mode

Default Webwright runs produce a one-shot `final_script.py` solving the task for the literal values the user provided. **CLI tool mode** instead produces a reusable parameterized tool: one function with a Google-style `Args:` docstring + an `argparse` wrapper whose flags default to the concrete task values, so the user can re-run it later with different inputs.

Trigger CLI tool mode when the user says "make it reusable", "parameterize", "turn this into a CLI", "I want to call this again with different X", or similar. Defaults must reproduce the original task exactly — `python final_script.py` with no arguments equals the original run.

Full contract (the `# Parameters` table, reusable-function shape, `step 0 params:` line, import-safety smoke test, completion gate) is in `references/cli_tool_mode.md`. Load it whenever the user asks for a reusable tool.

## Delegating to Claude Code

For non-trivial automation (long flows, fragile selectors, multi-engine comparison) delegate the script-authoring loop to Claude Code and tell it to follow this skill / workspace contract. Hermes retains final verification.

```bash
HOME=/root/.hermes/profiles/default/home claude -p \
  "Use the Webwright skill at skills/software-development/webwright/. \
   Build a bounded Playwright script for <task>. \
   Workspace: <workspace>. Produce plan.md, final_script.py, \
   final_runs/run_<id>_<tag>/ with screenshots, final_script_log.txt, \
   results.json. Verify every CP. Do not persist secrets/cookies/IPs." \
  --allowedTools 'Read,Write,Edit,MultiEdit,Bash(python3 *),Bash(uv *),Bash(playwright *),Bash(git *)' \
  --permission-mode acceptEdits \
  --max-turns 40
```

Hermes is then responsible for:

- Reading the final `plan.md`, `final_script.py`, `results.json`, and key screenshots.
- Confirming every CP is ticked with cited evidence.
- Running `git diff --stat` if the spike committed anything and confirming bounds.
- Reporting the final datum + artifact paths back to the user.

Do not skip these checks — the delegated agent only describes what it *intended* to do, not necessarily what landed on disk.

## Hard Rules

- One bash command per exploration step; observe its output before issuing the next.
- Use stable selectors (role + accessible name, `aria-label`) and current-run evidence — never guess UI state.
- If a site exposes a dedicated control for a requirement, **use that control**. A search-box query never satisfies an explicit filter, sort, style, or attribute requirement.
- Ranking language must be grounded in the site's actual sort/filter, not in your own ordering of the results.
- Numeric / date / quantity / unit constraints are **exact**. Wider buckets are failures unless the site offers no exacter control (record that in `plan.md`).
- If a selected state becomes hidden after a drawer / modal / dropdown closes, reopen it or capture a visible chip/summary before treating the state as verified.
- For blocker claims (Access Denied, 403, "control unavailable"), require repeated evidence from the actual UI — not a single timeout.
- Never hardcode proxy endpoints or credentials. Accept `--proxy` and `JURISNET_BROWSER_PROXY`.
- Do not install extra packages unless the project's runtime is missing one. `playwright`, `httpx`, `pydantic` are typically already installed.
- Once `final_script.py` exists, prefer incremental `Edit` over rewriting the whole file.

## Common Pitfalls

1. **Reporting success from live browser state alone.** The deliverable is the reusable script + cited evidence, not a manual click path that happened to work once.
2. **No critical-point plan.** Without `plan.md`, verification becomes vibes-based. Write CPs *before* coding.
3. **Saving huge HTML dumps, cookies, or secrets.** Persist `results.json` + structured snippets only. Never persist proxy creds or local IPs.
4. **One engine only.** For WAF / TLS / H2-fingerprint issues, compare Firefox and Chromium before concluding the site is unreachable.
5. **Deep-link URLs as the primary path.** Sites silently drop unparseable query params. Drive the on-page form interactively; treat deep links as an opportunistic shortcut and verify form state afterwards.
6. **Declaring filters applied from URL or search text alone.** Verify with visible chips, selected controls, or a result-summary banner — not the URL bar.
7. **`full_page=True` screenshots.** Banned everywhere. Use the 1280×1800 viewport.
8. **Committing generated bulk by default.** Commit spike evidence only when the repo expects it (`spikes/<task-id>/`); otherwise add `final_runs/`, `scratch/`, `outputs/` to `.gitignore`.
9. **Reusing a run folder after a partial / crashed execution.** Either delete the partial PNGs or allocate `run_<id+1>_<tag>/`. Mixed evidence inside one run folder breaks verification.
10. **Skipping the import-safety check in CLI tool mode.** A `final_script.py` that launches a browser at import time is not a reusable tool. See `references/cli_tool_mode.md`.

## Verification Checklist

- [ ] `plan.md` exists and lists every CP as a checklist item.
- [ ] `final_script.py` runs from a clean workspace with no manual setup.
- [ ] A fresh `final_runs/run_<id>_<tag>/` was produced for the reported run.
- [ ] `results.json` captures status / URLs / captured data / timestamps.
- [ ] `final_script_log.txt` has a `step <n> action: ...` line for every constraint-relevant interaction (and `step 0 params: ...` in CLI tool mode).
- [ ] One screenshot per CP under `screenshots/`, named `final_execution_<step>_<action>.png`. No `full_page=True`. Viewport 1280×1800.
- [ ] Every CP in `plan.md` is ticked with a cited screenshot and/or log line; the agent has `Read` each cited PNG.
- [ ] The final datum (if the task asked for one) is reported verbatim and is also present in `final_script_log.txt` / `results.json`.
- [ ] No secrets, cookies, session storage, local IPs, or proxy credentials are persisted.
- [ ] If committed, `git diff --stat` is bounded and matches the spike's stated scope.
- [ ] If delegated to Claude Code, Hermes has independently verified plan.md, script, results, and key screenshots before reporting success.
