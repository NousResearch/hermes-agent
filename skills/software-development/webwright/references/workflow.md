# Workflow

Detailed expansion of the six-step Webwright loop, adapted for terminal-native agents (Hermes / Claude Code). The original Microsoft Webwright loop relied on `webwright.tools.image_qa` for visual QA and `webwright.tools.self_reflection` for the final verdict. Both are replaced here by the agent's native abilities (`Read` on PNG files + reasoning against `plan.md`). No `OPENAI_API_KEY` is required.

## 1. Plan

Parse the task into critical points (CPs) and write `WORKSPACE_DIR/plan.md`:

```markdown
# Task
<verbatim task description>

# Critical Points
- [ ] CP1: <constraint / filter / sort / selection / required datum>
- [ ] CP2: ...
```

Rules for CPs:

- One CP per independently verifiable requirement.
- Numeric, date, quantity, and unit CPs must be exact. Wider buckets are failures unless the site genuinely offers no exacter control (record that exception in `plan.md`).
- Ranking CPs ("cheapest", "best-selling", "highest-rated", "most cited", "latest", …) must reference the site's actual sort/filter control, not the order results happen to come back in.
- If the task asks for a final datum (price, code, winner, fallo id, holding, quote), make it its own CP (e.g. `CP5: Record the displayed cheapest economy fare`).
- For browser-gated source spikes, include a CP for the access verdict itself (e.g. `CP1: Direct Firefox loads result list without 403/WAF interstitial`).

## 2. Explore

Goal: discover stable selectors, confirm every required filter control exists, and identify how to capture evidence for each CP.

- Run scratch Playwright scripts (see `playwright_patterns.md`) inside `WORKSPACE_DIR/scratch/`. Save scratch PNGs there — keep `final_runs/` clean.
- Print URL, title, and `aria_snapshot()` for the region of interest at every step.
- Use `Read` on saved PNGs to confirm UI state when ARIA evidence is ambiguous.
- If a filter looks unavailable, expand drawers / accordions / mobile filter panels and inspect again before concluding it doesn't exist.
- A search-box query never substitutes for a dedicated filter control — if the site has a "Year" facet, use it, don't append "year:2024" to the search.
- For WAF/TLS issues, try the same exploration step under Chromium *and* Firefox before concluding the site rejects all clients.

## 3. Author `final_script.py`

Create a fresh `final_runs/run_<id>_<tag>/` (use the next integer above any existing `run_*`, and a short `<tag>` like `direct`, `chromium`, `socks`) and place `final_script.py` inside it. Instrument per `playwright_patterns.md`:

- viewport 1280×1800, headless local Firefox by default, no `full_page=True`;
- one `final_execution_<step>_<action>.png` per CP;
- one `step <n> action: <reason and action>` log line per constraint-relevant interaction;
- the final datum printed into `final_script_log.txt` at the end (e.g. `FINAL_RESPONSE: <value>`);
- `results.json` with status, URLs, captured data, and timestamps.

Each screenshot should map to a CP from `plan.md` so verification is mechanical.

## 4. Execute

Run the script once. Compare engines / proxy modes when relevant:

```bash
python final_script.py --tag direct
python final_script.py --engine chromium --tag chromium
JURISNET_BROWSER_PROXY=socks5://127.0.0.1:18080 python final_script.py --tag socks
```

If a run crashes mid-flight and produces partial screenshots that don't match the fixed flow, delete the partial run folder before re-executing — mixed evidence inside one run folder breaks the verification step.

## 5. Self-verify (replaces `self_reflection`)

For every CP in `plan.md`:

1. Identify the screenshot(s) and/or log line that provide evidence.
2. `Read` each cited PNG.
3. Confirm the evidence is **unambiguous**:
   - Filter chip / selected state visibly applied (not hidden behind a closed drawer);
   - Numeric / date values match exactly (not broadened or "close enough");
   - Sort applied via the site's control (not implied by result order);
   - Required submit / search / apply action visibly taken;
   - Final datum legibly displayed.
4. Tick the CP only when the evidence is concrete. Be harsh on partial, occluded, or ambiguous states.

If any CP fails, diagnose the *specific* issue — wrong filter value, missing control, hidden chip, broadened range, missing confirmation, missing screenshot, etc. Fix `final_script.py`, run it again inside `final_runs/run_<id+1>_<tag>/`, and re-verify against `plan.md`.

Empty result sets are acceptable when the correct filters were demonstrably applied and the screenshot shows the empty state with the filter chips visible.

## 6. Done

Stop only when **all** of the following are true:

1. `plan.md` exists with every CP enumerated as a checklist item.
2. `final_runs/run_<id>_<tag>/final_script.py` ran cleanly from scratch and produced `final_script_log.txt`, `results.json`, and all CP screenshots.
3. Every CP is checked off with a cited screenshot and/or log line, and the agent has actually `Read` each cited PNG.
4. The final datum (if the task asked for one) is reported to the user verbatim and is also present in `final_script_log.txt` / `results.json`.
5. `ls -R final_runs/run_<id>_<tag>` and `cat final_runs/run_<id>_<tag>/final_script_log.txt` show the expected artifacts.
6. No secrets, cookies, local IPs, or proxy credentials persisted.

If any of those is false, do not declare done — diagnose, fix, and re-run in a new `run_<id+1>_<tag>/`.
