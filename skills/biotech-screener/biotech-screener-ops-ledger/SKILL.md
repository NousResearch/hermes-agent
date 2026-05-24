---
name: biotech-screener-ops-ledger
description: >
  Umbrella skill for the biotech screener ops/knowledge layer — held-spec ledger,
  first-fire validator, contradiction detector, daily/weekly briefs, DE schema gate
  fixes and data-integrity checks, in-code operator alert patterns (Telegram +
  Town/email), and TrapOps monitoring. Covers the four-layer
  capture→normalize→reason→return model, SNAPSHOT_COLUMNS drift fixes, rank-check
  logic, catalyst-date staleness, two-channel alert design, and Spec 090 Town delivery.
when_to_use: >
  User asks about spec status, held branches, first-fire validation, ops ledger,
  knowledge layer, contradiction detection, or operator brief generation for the
  biotech screener. Also use when extending or re-seeding held_spec_ledger or
  first_fire_ledger after a new spec install.
---

# Biotech Screener — Ops Ledger / Knowledge Layer

Spec 089. Repo-native knowledge layer that answers: what is current state, what changed,
what is held/blocked, what contradictions exist, what is the next allowed action.

NOT a source of production truth. Production truth is in code, cron, AGENT_REGISTRY.json,
receipts, and deterministic artifacts under output/ and data/.

---

## Repo layout

```
artifacts/ops/knowledge_layer/
  latest_state.json      -- machine-readable normalized snapshot
  latest_state.md        -- human-readable summary
  README.md

artifacts/ops/held_spec_ledger/
  latest.json            -- structured held item records
  latest.md              -- human-readable (may be manually authored or generated)
  YYYY-MM-DD.md          -- dated copies

artifacts/ops/first_fire_ledger/
  latest.json            -- first-fire job records with eval status
  latest.md              -- human-readable with pass/fail criteria
  YYYY-MM-DD.json        -- dated copies

artifacts/ops/contradiction_ledger/
  latest.md              -- contradiction scan output
  YYYY-MM-DD.md          -- dated copies

artifacts/ops/operator_brief/
  daily/YYYY-MM-DD.md
  weekly/YYYY-MM-DD.md
```

---

## Regenerate (Phase 1)

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
python3 tools/build_hermes_knowledge_layer.py
```

Read-only. Writes only to artifacts/ops/ tree. No production files touched.
Run after any spec install, cron change, or held-item state change.

---

## Four-layer model

Layer 1 — Capture (read-only sources):
  git status/log, crontab -l, agents/AGENT_REGISTRY.json,
  specs/changes/*.md, artifacts/audit/*.md, artifacts/ops/**/*.md,
  output/hedge_report/, data/snapshots/<date>/, production_data/decision_rulesets/

Layer 2 — Normalize:
  Convert to small machine-readable ledgers (see layout above).

Layer 3 — Reason:
  Contradiction detection, first-fire eval, held-branch drift checks,
  uncommitted-file detection, registry vs cron mismatch.

Layer 4 — Return:
  Daily brief (5-min operator summary), weekly synthesis.

---

## Held item schema (HELD_ITEMS_SEED in build tool)

Each item:

```python
{
  "id": "spec_087_b1b",              # snake_case, unique
  "title": "...",
  "status": "AWAITING_FIRST_FIRE",   # HELD | AWAITING_FIRST_FIRE | HELD_SUPPRESSED | SPEC_REQUIRED | CLOSED
  "last_evidence": "...",
  "blocker": "...",
  "next_allowed_action": "...",
  "not_allowed": ["...", "..."],
  "requires_operator_approval": True,
  "related_artifacts": ["path/relative/to/repo"],
  "related_cron": "cron expression + comment",
  "alert_condition": "...",
}
```

Status values:
  AWAITING_FIRST_FIRE  -- cron installed, first run not yet observed
  HELD                 -- blocked by dependency or decision
  HELD_SUPPRESSED      -- active suppression (crontab comment, registry suppressed)
  SPEC_REQUIRED        -- streak/alert condition requires spec before action
  CLOSED               -- move to "recently_closed" section, remove from active seed

---

## First-fire item schema (FIRST_FIRE_SEED in build tool)

```python
{
  "job": "biotech_hedge_report",
  "cron": "0 18 * * 5",
  "expected_first_fire": "2026-05-08T18:00:00-04:00",  # ISO8601 with tz
  "expected_artifacts": ["output/hedge_report/hedge_report_2026-05-08.json"],
  "expected_log": "logs/biotech_hedge_report.log",
  "status": "PENDING",              # seed value; check_first_fire_status() overwrites with eval
  "notes": "...",
  "pass_criteria": ["..."],
  "fail_criteria": ["..."],
  "alert_deadline": "2026-05-09T09:00:00-04:00",   # when FAIL/WARN escalates
}
```

Eval status values (set by check_first_fire_status()):
  PENDING_NOT_YET_DUE
  PASS
  WARN_ARTIFACT_NOT_YET_PRESENT
  WARN_DATE_MISMATCH
  WARN_VERDICT_UNREADABLE
  FAIL_ARTIFACT_MISSING_PAST_DEADLINE
  MISSED_DUE_TO_WSL_SLEEP          # legacy/manual fallback tag only
  MISSED_DUE_TO_ENV_SLEEP_UNKNOWN  # use when misses occur but host sleep state is unverified

Host baseline note:
- As of 2026-05-07, laptop sleep while plugged in was set to Never in this ops environment.
- Do NOT default misses to WSL sleep without evidence from power/scheduler logs.

---

## Contradiction check IDs

C1  bioshort_watch cron vs AGENT_REGISTRY status
C2  watchlist_current.json uncommitted
C3  biotech_hedge_report.py cron present/active
C4  uncommitted working tree (general)
C5  BIOSHORT_VERDICT as_of_date vs first-fire expected date

Add new Cxx checks to detect_contradictions() in build_hermes_knowledge_layer.py.
Severity: HARD_CONTRADICTION | POSSIBLE_DRIFT | OK

---

## Delivery: routing ledger outputs to the operator

After ledger artifacts are written, use `common/operator_delivery.py` to route
summaries to Town (Phase B+) or Telegram (hard failures):

```python
from common.operator_delivery import send_operator_event

send_operator_event(
    channel="town",
    severity="INFO",
    event_type="held_spec_ledger",
    title="Held-spec ledger updated",
    summary="6 held items. Bioshort first-fire due Fri 18:00 ET.",
    artifact="artifacts/ops/held_spec_ledger/latest.md",
    next_operator_action="Validate bioshort first-fire after 18:00 ET",
)
```

Default (Phase A): `OPERATOR_DELIVERY_DRY_RUN=1` — logs only, no sends.
Phase B: set `OPERATOR_DELIVERY_DRY_RUN=0`. Town channel sends email to
`TOWN_EMAIL` (default `djschulz@gmail.com`) via existing SMTP credentials.
Town routine filters on subject `[Hermes]` and creates a task.
No webhook URL or bearer secret needed — SMTP is already wired in `.env`.

See `in-code-operator-alerts` skill for full API, email format, and event types.

---

## Extending after a new spec install

1. Add entry to HELD_ITEMS_SEED in build tool with status AWAITING_FIRST_FIRE or HELD.
2. Add entry to FIRST_FIRE_SEED if a new cron was installed.
3. Add Cxx contradiction check if a new invariant should be monitored.
4. Run build tool. Review contradiction_ledger/latest.md.
5. Commit: spec doc + build tool changes together. Ledger artifacts go in same commit
   or as a separate "ops: regenerate knowledge layer" commit — do NOT bundle with
   unrelated production code changes.

---

## Analyst-synthesis receipt pattern (fleet steward)

The fleet steward produces TWO distinct output files per day:

1. `agents/fleet_steward/memory/<date>_receipt.md` — **deterministic receipt** (3–5KB).
   Generated by `tools/agent_heartbeat_checks.py`. Contains verdict (RED/YELLOW/GREEN),
   FAIL/WARN/STALE lists, per-agent heartbeat status. This is what cron writes automatically.

2. `agents/fleet_steward/memory/<date>_analyst_synthesis.md` — **narrative layer** (~18KB).
   Written by Hermes audit sessions (Mode B). Adds: per-signal IC trajectory, governance framing,
   shadow portfolio state, Spec status, forward action items, diff against prior receipt.
   NOT written by the deterministic tool — only written when an analyst session runs.

When running Mode B audit, the goal is to produce the `_analyst_synthesis.md`, using the
`_receipt.md` as the primary input (read it first, do NOT re-derive its findings).

The synthesis file should follow this structure:
```
FLEET VERDICT: <RED|YELLOW|GREEN>  (<ok>/<total> OK)  [quote receipt verbatim]

GOVERNANCE / CONTROL-PLANE
  [sentinel, promotions, ruleset health, any Frame B items]

SIGNAL HEALTH
  [IC dashboard readings with trajectory, not just latest]

PIPELINE
  [data auditor findings, price gaps, feed health]

PRODUCTION QA
  [verdict + per-check table excerpt + pool trajectory]

SHADOW PORTFOLIO
  [cumPnL, excess, maxDD, win rate, drawdown streak]

STANDING YELLOW (unchanged from prior)
  [list items with "unchanged since <date>"]

FORWARD ACTIONS
  [operator decision points, escalation thresholds, next dates]
```

Note: cron sessions that run the audit skill but do NOT have a chat session writing the
synthesis will only produce `_receipt.md`. The synthesis requires an LLM analyst pass.

---

## Cross-skill routing map (email/triage extensions)

Use these skills together with this umbrella ledger when inbox-driven signals or incident floods appear:

- `biotech-email-signal-triage`
  - Trigger: bulk inbox scan for investment-relevant biotech signals.
  - Output: normalized watchlist JSON with confidence labeling.

- `biotech-mover-validation`
  - Trigger: high-move biotech alert needs same-day source verification.
  - Output: VERIFIED_CATALYST vs PRICE_ONLY_NO_CATALYST vs UNVERIFIED.

- `github-email-incident-deduper`
  - Trigger: many GitHub `Run failed` emails across workflows/SHAs.
  - Output: deduped incident clusters + exact `gh run view ... --log-failed` handoff.

- `ci-account-gating-triage`
  - Trigger: CI flood may be caused by billing/quota/account gates.
  - Output: account-gate classification before code-level debugging.

- `security-alert-response-playbook`
  - Trigger: OAuth-app-added or unusual-sign-in security emails.
  - Output: severity + containment actions + evidence memo.

- `town-hermes-feedback-protocol`
  - Trigger: Town/Hermes findings must sync bidirectionally or policy conflicts require adjudication.
  - Output: channel selection (email/GitHub/MCP phase), owner assignment, and dual-system sync closure.

Operator sequence (recommended):
1. Run `biotech-email-signal-triage` for investment extraction.
2. For mover outliers, immediately run `biotech-mover-validation`.
3. In parallel for technical flood, run `github-email-incident-deduper`.
4. If systemic failures or budget alerts exist, run `ci-account-gating-triage` before CI deep-dive.
5. For auth/security alerts, run `security-alert-response-playbook` first (HIGH-priority path).

## Guardrails (always active)

MAY:
  read repo state, write ledger artifacts, produce briefs,
  flag contradictions, recommend next actions

MAY NOT:
  change scoring, change selector/ranker/EV/sizing,
  change cron without explicit approval, reactivate agents,
  commit held files, infer approval from stale notes,
  treat LLM synthesis as alpha evidence

---

## Phase 2B prep — Agent definition workflow

### Who is this for

When extending the ops ledger with LLM-driven agents (held-spec reporter, first-fire
validator, contradiction detector, operator brief), ALWAYS do a prep-only phase FIRST:
define the agent, register it as shadow (non-executable), write the runbook, and
validate via manual dry-run before wiring any cron.

### Workflow steps

1. **Create agent directory** under `agents/<agent-name>/`:
   - `IDENTITY.md` — one-line role description (e.g. "I am the Keeper of the Held Spec Ledger")
   - `SOUL.md` — full principles, constraints, trigger conditions, output format
   - `HEARTBEAT.md` — execution checklist: preflight checks, input reading, reasoning steps, output writing, verification

2. **Register in `agents/AGENT_REGISTRY.json`** with shadow status:
   ```json
   {
     "name": "hermes-held-spec-ledger",
     "status": "shadow",
     "llm_policy": "manual_only",
     "authority_level": "observe_only",
     "requires_preflight": true
   }
   ```
   - `"status": "shadow"` — prevents accidental cron wiring; agent exists as definition only
   - `"llm_policy": "manual_only"` — LLM calls require explicit `--skip-preflight` flag
   - `"authority_level": "observe_only"` — cannot write to production paths or trigger actions
   - `"requires_preflight": true` — preflight checklist runs before LLM prompt

3. **Create project-level runbook** at `references/hermes-<agent-name>-jobs.md` covering:
   - Invocation commands (both preflight and `--skip-preflight` dry-run)
   - Exact inputs the agent reads (artifact paths, ledger files)
   - Output format and destination
   - Failure modes with diagnostic steps
   - Rollback procedure
   - Activation criteria (see below)

4. **Determine if agent needs file data** before invoking:
   - Agent analyzes only its own definitions (SOUL.md, HEARTBEAT.md) or periodic state
     that fits in a short message? → Use `run_agent_direct.py` directly.
   - Agent needs to read ledger files, artifacts, or any filesystem data? → **Create a
     thin inline-data wrapper script** (see pattern below).

   **Critical constraint**: `run_agent_direct.py` is a **pure LLM call with no tool
   access** — the LLM cannot read files, run shell commands, or access the filesystem.
   It only sees what is passed in the `--message` parameter and the system prompt
   (constructed from IDENTITY.md + SOUL.md + HEARTBEAT.md). If the agent needs to
   analyze ledger contents, those contents must be serialized into the message before
   the LLM call.

5. **Create inline-data wrapper** for file-reading agents:
   Write a thin Python script (NOT a new agent, just a data-prep layer) that:
   - Reads the required input files (ledgers, artifacts)
   - Serializes file contents into a structured `--message` string
   - Invokes `run_agent_direct.py` as a subprocess with the constructed message
   - Handles missing files gracefully (reports MISSING, doesn't crash)

   The wrapper lives in `tools/run_hermes_<agent-name>.py` — it is NOT a new agent
   definition, it is a data-inlining utility for an existing agent definition.

   Example pattern (from `run_hermes_held_spec_ledger.py`):
   ```python
   # Load input files
   ledger = load_json(PROJECT_ROOT / "artifacts" / "ops" / "held_spec_ledger" / "latest.json")
   knowledge = load_json(PROJECT_ROOT / "artifacts" / "ops" / "knowledge_layer" / "latest_state.json")

   # Build message with inline data
   msg_parts = [f"as_of_time: {now}"]
   if ledger:
       msg_parts.append(f"--- Held Spec Ledger ---\n{json.dumps(ledger, indent=2)}")
   else:
       msg_parts.append("--- Held Spec Ledger ---\nMISSING")
   message = "\n\n".join(msg_parts)

   # Invoke agent via run_agent_direct.py
   subprocess.run([sys.executable, "tools/run_agent_direct.py",
       "--agent", "hermes-held-spec-ledger",
       "--message", message, ...])
   ```

   The wrapper should accept `--dry-run` (skip memory writes, pass `--skip-preflight`)
   and `--model <name>` (for provider fallback).

6. **Dry-run with `--skip-preflight`** to verify LLM output quality before wiring cron:
   ```bash
   # For agents without file dependencies (simple)
   python3 tools/run_agent_direct.py --agent <agent-name> --message DRY_RUN --skip-preflight

   # For agents with inline-data wrappers (file-reading)
   python3 tools/run_hermes_<agent-name>.py --dry-run --model claude-sonnet-4-20250514
   ```

   **Provider note**: The project default model (`meta-llama/Llama-3.3-70B-Instruct-Turbo`
   via Together AI) may be credit-capped (402). The wrapper scripts accept `--model`
   to route to an alternative:
   - `--model claude-sonnet-4-20250514` → Anthropic SDK via `ANTHROPIC_API_KEY`
   - See `run_agent_direct.py` for the full routing table: `llama` → Together,
     `claude-*` → Anthropic, anything else → Anthropic (default fallback).

7. **Commit prep branch** with scope prefix `spec: phase-2b-prep` — agent dirs +
   registry changes + runbook on their own branch. Do NOT bundle with cron wiring
   or feature work.

### Activation criteria (prep -> live)

ALL of these must be true before wiring a Phase 2B agent to cron:

- [ ] Agent directory exists with IDENTITY.md, SOUL.md, HEARTBEAT.md — reviewed and operator-approved
- [ ] AGENT_REGISTRY.json entry exists with `status: shadow` (flip to `active` only at activation)
- [ ] Runbook written and verified against actual agent output
- [ ] Minimum 2 dry-run invocations with `--skip-preflight` producing useful, non-repetitive LLM output
- [ ] No governance mutations or scoring changes in agent SOUL.md
- [ ] No Town/alert delivery in agent HEARTBEAT.md (Phase B activation is a separate gate)
- [ ] Operator has confirmed the schedule window (e.g. Mon 08:45 ET does not conflict with other cron jobs)
- [ ] OS crontab line uses the `run_agent_direct.py` pattern — NOT native Hermes cron (unless operator explicitly overrides)
- [ ] Output paths confirmed gitignored: `git check-ignore <path>` returns <path> for each

### Phase 2B activation state (as of 2026-05-19)

| Agent | Status | Scheduler | Driver | Notes |
|-------|--------|-----------|--------|-------|
| hermes-held-spec-ledger | **ACTIVE** (Mon 08:45 ET) | OS crontab (`45 8 * * 1`) | `run_hermes_held_spec_ledger.py` (inline-data wrapper, `--model claude-sonnet-4-20250514`) | Wrapper reads held_spec_ledger + knowledge_layer, inlines into `--message`. Observe-only. No Town/alert delivery. |
| hermes-first-fire-validator | PREP (shadow, manual-only) | On-demand only | `run_hermes_first_fire_validator.py` (inline-data wrapper) | Wrapper reads first_fire_ledger + knowledge_layer, inlines into `--message`. Deliberately excluded from cron — activate only after a new job's first fire occurs and 3+ dry-runs pass. |
| hermes-contradiction-detector | NOT YET STARTED | TBD | TBD | Phase 2C candidate |
| hermes-daily-operator-brief | NOT YET STARTED | TBD | TBD | Phase 3 candidate |
| hermes-knowledge-indexer | N/A | N/A | N/A | Not needed — build_hermes_knowledge_layer.py covers this deterministically |

Key decision: Phase 2B uses OS crontab + `run_agent_direct.py` pattern, NOT native
Hermes cron. Rationale: keep scheduler surface simple, one `crontab -l` is authoritative,
avoid double-scheduler confusion during prep. Re-evaluate if native Hermes cron gains
deterministic reliability for LLM reasoning jobs.

### Delivery (NOT YET WIRED)

Phase 2B prep agents are observation-only. They write analysis text to stdout only.
No Telegram, no Town, no email delivery. This is gated behind a future Phase B
checklist (see runbook for details).

### Branch naming convention

```
spec-<spec-number>-phase-2b-<component>-<YYYY-MM-DD>
```

Example: `spec-089-phase-2b-alerting-prep-2026-05-19`

Production cron-wiring goes on a SEPARATE branch from prep work. Never mix
feature/knowledge-graph code with cron/infrastructure changes on the same branch.

---

## Post-first-fire validation sequence (Spec 087 B1b — PASSED 2026-05-08)

**STATUS: PASSED.** `output/hedge_report/hedge_report_2026-05-08.json` confirmed:
as_of_date=2026-05-08, created 2026-05-08T23:23:34Z, n_positions=30, EW.
Spec 087 B2 (dashboard freshness envelope) is now UNBLOCKED.

Original validation sequence (for reference on any re-run or similar first-fire):

1. Run build tool: `python3 tools/build_hermes_knowledge_layer.py`
2. Check first_fire_ledger/latest.md — status should flip to PASS.
3. Verify BIOSHORT_VERDICT.json as_of_date == 2026-05-08.
4. Verify hedge_report_2026-05-08.json exists with recommendation line.
5. Check logs/biotech_hedge_report.log for no MASSIVE_API_KEY warnings.
6. If PASS: update HELD_ITEMS_SEED — flip spec_087_b1b to CLOSED, unblock spec_087_b2.
7. If FAIL/MISSED: set status by evidence.
   - Use MISSED_DUE_TO_WSL_SLEEP only when sleep evidence is confirmed.
   - Otherwise use MISSED_DUE_TO_ENV_SLEEP_UNKNOWN or FAIL as appropriate.
   Do NOT advance to B2. Surface to operator.

**Path note:** hedge_report artifact lives at `output/hedge_report/` — NOT `artifacts/hedge_report/`.
This is the only major artifact outside the `artifacts/` tree.

See references/spec-087-b1b-first-fire-checklist.md for the complete checklist.

---

## Agent Deactivation for Token Conservation

When token usage needs to be conserved (e.g., Grok API costs), agents can be deactivated following this procedure:

### Steps:
1. **Stop running agent processes**:
   - OpenClaw gateway: `pkill -f "openclaw/dist/index.js gateway"`
   - Hermes agent gateway: `pkill -f "hermes_cli.main gateway"`

2. **Disable systemd services** (prevents auto-restart):
   ```bash
   systemctl --user stop hermes-gateway.service openclaw-gateway.service
   systemctl --user disable hermes-gateway.service openclaw-gateway.service
   ```

3. **Comment out cron jobs** that invoke agents:
   - All `run_agent_direct.py` calls
   - All Grok API calls (Biotech Watch, Conference Abstracts)

4. **Verify**:
   - No agent processes running
   - Cron jobs commented out
   - Systemd services disabled

### Important Notes:
- The `agent_heartbeat_checks.py` script is designed to be efficient and only invokes an LLM when anomalies are detected, so it can remain active if needed.
- Always check `ps aux | grep -E "openclaw|hermes"` to verify no agent processes remain.
- Use `systemctl --user status` to confirm services are inactive.
- Cron changes should be verified with `crontab -l`.

### When to Use:
- When Grok/XAI token usage is high and needs to be reduced
- During maintenance windows
- When agents are not needed for extended periods

### Pitfalls:
- Forgetting to disable systemd services will cause agents to restart automatically
- Some cron jobs may be managed by different users (check with `crontab -l` and `/etc/crontab`)
- Always verify changes after making them

Untracked file: `specs/changes/spec_091_score_rank_pct_degradation_2026_05_07.md`

This is a GOVERNANCE MEMO ONLY — no code changes. It defines the evidence bundle
(CRT + IC + PIT + Checklist v2) required before any future action on `score_rank_pct`
or composite_score weights. Key dates:

- ~2026-05-15: post-13F cohort window closes (inst_delta_z inflation ends)
- ~2026-06-15: earliest defensible CRT + Checklist v2 (need ≥20 trading days post-window + 30 resolved PIT outcomes)

Decision tree:
- Branch A (best case): streak breaks on its own → spec closes, no evidence work
- Branch B: CRT shows cohort-window-driven → continue monitoring, no weight action
- Branch C+: CRT shows structural → full IC+PIT+Checklist v2 bundle required

Nothing to do until ~2026-06-15 unless streak breaks first (Branch A).
Streak monitor `4a96ad05405c` runs 22:00 ET Mon–Fri.

## DE Schema Gate — Absorbed from de-schema-gate-fix (CLOSED 2026-05-06)

Both bug types fixed in commit `bd777483`. Reference files contain full detail:
- `references/catalyst-validation-checks.md` — Python validation script, SNAPSHOT_COLUMNS
  drift fix (Spec 057 + 061 columns), set-based rank check logic, catalyst-date staleness fix
- `references/trapops-validation.md` — TrapOps monitoring tool, EES cross-check math,
  staleness patterns, execution stress thresholds, known expected data gaps

**Bug 1 (CLOSED):** `SNAPSHOT_COLUMNS` in `run_screen_columns.py` missing 11 Spec 057/061 columns.
**Bug 2 (CLOSED):** Sequential `expected_rank` iteration produced false WARNs — replaced with set-based check.
**Bug 3 (open):** `next_catalyst_date` staleness for `far_window` rows — see catalyst-validation-checks.md.

---

## In-Code Operator Alerts — Absorbed from in-code-operator-alerts

Pattern for condition-triggered alerts in pipeline ops/QA wrappers.
Reference files contain full detail:
- `references/two-channel-alert-design.md` — decision record: Channel 1 (Hermes cron deliver:telegram)
  vs Channel 2 (common/alerts.py send_operator_alert), rules, approved hook points
- `references/town-operator-delivery-design.md` — Spec 090 Phase A: Town email integration path,
  subject format, JSON payload schema, SMTP env vars, phase plan (A=dry-run done, B=live, C=hard fails)

**API summary:**
```python
from common.alerts import send_operator_alert
send_operator_alert(severity="FAIL", system="daily_production",
    message="...", dedupe_key="daily_production:snapshot_missing:<date>")

from common.operator_delivery import send_operator_event
send_operator_event(channel="town", severity="INFO", event_type="held_spec_ledger",
    title="...", summary="...", artifact="...", next_operator_action="...")
```

**Key rules:** Never alert from scoring math. Call only from ops/QA wrappers after
deterministic verdict. `OPERATOR_DELIVERY_DRY_RUN=1` is Phase A default.
`TELEGRAM_CHAT_ID` must be personal user ID from @userinfobot, NOT the bot token prefix.

---

## OpenClaw Cost Optimization (absorbed from openclaw-cost-optimization)

Class-level rule: treat token/cost controls as an ops scheduling problem, not as a one-off session artifact.

See `references/openclaw-token-cost-optimization.md` for:
- frequency-reduction heuristics
- deterministic/script-only substitution patterns
- operator budget review questions
- post-change validation checklist

When triaging fleet jobs, preserve governance/risk controls first, then reduce spend by
moving non-critical synthesis workloads from daily to weekly cadence.

---

## Pitfalls

- Scripts in `tools/` that import `common/` need `sys.path.insert(0, repo_root)` at the top,
  because `tools/` is one level below repo root and `common` is not on the default path.
  Pattern used in `build_hermes_knowledge_layer.py`: `REPO = Path(__file__).resolve().parent.parent`.
  Running directly (`python3 tools/build_hermes_knowledge_layer.py`) works from repo root via
  that REPO path. Running inline (`python3 -c "from common import ..."`) or from another dir
  requires `PYTHONPATH=/path/to/repo` prefix. Other tools use `sys.path.insert(0, str(REPO_ROOT))`
  explicitly (see `tools/biotech_hedge_report.py` line 34 for reference).
- Inline `-c` or heredoc Python with f-strings breaks silently when the string body contains
  `{dict_literal}` — Python treats `{key: val}` as a format specifier. Always write a `.py`
  script file for any smoke test or ad-hoc run that passes dict arguments. B0/B0.1 investigation showed child-process stdout can be
  captured and discarded. Always prefer filesystem artifact existence and as_of_date
  over log token presence.
- Do NOT use `crontab -e` to inspect; use `crontab -l`. The -e flag opens an editor.
- The held_spec_ledger/latest.md may be manually authored (seed from ops session) while
  held_spec_ledger/latest.json is generated. Both exist. The JSON is the machine-readable
  source; the .md may have richer narrative. Keep them in sync when updating.
- Date contradiction: the existing latest.md seeded with first-fire date "2026-05-09"
  (Saturday) but the cron `0 18 * * 5` fires Friday = 2026-05-08. The build tool uses
  2026-05-08 as the correct date. The manually authored .md had the wrong date.
- watchlist_current.json was accidentally committed and then reverted (commit 9c65f239).
  It is now clean. Do not re-introduce it in any Spec 087/088 commit.
- **Narrow-activation cron pattern**: When the user approves only 1 of N agents for cron activation, the safety sequence is: (1) live test (--no-dry-run) first to verify memory write works, (2) verify registry.json shows BOTH agents as shadow/manual_only/observe_only, (3) backup crontab: `crontab -l > ~/crontab.bak.$(date +%Y%m%d)`, (4) add ONLY the approved agent's entry, (5) verify the EXCLUDED agent is absent: `crontab -l | grep <excluded-name> | echo`, (6) save post-addition backup: `...post-<phase>`, (7) update runbook with cron line, model route, log path, exact disable command. The two-backup pattern (pre + post) lets you roll back to either state independently.
- artifacts/ops/ is NOT gitignored by default — verify .gitignore before assuming
  ledger artifacts are excluded. They may need explicit .gitignore entries if you do
  not want them tracked.
- Pre-commit hooks (black → isort → flake8) run in sequence in this repo. First commit
  attempt will FAIL if the file has: bare f-strings (F541), unused imports (F401).
  black/isort reformat on first attempt; re-stage reformatted file, then flake8 fires.
  Pattern: write plain strings (not f"...") for any string with no { } interpolation
  in markdown-builder list literals. See github-pr-workflow skill for the full recipe.
- Commit timeout on first attempt is normal (~15-20s for hooks). Check `git log --oneline -2`
  before assuming the commit failed — it may have landed after a hook-induced reformat.
