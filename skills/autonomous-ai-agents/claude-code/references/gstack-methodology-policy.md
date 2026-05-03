# GStack Methodology Policy for Hermes → Claude Code

Date adopted: 2026-05-01

## Position

GStack is valuable as a Claude Code engineering-methodology pack, not as a Hermes main-runtime skill pack.

Do not install `garrytan/gstack/*` into `~/.hermes/skills` or make Hermes load the full GStack bundle. It is Claude Code-first, references `~/.claude/skills/gstack`, `CLAUDE_SKILL_DIR`, and Claude-style hooks. Direct Hermes installation creates half-working skills and confusing triggers.

## Preferred Use

Use GStack through Claude Code subprocesses when the task is coding/product/QA/review work:

- product / founder planning → GStack-style office-hours / CEO review
- implementation plan review → GStack-style eng review
- code review → GStack-style review
- root cause debugging → GStack-style investigate
- browser QA report → GStack-style qa-only
- weekly project reflection → GStack-style retro

Hermes remains the cockpit. Claude Code is the coding worker. GStack is the worker's methodology layer.

## Two Execution Modes

### Mode A — Distilled methodology prompt, safest default

Use `~/.hermes/scripts/claude_code_safe.py` in `--mode ask|review|code` and put the relevant GStack methodology into the prompt explicitly.

This keeps the safe Hermes wrapper (`--bare`) and avoids loading unknown Claude hooks/plugins.

Example:

```bash
~/.hermes/scripts/claude_code_safe.py --mode review --workdir /repo --turns 3 --timeout 600 --prompt '
Use GStack-style review methodology, but do not run GStack slash commands.
Review current diff for production bugs, trust-boundary issues, data loss, missing tests, and operational risks.
Return: BLOCKERS, SHOULD_FIX, NICE_TO_HAVE, TESTS_RUN, VERDICT.
'
```

### Mode B — Real `/gstack-*` slash skills, isolated only

Only use this when the user explicitly wants real GStack slash skills.

Requirements:

1. GStack installed under Claude Code, not Hermes.
2. Namespaced skills enabled (`/gstack-qa`, `/gstack-review`, not `/qa`).
3. Telemetry off.
4. Auto-upgrade/team mode off.
5. Run in an isolated Claude Code session/worktree when edits are possible.
6. Never run deploy/ship skills without separate explicit approval.

If installing, prefer this guarded shape after checking `bun` exists:

```bash
git clone --single-branch --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
cd ~/.claude/skills/gstack
bin/gstack-config set telemetry off
bin/gstack-config set auto_upgrade false
bin/gstack-config set update_check false
bin/gstack-config set skill_prefix true
./setup --host claude --prefix --no-team --quiet
```

Do not install GStack just because a task mentions review/QA. Use Mode A unless real slash-skill behavior is required.

## Allowed / Preferred GStack Subset

Good candidates:

- `/gstack-plan-ceo-review`
- `/gstack-plan-eng-review`
- `/gstack-review`
- `/gstack-investigate`
- `/gstack-qa-only`
- `/gstack-retro`
- `/gstack-cso` for explicit security audits

High-side-effect skills require explicit user approval and usually a worktree:

- `/gstack-ship`
- `/gstack-land-and-deploy`
- `/gstack-setup-gbrain`
- `/gstack-setup-browser-cookies`
- `/gstack-pair-agent`
- `/gstack-skillify`
- `/gstack-gstack-upgrade`

## Safety Notes

- The GStack hub skills are community source. Hermes security scan flagged GStack `guard` as `CAUTION` because it declares privileged tool surfaces.
- Direct single-skill install can be broken: e.g. `guard` references sibling `careful` and `freeze` hook scripts that are not present when installed alone.
- GStack has local analytics and optional Supabase telemetry. Keep telemetry off unless the user explicitly chooses otherwise.
- Full setup may require Bun and Playwright Chromium; that is a supply-chain/system side effect and should not be hidden inside an unrelated task.

## Prompt Patterns

### Review

```text
Use GStack-style review methodology without loading slash skills. Inspect the diff as a production reviewer. Check: data loss, auth/trust boundaries, concurrency/races, migration safety, missing tests, observability, rollback. Return blockers first.
```

### Investigate

```text
Use GStack-style investigate methodology: reproduce/observe, collect evidence, form hypotheses, verify root cause before fixing. Do not implement until root cause is stated with evidence.
```

### QA-only

```text
Use GStack-style QA-only methodology. Produce a report only, no code changes. Include tested flows, evidence, screenshots/log references if available, severity, repro steps, and ship-readiness verdict.
```

### Plan review

```text
Use GStack-style CEO + eng review: challenge the product premise, then lock architecture/data flow/test matrix. Return plan gaps and concrete revisions before implementation.
```
