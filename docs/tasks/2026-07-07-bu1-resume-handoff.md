# Hermes adoption — Fable resume handoff (2026-07-07)

> Durable takeover record. A prior orchestrator session was silently downgraded off Fable
> partway through BU-1; this doc is the exact resume state so a fresh Fable session continues
> without redoing anything. Paste the "Takeover prompt" block into the new session.
> Plan of record: `~/brain/wiki/synthesis/hermes-adoption-plan-v4.md` (brain d15bafd).
> Working plan + phase list: `~/.claude/plans/you-are-the-fable-compressed-puzzle.md`.

## Current state at a glance

- **DONE + committed:** Step 0 (brain d15bafd), fork+pin (fork main @ 6a6766f), F0, F1.5, F2
  (~/.claude a45879b, ~/HERK-2 1a418f5, ~/brain 2503580), lesson (~/.claude 7fb6743).
- **RESUME POINT:** BU-1 guardrails built on `feature/guardrails` (3 commits, NOT pushed, no PR);
  two-pass security review complete with FOUR findings; must self-repair 1-3 + document 4 before
  closer lands it.
- **BLOCKED ON MIKE:** F1 P-1 smoke (Claude subscription `claude setup-token` — Keychain creds
  expired). Also A-0006 / A-0007 ledger rows unbooked (Task-tool subagents lack Calendar MCP).
- **Live:** tmux `codex-guardrails` alive + idle (resume the warm Codex context with `resume --last`).
- **Never** run `hermes update` / `git pull` on ~/hermes-agent — the pin is deliberate.

## Security findings (both passes complete) — BU-1 must fix 1-3, document 4

1. **CRITICAL (9/10) — guardrail self-disarm.** write-fence allow-lists ALL of `~/.hermes`, but
   the hooks live at `~/.hermes/agent-hooks/*.sh` and wiring is `~/.hermes/config.yaml`, so a
   prompt-injected agent can overwrite its own fence to `exit 0` — total persistent bypass. SPEC
   refinement: narrow the allow-root to agent-writable subpaths ONLY (`~/.hermes/workspace`,
   `~/.hermes/outbox`, `~/.hermes/MEMORY.md`), and add to deny_roots: `~/.hermes/agent-hooks`,
   `~/.hermes/config.yaml`, `~/.hermes/.env`, `~/.hermes/auth.json`, `~/.hermes/SOUL.md`,
   `~/.hermes/USER.md`. Deny runs before allow. Add matrix cases both ways.
2. **HIGH (10/10, reproduced vs real tools/patch_parser.py) — V4A parser drift.** Fence misses
   `*** Move File:` and no-space `***Add File:`; a decoy allowed `*** Update File:` line plus a
   smuggled Add/Move into ds-max/HERK-2 is ALLOWED. Fix: import
   `tools.patch_parser.parse_v4a_patch()` in write-fence, collect file_path + new_path from every
   op incl MOVE. Add matrix cases (Move into ds-max, no-space Add into HERK-2).
3. **MEDIUM (8/10) — NPI regexes format-specific.** Spaced SSN / bare score / paraphrased income
   evade onto allowed paths. Widen SSN (spaces/dots/bare-9-digit); loosen or explicitly document
   FICO/income as best-effort. Add matrix cases.
4. **MEDIUM (8/10) — repo-guard string-classifier evasion** (`printf \xNN`, `alias g=git`). Keep
   best-effort (approvals.mode:manual is backstop); just harden the header to state it is trivially
   evadable and never the sole control. No matrix change.

After fixing: `bash bop/tests/hook-matrix.sh` green -> re-run security pass -> closer (push, PR vs
fork main, auto-merge on all-green; no direct-to-main) -> `bash bop/install.sh` -> re-run matrix vs
installed copies -> F3.5 chain-smoke -> kill codex-guardrails -> /checkpoint. That closes BU-1 + F3/F3.5.

## Takeover prompt (paste into the fresh Fable session)

You are the Fable orchestrator seat, resuming the Hermes adoption build. A prior session was
silently downgraded off Fable partway through; you are taking over to finish it correctly. First,
confirm you are actually on the Fable seat before making any judgment calls. Do NOT restart from
scratch — nearly everything is done and durable in git; you are resuming at one specific point.

PLAN OF RECORD: ~/brain/wiki/synthesis/hermes-adoption-plan-v4.md (committed, brain d15bafd).
WORKING PLAN + full phase list: ~/.claude/plans/you-are-the-fable-compressed-puzzle.md.
Orchestrate autonomously per the plan's "Orchestrator model" section. Ultracode authorized for
the judgment calls below; token-budget guardrails apply. The ONLY stops: Mike-physical actions
(BotFather, OAuth consents, pairing approvals, sending drafts), closer ESCALATE / red machine gate
after one self-repair, class-independent escalations (new spend, destructive/irreversible ops,
verification-truth surface edits), and 3-round dual-review non-approval. Do NOT put
reasoning-extraction phrasing ("show/explain/narrate your reasoning") in any agent, skill, or
prompt you author — that is what tripped the classifier and caused this downgrade.

=== DONE and committed — verify with git log, do NOT redo ===
- Step 0 durable handoff: v4 plan of record (brain d15bafd).
- Fork BuiltOnPurpose/hermes-agent created; ~/hermes-agent has origin=fork, upstream=NousResearch;
  fork main force-pinned to 6a6766fb8960 (the AUDITED commit). NOTE: the local clone was 14,773
  commits behind upstream — the pin intentionally stays on the audited tree. NEVER run
  `hermes update` or `git pull` on ~/hermes-agent. ~/.hermes/PINNED-COMMIT records this.
- F0: ~/.hermes-venv (uv, Python 3.13.13), hermes v0.14.0 editable install, faster-whisper.
- F1.5: /auto-forge + /absence backlog audits saved to ~/.hermes/workspace/backlog.md. Key
  business items surfaced (already added to the assistant ledger, see below): WelcomeFunds Gold
  Plan $7.5k signed-but-uncollected 12 days; CV3 Financial (Brando Tessar) primary job-pursuit
  target cold 48 days.
- F2: ~/.hermes/{SOUL,USER,MEMORY}.md authored (SOUL is rules-first, <20k chars, no
  reasoning-extraction phrasing). AGENTS-REGISTRY.md gained a "Hermes agents" runtime section +
  hermes row; MEMORY-CONTRACT.md gained three Hermes surface rows (MEMORY.md, state.db LOCAL-ONLY,
  workspace+outbox); contract-version bumped v2.6 -> v2.7 in BOTH files; system-sync-check.sh
  taught the new runtime header. Committed: ~/.claude a45879b, ~/HERK-2 1a418f5, ~/brain 2503580.
  ~/.hermes/config.yaml has the v4 bootstrap model block (provider anthropic, default
  claude-sonnet-5; codex vision/web_extract aux; openai-codex gpt-5.4-mini delegation; approvals
  manual, cron_mode deny; stt local). Codex auxiliary is authed (own token copy in
  ~/.hermes/auth.json).
- Lesson committed (~/.claude CLAUDE.md 7fb6743): Task-tool subagents do NOT receive
  ToolSearch/claude.ai-MCP grants on this build, so the assistant agent's booking pass fail-closes.
  Ledger rows A-0006 (WelcomeFunds $7.5k collection) and A-0007 (CV3 follow-up) were ADDED but
  remain UNBOOKED — they book once Hermes A3 owns the calendar lane, or from a main session with
  Calendar MCP. Do not re-dispatch the assistant subagent expecting Calendar tools.

=== RESUME EXACTLY HERE: BU-1 guardrails — built, security-reviewed, must self-repair before landing ===
Code is committed on branch feature/guardrails in worktree ~/projects/hermes-agent-guardrails
(3 commits: guard hooks, hook matrix, install template + spec). NOT pushed, NO PR. The tmux
session codex-guardrails is ALIVE and idle (no process inside) — resume the warm Codex context
with `codex exec -m gpt-5.5 -c model_reasoning_effort=high --dangerously-bypass-approvals-and-sandbox
-C ~/projects/hermes-agent-guardrails resume --last "<prompt>"`. Build verified before review:
bash bop/tests/hook-matrix.sh = 32 passed 0 failed; git diff --stat origin/main..HEAD shows only
bop/ + docs/tasks/2026-07-07-bu1-guardrails.md.

A TWO-PASS security review (both passes complete) found FOUR findings. BU-1 MUST NOT land until
1-3 are fixed and 4 is documented. Drive the fix through the SAME codex-guardrails session
(dual-review: you adversarially review Codex's patch, then it executes), then re-run the matrix
to green and re-run a security pass, THEN dispatch closer:

  1. CRITICAL (9/10) — guardrail self-disarm. write-fence.sh allow-lists ALL of ~/.hermes, but the
     hooks themselves live at ~/.hermes/agent-hooks/*.sh and their wiring is ~/.hermes/config.yaml,
     so a prompt-injected agent can write_file its own fence to `exit 0` — a total persistent
     bypass. This is a SPEC refinement, not just a code fix: the v4 plan's allowlist intended
     "~/.hermes/ (scratch, workspace, outbox)", not the whole tree. Narrow the write-fence
     allow-root from ~/.hermes to the genuinely agent-writable subpaths ONLY — ~/.hermes/workspace,
     ~/.hermes/outbox, and ~/.hermes/MEMORY.md (the B2.5 cron writes MEMORY.md) — and add to
     deny_roots: ~/.hermes/agent-hooks, ~/.hermes/config.yaml, ~/.hermes/.env, ~/.hermes/auth.json,
     ~/.hermes/SOUL.md, ~/.hermes/USER.md. Deny runs before allow, so a deny entry closes it
     cleanly. Add matrix cases proving each control-surface path is blocked and workspace/outbox/
     MEMORY.md still allowed.
  2. HIGH (10/10, reproduced against the real tools/patch_parser.py) — V4A patch parser drift.
     write-fence's hand-rolled regex misses `*** Move File:` and no-space `***Add File:` directives
     that the real parser executes, so a patch with a decoy allowed-path `*** Update File:` line
     plus a smuggled Add/Move into ds-max/HERK-2 is ALLOWED. Fix by importing
     tools.patch_parser.parse_v4a_patch() inside write-fence and collecting file_path AND new_path
     from every returned op including MOVE. Add matrix cases: Move File into ds-max, ***Add File:
     (no space) into HERK-2.
  3. MEDIUM (8/10) — NPI regexes are format-specific. Spaced SSN ("123 45 6789"), a bare score
     ("742"), and paraphrased income ("brings home about $85,000") all evade onto an allowed path.
     Widen SSN to spaces/dots/bare-9-digit; either loosen FICO/income proximity or document
     NPI-scan as explicitly best-effort. Add matrix cases.
  4. MEDIUM (8/10) — repo-guard string-classifier evasion (`printf \xNN`, `alias g=git`). This
     hook is ALREADY documented best-effort with approvals.mode:manual as backstop, so keep it
     best-effort — do NOT try to make a shell-string classifier sound. Just harden the header to
     state it is trivially evadable and must never be the sole control; the real defense is
     approvals + running Hermes without push creds to ds-max/HERK-2. No matrix change required.

  After the fix: bash bop/tests/hook-matrix.sh green -> re-run a security pass on the three scripts
  -> dispatch the closer agent with the gate-receipt packet (matrix PASS + diff-scope proof +
  security verdict) to push feature/guardrails, open the PR against fork main, auto-merge on
  all-green. No direct-to-main. On red/ESCALATE after one self-repair, stop and surface to Mike.
  After merge: run `bash bop/install.sh` to deploy hooks+config into ~/.hermes/, re-run the matrix
  against the INSTALLED copies, then F3.5 chain-smoke canary to prove the HERK-2 chain is untouched.
  Kill the codex-guardrails tmux session at close-out. /checkpoint. That completes BU-1 + F3/F3.5.

=== STILL BLOCKED ON MIKE: F1 P-1 smoke ===
Hermes's primary brain is the native anthropic provider on the Claude Code SUBSCRIPTION — this is
code-verified SUPPORTED-IN-CODE (hermes model option 1, OAuth, Authorization: Bearer +
anthropic-beta oauth-2025-04-20; see plan CHANGE 1), NOT policy-blocked. The on-disk Claude Code
Keychain credentials are EXPIRED and the refresh endpoint returns 400, so `hermes -z` fails with
AuthError on provider anthropic. Mike must re-run `claude setup-token`: if it prints an
sk-ant-oat token, it goes into ~/.hermes/.env as CLAUDE_CODE_OAUTH_TOKEN (chmod 600); if it only
does browser consent, it refreshes the Keychain creds and the resolver (env var outranks Keychain)
picks them up. Then run the P-1 smoke:
  ~/.hermes-venv/bin/hermes -z "What macOS version is this machine running? Check with your terminal tool and answer in one line."
Expect a correct tool-loop answer on provider anthropic, zero vault access. Gates: no metered
ANTHROPIC_API_KEY anywhere (grep ~/.hermes/{.env,config.yaml}); confirm ~/.claude/.credentials.json
mtime unchanged by the Hermes run. If auth still fails after a fresh token, drop to fallback rung 2
(local Anthropic-Messages bridge via provider: custom) per plan CHANGE 1 — do NOT wire a metered key.

=== AFTER BU-1 lands and F1 passes ===
Continue the plan sequencing: A1 (Telegram gateway — Mike does BotFather), then BU-2+A2 (assistant
skills + single-unit ledger cutover that retires the assistant agent), then BU-3/A3/A4, BU-4/A5,
BU-5/B1, BU-6/B2/B2.5, B3, B4. One build unit through the chain at a time (one Codex session, no
concurrent workflows). Every unit: spec -> validator 8-check -> Codex dual-review in a
~/projects/hermes-agent-<slug> worktree -> you adversarially review (fable-mode) -> closer lands ->
install/deploy -> /checkpoint. Registry/contract rows ship same-unit; nothing retired until its
replacement passes acceptance + grace.
