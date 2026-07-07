# BU-1 — Hermes guardrails: write-fence + repo-guard hooks, test matrix, config template, installer

Plan of record: `~/brain/wiki/synthesis/hermes-adoption-plan-v4.md` (Part I amendment + Part II v3 §BU-1, F3, F3.5).
Repo: BuiltOnPurpose/hermes-agent fork, pinned base `6a6766fb896043ab3757a30713f1248b6054e0a6`.
Worktree: `~/projects/hermes-agent-guardrails`, branch `feature/guardrails`, tmux `codex-guardrails`.
All new artifacts live under a NEW top-level dir `bop/` — zero edits to any upstream file (upstream pulls must never conflict).

## Objective

Ship the two Hermes pre_tool_call guard scripts, their test matrix, the v4 config template, and an
idempotent installer, so Hermes can be granted write tools with hard boundaries enforced at the
hook layer (defense-in-depth under `approvals.mode: manual`).

## Verified runtime contract (Codex: re-verify each claim in the worktree before coding)

- Hook wire protocol (`agent/shell_hooks.py` module docstring, lines ~29-52): script receives
  stdin JSON `{"hook_event_name","tool_name","tool_input":{...},"session_id","cwd","extra"}`;
  BLOCK by printing stdout JSON `{"decision":"block","reason":"<msg>"}`; empty stdout = allow.
  Scripts run via `shlex.split`, `shell=False`, `~` expanded; non-TTY registration requires
  `hooks_auto_accept: true` in config.
- Matcher semantics (`agent/shell_hooks.py` ~line 139): regex FULLMATCH against `tool_name`.
  So matchers are exactly `write_file|patch` and `terminal`.
- File tools both carry the target in `tool_input.path` (`tools/file_tools.py:793`
  `write_file_tool(path, content, ...)`; `:850` `patch_tool(mode, path, old_string, ...)`);
  content fields: `content` (write_file), `old_string`/`new_string` (patch).
- Terminal tool carries the command in `tool_input.command`.

## Deliverables (all under `bop/`)

1. `bop/agent-hooks/write-fence.sh`
   - Triggers on `write_file|patch`. Parse stdin JSON with `python3 -c` (no jq dependency).
   - Resolve `tool_input.path` (absolute-ize against `cwd`, then resolve symlinks/`..` via
     `python3 os.path.realpath` so traversal can't escape).
   - ALLOWLIST (allow only if the resolved path is inside): `~/assistant/`, `~/brain/raw/`,
     `~/osr/_intake/`, `~/.hermes/workspace/`, `~/.hermes/outbox/`, and the exact file
     `~/.hermes/MEMORY.md`. Everything else → BLOCK. (B2 later widens to
     `~/brain/wiki/entities/` + `concepts/` — ship those two paths now behind
     `~/.hermes/fence-wiki-enabled` with `HERMES_FENCE_WIKI=1` also honored for tests,
     default off.)
   - Explicit deny doubles (belt+braces, checked before allowlist): any path under `~/ds-max`,
     `~/HERK-2`, `~/brain/wiki/synthesis`, `~/brain/_meta`, `~/.hermes/agent-hooks`,
     `~/.hermes/config.yaml`, `~/.hermes/.env`, `~/.hermes/auth.json`,
     `~/.hermes/SOUL.md`, `~/.hermes/USER.md`.
   - In patch mode, parse V4A content with the vendored pinned copy of `tools/patch_parser.py`
     under `bop/agent-hooks/patch_parser.py`; collect `op.file_path` and non-empty `op.new_path`
     from every operation. Parser import errors and parser errors → BLOCK; never fall back to a
     regex parser.
   - NPI content scan on `content`/`old_string`/`new_string`/`patch`: BLOCK on best-effort
     pattern matches for SSNs (hyphen/space/dot/bare 9-digit), FICO-context numbers, and
     income-context amounts. Scan runs even for allowlisted paths; this is pattern-based and not
     sound.
   - Fail CLOSED: any parse error, missing python3, or unexpected input → BLOCK with reason.
2. `bop/agent-hooks/repo-guard.sh`
   - Triggers on `terminal`. BLOCK when `tool_input.command` matches (case-insensitive,
     word-boundary aware): `git (push|commit|merge|rebase|reset|worktree)` / `gh pr` when the
     command string also references `ds-max` or `HERK-2` (path or `-C` arg); AND any
     write-redirect/mutation pattern (`>`, `>>`, `tee`, `sed -i`, `rm`, `mv`, `cp`, `mkdir`,
     `touch`, `ln`) whose argument string contains `ds-max/` or `HERK-2/` or `~/ds-max` or
     `~/HERK-2` or the absolute equivalents.
   - Best-effort by design (documented in header comment): the shell-string classifier is
     trivially evadable; `approvals.mode: manual` and running Hermes without push credentials
     to `ds-max`/`HERK-2` are the real backstops.
   - Fail CLOSED on parse errors, same as write-fence.
3. `bop/tests/hook-matrix.sh` — self-contained runner, NO Hermes runtime needed: pipes synthetic
   stdin JSON through both scripts and asserts stdout. Cases (minimum):
   - write allowed: `~/assistant/ledger.md`, `~/.hermes/workspace/r.md`,
     `~/.hermes/outbox/r.md`, `~/.hermes/MEMORY.md`, `~/brain/raw/x.md`, `~/osr/_intake/y.md`
   - write blocked: `~/ds-max/x.py`, `~/HERK-2/a.md`, `~/brain/wiki/synthesis/s.md`, `~/brain/_meta/rules.md`, `~/Documents/z.txt`
   - Hermes control-surface blocked: `~/.hermes/agent-hooks/write-fence.sh`,
     `~/.hermes/config.yaml`, `~/.hermes/.env`, `~/.hermes/auth.json`,
     `~/.hermes/SOUL.md`, `~/.hermes/USER.md`
   - traversal blocked: `~/assistant/../ds-max/x.py`, path with symlink into ds-max (create temp symlink in the test)
   - V4A parser drift blocked: `*** Move File:` with destination inside `~/ds-max`,
     `***Add File:` targeting `~/HERK-2`, and missing vendored `patch_parser.py`
   - NPI blocked (even on allowed path): seeded SSN, spaced SSN, "FICO 742", "income $85,000"
   - NPI clean pass: "loan stage: submitted, next action: call" writes fine
   - terminal blocked: `git -C ~/ds-max commit -m x`, `gh pr create` with cwd context arg referencing ds-max, `echo hi > ~/HERK-2/f`, `sed -i '' s/a/b/ ~/ds-max/f.py`
   - terminal allowed: `git -C ~/.hermes/workspace commit -m receipts`, `ls ~/ds-max` (read-only), `cat ~/HERK-2/docs/x.md`
   - malformed stdin → both scripts BLOCK (fail-closed proof)
   - exit non-zero on any case failure; print per-case PASS/FAIL table.
4. `bop/config/config.yaml.template` — full v4 wiring: model block (anthropic/claude-sonnet-5;
   codex aux vision/web_extract/session_search; delegation openai-codex gpt-5.4-mini),
   `approvals: {mode: manual, cron_mode: deny}`, `hooks_auto_accept: true`, `hooks.pre_tool_call`
   entries wiring BOTH scripts at `~/.hermes/agent-hooks/`, `platform_toolsets.cli: [hermes-cli]`,
   commented-out telegram block (A1 fills token/chat-id), `stt: {enabled: true, provider: local}`.
5. `bop/install.sh` — idempotent deploy: copy agent-hooks/* to `~/.hermes/agent-hooks/` (chmod 755),
   merge-or-write config from template WITHOUT clobbering an existing `~/.hermes/config.yaml`
   secrets/customizations (strategy: if config exists, print a diff and require `--force` to
   overwrite; always safe-copy hooks); never touches `.env`/`auth.json`. Prints receipt lines.

## Data-engineer pre-checks

Freshness: N/A. Temporal bias: N/A. Normalization: N/A. Sample size: N/A. Pipeline integrity:
N/A. Data lineage: N/A. (Infra/guardrail tooling — no data pipeline, no scored data surface.)

## Non-goals

- No edits to any file outside `bop/`. No upstream code changes. No Telegram/BotFather setup.
- No Hermes skills (BU-2+). No launchd. No changes to `~/.hermes` at build time (install is a
  separate post-merge step run by the orchestrator).

## Acceptance (machine gate for closer)

- `bash bop/tests/hook-matrix.sh` exits 0 with all cases PASS (this IS the repo's test gate for
  this BU; upstream test suite NOT run — bop/ is isolated and upstream tests take the tree as-is).
- `shellcheck` clean on all three shell scripts if shellcheck present; otherwise `bash -n` clean.
- Zero diffs outside `bop/` (`git diff --stat origin/main` shows only `bop/` + this spec file).
- Security scan: Claude Code `/security-review` runs on `bop/agent-hooks/write-fence.sh`,
  `bop/agent-hooks/repo-guard.sh`, and `bop/install.sh` before the landing (these scripts ARE the
  privilege boundary: path allowlist, NPI regex scan, repo-mutation interdiction, secrets-safe
  install). Scan verdict attaches to the gate-receipt packet.

## Post-merge (orchestrator, not this BU)

`bash bop/install.sh` → re-run matrix against installed copies → F3.5 chain-smoke canary → wire
hooks live in `~/.hermes/config.yaml`.

## Rollback

Revert the squash-merge commit; `~/.hermes` untouched until install step, so runtime rollback =
don't run install.sh.
