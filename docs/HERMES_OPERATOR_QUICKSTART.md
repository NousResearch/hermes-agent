# Hermes Operator Quickstart

Use this when you need to check, run, restart, or debug the local Hermes system.

## Canonical Runtime

- Core repo: `/Users/agent1/Code/hermes-agent`
- Hermes home: `/Users/agent1/.hermes`
- Operator scripts: `/Users/agent1/Operator/scripts`
- Active gateway label: `ai.hermes.gateway`
- Legacy label to avoid unless intentionally migrating:
  `com.agent1.hermes.gateway`
- Gateway wrapper: `/Users/agent1/Operator/scripts/hermes-gateway.sh`
- Gateway logs: `/Users/agent1/.hermes/logs/gateway.log` and
  `/Users/agent1/.hermes/logs/gateway.error.log`

## Health Check

```bash
cd /Users/agent1/Code/hermes-agent
hermes ops status
hermes ops status --markdown
hermes doctor
hermes gateway status
hermes gateway validate
hermes gateway incident-bundle --output /tmp/hermes-gateway-incident --force
hermes tools list
hermes fallback list
```

Interpretation:

- `hermes doctor` may warn about optional missing providers; that is not a
  broken startup path by itself.
- `hermes ops status` is the first-stop local operator view. It is read-only
  and redacted, summarizes gateway validation, API health, cron counts,
  health-loop receipts, disk usage, and log warning/error counts, and does not
  print raw log lines, cron prompt bodies, private memory, env values, or
  secrets.
- `hermes ops status --markdown` produces the same redacted status as a
  handoff-ready receipt for docs, tickets, or build-log summaries.
- `hermes gateway status` should show `ai.hermes.gateway` loaded and pointing
  at `Operator/scripts/hermes-gateway.sh`.
- `hermes gateway validate` is read-only. It should pass when the active
  service is wrapper-backed and `/health` is reachable; a loaded legacy label
  is reported as a warning, not an automatic failure.
- `hermes gateway incident-bundle` writes a local redacted bundle with
  validation output and metadata-only log/health-loop receipts. It does not
  copy raw logs, read private memory, restart services, or dump launchd
  environment.
- `hermes fallback list` currently reporting no fallback providers is a planned
  improvement, not a Phase 0 blocker.

## Launchd Check

Prefer the filtered check:

```bash
hermes gateway validate --json
launchctl print gui/$(id -u)/ai.hermes.gateway \
  | rg 'Label|PID|Program|ProgramArguments|StandardOutPath|StandardErrorPath|LastExitStatus'
```

Only inspect non-secret fields such as label, PID, stdout/stderr paths, and
program path. Do not dump launchd environment. Prefer
`hermes gateway validate --json` when you need a redacted receipt; if that is
not enough, review `hermes gateway status` before using raw `launchctl print`.

## Start, Stop, Restart

Prefer Hermes CLI paths:

```bash
hermes gateway start
hermes gateway stop
hermes gateway restart
```

If a launchd action is required, use only the active label:

```bash
launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway
```

Avoid the legacy label unless the task is explicitly to migrate or remove it:

```bash
com.agent1.hermes.gateway
```

## Logs

Use concise summaries when reporting to the user:

```bash
hermes logs gateway --since 30m --level WARNING
```

Do not paste raw log dumps unless they have been reviewed for secrets and
private content.

## Health Guardian

Health-loop receipts live under:

```bash
/Users/agent1/Operator/health-loop/
```

Start with:

```bash
cat /Users/agent1/Operator/health-loop/status.md
```

The guardian should stay pulse-first: confirm failure before repair, repair
only the active gateway, and preserve receipts.

## Common Recovery Ladder

1. Run `hermes gateway validate`.
2. Generate a redacted incident bundle if you need a shareable receipt.
3. Run `hermes gateway status`.
4. Check the active launchd label and wrapper path.
5. Check health-loop `status.md` and `status.json`.
6. Review recent warning/error logs in summary form.
7. If down, restart `ai.hermes.gateway` through Hermes or launchd.
8. Re-run `hermes doctor`, `hermes gateway validate`, and
   `hermes gateway status`.
9. Record commands and results in `docs/HERMES_BUILD_LOG.md`.

## Phase 0 Rollback Aid

This Phase 0 pass uses two local rollback aids:

- Branch isolation: `hermes-control-plane-20260520-182036`.
- Non-empty staged-doc patch snapshot:

```bash
/Users/agent1/Code/hermes-agent/.codex-backups/phase0-control-plane-staged-20260520-validated.patch
```

To abandon only the staged Phase 0 docs without touching unrelated work:

```bash
cd /Users/agent1/Code/hermes-agent
git status --short --branch
git restore --staged AGENTS.md docs/HERMES_*.md
git restore AGENTS.md
rm docs/HERMES_*.md
```

To reapply the Phase 0 docs from the local patch after inspecting the worktree:

```bash
cd /Users/agent1/Code/hermes-agent
git apply .codex-backups/phase0-control-plane-staged-20260520-validated.patch
```

Do not use broad reset commands unless the user explicitly asks for destructive
cleanup.

## Do Not

- Do not print secrets or raw `.env`.
- Do not replace the gateway wrapper with a bare command.
- Do not restart the legacy gateway label by habit.
- Do not perform external sends, posts, purchases, trades, deploys, or account
  changes without explicit confirmation.
- Do not treat optional missing provider credentials as a core startup failure.
