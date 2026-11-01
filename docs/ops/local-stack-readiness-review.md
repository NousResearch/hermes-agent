# Local Stack Readiness Review

This review path exists to separate connection risk from model/runtime changes.
Do not switch the main Hermes provider or local default model until this review
is green.

Remote-only constraint:

- Assume the operator may have only remote/headless access to the Mac.
- Prefer terminal-safe login paths and avoid steps that depend on opening a local
  browser window on the machine.

## Goal

Prepare a safe reviewable path for:

- local `gh` authentication
- Codex/ChatGPT GitHub connector access
- GitHub Copilot provider reuse inside Hermes
- Hugging Face access for later model selection/download
- later local-model rollout with rollback preserved

## Current local baseline

- Repo: `hermes-agent`
- Local review branch: `benne/review-local-stack-readiness`
- Remote: `origin https://github.com/NousResearch/hermes-agent.git`
- Known risk: local branch is behind `origin/main`; do not mix a large upstream sync
  with provider/model surgery in the same step
- Known Hermes risk: at least one cron job is already in `drift_skip` because the
  global inference provider moved from `copilot` to a local custom provider

## Required gates before model changes

1. GitHub CLI
   - `gh auth status` must show the intended GitHub account
   - for remote-only access, prefer `gh auth login --web --clipboard` only if a
     browser can be completed elsewhere, otherwise use a token-based or other
     headless-safe flow
   - push rights for the eventual review branch must be confirmed before relying
     on PR-based review

2. Codex/ChatGPT GitHub connector
   - confirm the connector can read the intended `hermes-agent` repository
   - treat this as separate from local `gh` login

3. GitHub Copilot inside Hermes
   - verify whether Copilot remains a reference/cloud path or should be removed
     from active routing
   - do not overwrite provider settings until cron drift is reviewed

4. Hugging Face
   - `hf auth whoami` must succeed, or `HF_TOKEN` must be deliberately supplied
   - for remote-only access, use CLI token entry or env-based auth, not a local
     browser dependency
   - no token may be embedded in config diffs, issue bodies, or logs

## Audit command

Run:

```bash
python3 scripts/audit_local_stack_readiness.py --remote-only
```

The script reports:

- local git branch/remote state
- `gh` login state
- `hf` login state
- Hermes `model:` block summary
- Copilot/Hugging Face provider traces
- drifted cron jobs
- auth-source metadata without exposing secrets

Machine-readable output:

```bash
python3 scripts/audit_local_stack_readiness.py --json
```

## Recommended review order

1. Run the audit script and save the output as review evidence.
2. Fix external auth blockers first:
   - `gh auth login`
   - Hugging Face login or `HF_TOKEN`
   - connector access confirmation
3. Decide Copilot role:
   - keep as reference/cloud fallback
   - or retire from active routing after drift cleanup
4. Only then start the local model rollout branch/work:
   - primary local model for Hermes agent use
   - secondary freer local model kept separate
   - rollback target preserved

## Stop conditions

Stop and review before any provider/model switch if one of these is true:

- `gh` is not logged in
- Hugging Face auth is missing
- connector repo access is still unverified
- drifted cron jobs remain unreviewed
- secrets appear inline in config or command output
