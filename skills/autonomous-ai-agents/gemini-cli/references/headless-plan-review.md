# Gemini CLI headless plan review pattern

Use this when asked to have Gemini scan a repository plan/spec without editing files.

## Verified flow

1. Locate the repository in the intended project checkout (for example, `$HOME/<repo>`). Do not create or treat hidden tool directories such as `.hermes/` or `.codex/` as active working repo locations unless explicitly requested.
2. Check repo state and find plan/spec docs:
   ```bash
   git status --short --branch
   find docs -maxdepth 2 -type f \( -iname '*plan*.md' -o -iname '*spec*.md' -o -iname '*roadmap*.md' \)
   ```
3. Run Gemini in headless plan mode with an explicit `workdir`. If the repo has not been trusted interactively, include `--skip-trust` so non-interactive automation can proceed:
   ```bash
   gemini --skip-trust -m gemini-3-pro-preview \
     -p "You are a senior engineering reviewer. Scan @docs/plan.md. Do not edit files. Return strongest parts, critical gaps/blockers, sequencing risks, data/vendor/compliance risks, concrete next steps, and owner decisions." \
     --approval-mode plan \
     --output-format json
   ```
4. Parse the JSON response and inspect `stats.models` to verify the actual routed model. In one observed run, requesting `gemini-3-pro-preview` routed to `gemini-3.1-pro-preview`; report the actual model rather than assuming the requested alias.
5. Verify no edits occurred:
   ```bash
   git status --short --branch
   ```

## Pitfalls

- Without `--skip-trust` or `GEMINI_CLI_TRUST_WORKSPACE=true`, headless runs can fail with a trusted-folder error and may override `--approval-mode plan` to `default`.
- Plan review should use `--approval-mode plan` and a prompt that says `Do not edit files`.
- Do not trust Gemini's self-report alone; verify `git status` after the run.
