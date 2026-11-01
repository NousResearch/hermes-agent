# Gemini CLI headless plan review

Use this when the user asks Gemini to review a repository plan or specification.
Check installed `gemini --help` first; flags and model aliases change by release.

## Read-only requirements

An analysis prompt and `--approval-mode plan` do not enforce read-only access.
Current [headless Plan Mode](https://geminicli.com/docs/cli/plan-mode/#non-interactive-execution)
can auto-approve plan transitions and switch to YOLO on exit. If no edits are
allowed, use tested deny policies or a sandbox with the repository mounted
read-only. Worktrees isolate changes but still permit writes.

## Flow

1. Set Hermes `terminal(workdir=...)` to the intended checkout. Use `read_file`
   for its instructions and `search_files` to locate the requested plan/spec.
2. Record `git status --short --branch` and `git diff` before the run, including
   pre-existing untracked files. Inspect any project configuration before trust.
3. With authentication and any required read-only containment already in place,
   run through `terminal`:

   ```bash
   gemini -p "Review @docs/plan.md. Return critical gaps, sequencing risks, test coverage, and decisions needed. Analyze only; do not edit files, exit planning to implement, or run mutating commands." \
     --approval-mode plan \
     --output-format json
   ```

   Replace the document path with the one actually requested. Do not pin a model
   unless needed; verify an explicitly requested model with installed help.
4. If the workspace is untrusted, stop and inspect it. Only add `--skip-trust`
   after the user has authorized trust for that exact workspace. Never silently
   broaden trust or approval mode as a retry strategy.
5. Check exit status and JSON `error` before reading `response`. When present,
   inspect `stats.models` to report the actual routed model.
6. Compare the post-run `git status --short --branch` and `git diff` with the
   baseline; inspect new untracked files too. Independently verify the review's
   claims against project files. Preserve earlier work and report any unexpected
   changes rather than discarding them.

## Pitfalls

- Cached Google authentication can work headlessly; an API key is not mandatory.
- A prompt is not a permissions boundary, and project trust is not tool approval.
- Git status cannot detect every write (for example, ignored files or files
  outside the checkout). Use actual filesystem restrictions when that matters.
- Do not treat a successful model response as proof that no files changed.
