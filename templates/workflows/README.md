# Workflow templates

Portable, parameterized copies of this repo's GitHub Actions workflows, meant to
be dropped into another repository and customized. These files live **outside**
`.github/workflows/` on purpose — placing a `.yml` there activates it, and you
don't want two auto-mergers competing in this repo.

## `auto-merge.yml`

A generic version of [`../../.github/workflows/auto-merge.yml`](../../.github/workflows/auto-merge.yml).
It merges a PR — squash by default, configurable via `MERGE_METHOD` — once CI is
green, a review happened, and a required label is present. The full decision flow
and design notes are in [`../../AUTO_MERGE.md`](../../AUTO_MERGE.md).

### How to adopt it in another repo

1. **Copy** `auto-merge.yml` to `.github/workflows/auto-merge.yml` in the target
   repo.
2. **Edit the `on.workflow_run.workflows` list** — set it to the display `name:`
   of every CI workflow that must pass before merge. (This list cannot read env
   vars, so it's edited directly. Look for the `# 👈 EDIT` marker.)
3. **Edit the `env:` config block** on the `auto-merge` job. These drive the
   inline script. Note that `REQUIRED_LABEL` controls the in-script label check
   only — because a job-level `if:` cannot read `env`, you must *also* edit the
   `'reviewed'` literal in the job `if:` to match if you rename the label (that
   literal only gates the `pull_request: labeled` fast-trigger).

   | Variable | Purpose |
   |----------|---------|
   | `REQUIRED_LABEL` | Label that must be present to merge (default `reviewed`). Also update the matching `'reviewed'` literal in the job `if:`. |
   | `AUTO_MERGE_JOB_NAME` | Keep in sync with the job's `name:` so its own check runs are excluded. |
   | `MERGE_METHOD` | `squash`, `merge`, or `rebase` (repo must allow the chosen method). |
   | `REVIEW_BOTS` | Comma-separated bot logins whose activity counts as a review. |
   | `POLL_INTERVAL_MS` / `POLL_TIMEOUT_MS` | Poll cadence/limit for review/label triggers. |
   | `SETTLE_DELAY_MS` | Delay before reading state on non-`workflow_run` events. |
4. **If you rename the workflow** (the top-level `name:`), also update the
   `github.event.workflow_run.name != 'Auto-merge'` self-trigger guard in the
   job-level `if:` to match. (Renaming the *job* instead means updating
   `AUTO_MERGE_JOB_NAME` in `env:`.)
5. **Merge to the default branch.** `workflow_run` triggers are only active once
   the file is on the repo's default branch.

### Requirements in the target repo

- Create the `REQUIRED_LABEL` label (e.g. `reviewed`).
- The default `GITHUB_TOKEN` needs the `permissions:` already declared in the
  file. If your org sets "Read repository contents permission" as the default,
  the per-job `permissions:` block grants the rest.
- For the merge to succeed, repo settings must allow the chosen `MERGE_METHOD`
  (e.g. "Allow squash merging" enabled).
