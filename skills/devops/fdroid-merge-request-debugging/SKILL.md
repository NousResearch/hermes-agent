---
name: fdroid-merge-request-debugging
description: Use when preparing, debugging, or responding to F-Droid fdroiddata merge requests, especially app inclusion metadata, reproducible Binaries checks, checkupdates/autoupdate fixes, reviewer discussion resolution, and GitLab pipeline triage.
version: 1.0.1
metadata:
  hermes:
    tags: [fdroid, android, reproducible-builds, gitlab, release]
---

# F-Droid Merge Request Debugging

Use this skill for fdroiddata merge requests where the goal is a reviewer-ready
app inclusion or update. Start from the live GitLab MR and the real pipeline
state, not from stale screenshots.

## Preflight

1. Identify the fdroiddata worktree, app id, MR iid, source branch, and upstream
   app repo/tag/commit.
2. Check both worktrees before editing and record whether either tree already
   has unrelated local changes:

```powershell
git -C C:\path\to\fdroiddata status --short --branch
git -C C:\path\to\app status --short --branch
```

3. Pull live reviewer discussions and current MR metadata:

```powershell
$glab = "C:\Users\Ady\AppData\Local\Programs\glab\glab.exe"
& $glab api 'projects/fdroid%2Ffdroiddata/merge_requests/<iid>/discussions?per_page=100'
& $glab api 'projects/fdroid%2Ffdroiddata/merge_requests/<iid>'
```

4. Record every unresolved resolvable discussion id, comment body, file, line,
   and suggested change. Do not resolve threads until the branch is pushed and
   CI confirms the fix.
5. If screenshots were provided, treat them as hints only. Re-query GitLab
   before editing because labels, discussions, pipelines, and suggested changes
   can change quickly during F-Droid review.

## Metadata Rules

- Keep edits scoped to `metadata/<appid>.yml`.
- Prefer reviewer suggestions when they are valid F-Droid metadata.
- Use full commit hashes in `Builds.commit`, not tags.
- Set `subdir` to the Gradle module that produces the build directory.
- Avoid custom `output` unless F-Droid cannot discover the APK.
- Keep `Binaries` and `AllowedAPKSigningKeys` for reproducible builds when the
  upstream release publishes a comparable APK.
- Do not run `fdroid rewritemeta` casually; it can rewrite formatting and create
  unrelated churn. If CI requests it, inspect the diff before committing.
- If CI's `fdroid rewritemeta` job fails with a tiny formatter diff, apply the
  formatter output exactly even when local `fdroid lint` reports a cosmetic
  warning. The GitLab formatter job is the source of truth for merge readiness.
  Hermes Agent hit this with a `Binaries: ` multiline scalar where the formatter
  required a trailing space after the colon.
- Use unrestricted `UpdateCheckMode: Tags` unless there is a strong reason to
  filter tags. If version codes cannot be derived from a single regex capture,
  add a small upstream version metadata file and use `UpdateCheckData` to read
  both `versionCode` and `versionName` from each tag.

Hermes Agent uses:

```yaml
UpdateCheckMode: Tags
UpdateCheckData: fdroid/com.nousresearch.hermesagent.version|versionCode=(\d+)|.|versionName=(.*)
```

## Reproducible APK Checks

Before pushing a binary metadata update, verify the release APK independently:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\tmp\app.apk
& "$AndroidSdk\build-tools\35.0.0\apksigner.bat" verify --print-certs .\tmp\app.apk
```

For Chaquopy-based Android apps, also check that generated Python assets are
stable:

- no `*.dist-info/direct_url.json`
- inner `requirements-*.imy` entries use `ZIP_STORED`
- inner ZIP timestamps are normalized when the app normalizer requires it
- `build.json` references the actual `requirements-*.imy` SHA-1

If `fdroid build` compiles the source successfully but then fails while
retrieving `Binaries`, check the upstream release asset URL before changing
metadata. A `404` for a URL like
`https://github.com/<owner>/<repo>/releases/download/v%v/<apk-name>.apk` means
the source build may already be good and the failure is the missing signed
reference APK. Fix the upstream GitHub Release workflow or upload process,
verify the APK URL returns `200 OK`, then retry or retrigger the GitLab MR
pipeline. Do not leave the MR red when the only failure is a delayed release
asset upload.

For GitHub release workflows that remain queued on hosted runners, first check
the job labels and whether a repo variable such as `HERMES_ANDROID_RUNNER`
overrides `runs-on`. A temporary self-hosted runner can unblock a release asset
upload only when the repo secrets stay in GitHub Actions and the runner is
ephemeral, non-root on Linux, and deregisters after one job. Remove any temporary
runner override variable immediately after the release job completes.

## Local Validation

Run the cheap local gates before commit:

```powershell
git diff --check
fdroid lint <appid>
fdroid checkupdates --allow-dirty <appid>
```

On Windows, fdroidserver may fail after successful `checkupdates` processing
while writing status output if local config expects Unix tools such as `rsync`.
Treat that as an environment issue only if verbose output already proves the
app version was detected and the GitLab `checkupdates` job later passes.

Local lint can also be weaker than the MR pipeline. Always monitor GitLab before
claiming the MR is ready.

## Commit, Push, and Pipeline

Use noninteractive git commands. If Git opens a browser or credential selector,
configure the repo or host before retrying; do not rely on manual clicks.

```powershell
git add metadata/<appid>.yml
git commit -m "Address <app> review comments"
git push origin <branch>
```

For GitHub source repos on this Windows machine, prefer the GitHub CLI
credential helper over Git Credential Manager prompts:

```powershell
gh auth status
gh auth setup-git --hostname github.com
git config --local --unset-all credential.helper 2>$null
git config --local --add credential.helper ""
git config --local --add credential.helper "!gh auth git-credential"
git config --local credential.interactive never
git config --local credential.useHttpPath true
```

For GitLab/fdroiddata pushes, do not assume the cached Git credential is also a
REST API token. If `git push` works but GitLab API calls return `401
Unauthorized`, keep the branch and pipeline updated and give the exact reviewer
reply text for the user to paste.

Monitor the source-project pipeline, then inspect failed job logs directly:

```powershell
& $glab api 'projects/<user>%2Ffdroiddata/pipelines?ref=<branch>&per_page=5'
& $glab api 'projects/<user>%2Ffdroiddata/pipelines/<pipeline-id>/jobs?per_page=100'
& $glab api 'projects/<user>%2Ffdroiddata/jobs/<job-id>/trace'
```

Only declare the MR ready when the relevant jobs pass: `fdroid build`,
`check apk` when `Binaries` is present, `checkupdates`, `fdroid lint`,
`fdroid rewritemeta`, schema validation, source checks, git redirect, and tools
checks.

## Reviewer Response

After CI is green:

1. Update the MR description with current version, version code, source commit,
   metadata commit, binary URL, binary digest, signing key, autoupdate behavior,
   and latest passing pipeline.
2. Reply directly to reviewer questions with the concrete fix and pipeline URL.
3. Resolve only the threads whose requested changes are actually handled.
4. Re-query the MR to verify:

```powershell
unresolved_resolvable = 0
pipeline = success
detailed_merge_status = mergeable
blocking_discussions_resolved = True
```

If a maintainer-owned label such as `waiting-on-response` does not change via
API, leave a clear comment and rely on the resolved threads plus green pipeline.

When a reviewer asks whether `UpdateCheckMode: Tags` is limited to a specific
version line, answer from the metadata: unrestricted `Tags` is not limited to
`0.13.x`; any future semver tag can be detected when `UpdateCheckData` reads
`versionName` and `versionCode` from the upstream tag.

## Completion Audit

Before final handoff, map every reviewer note and user request to evidence:

- metadata diff for each reviewer suggestion
- local validation output
- binary digest and signing key evidence when applicable
- GitLab pipeline URL and job statuses
- discussion ids resolved or intentionally left open
- clean worktree status for fdroiddata and the upstream app repo
