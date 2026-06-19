# Intel macOS Release Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the community fork and automatically rebuild each upstream stable Hermes Agent Release for Intel macOS.

**Architecture:** A scheduled GitHub Actions workflow first checks the upstream latest stable Release and exits when the corresponding community Release already exists. For a new Release, a separate `macos-15-intel` job checks out the exact upstream tag, builds unsigned x64 DMG and ZIP artifacts, validates every native executable, and publishes a prerelease only after all checks pass.

**Tech Stack:** GitHub REST API, GitHub Actions, Bash, Node.js/npm, Electron Builder, macOS `file`, `hdiutil`, and `shasum`.

---

### Task 1: Rename And Describe The Fork

**Files:**
- No repository files changed.

- [ ] **Step 1: Rename the repository through the GitHub API**

Send `PATCH /repos/evencj11/hermes-agent` with:

```json
{
  "name": "hermes-agent-desktop-intel-mac-rebuild",
  "description": "Unofficial Intel macOS desktop rebuilds of NousResearch/Hermes-Agent stable releases.",
  "homepage": "https://github.com/NousResearch/hermes-agent",
  "has_issues": true
}
```

Expected: HTTP 200 and `full_name` equals
`evencj11/hermes-agent-desktop-intel-mac-rebuild`.

- [ ] **Step 2: Update and verify the local fork remote**

```bash
git remote set-url fork https://github.com/evencj11/hermes-agent-desktop-intel-mac-rebuild.git
git ls-remote fork HEAD
```

Expected: the renamed repository resolves and returns a commit SHA.

### Task 2: Add Stable Release Automation

**Files:**
- Create: `.github/workflows/build-intel-macos-release.yml`

- [ ] **Step 1: Create the workflow with narrow permissions and deduplication**

Configure:

```yaml
on:
  schedule:
    - cron: "17 */6 * * *"
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: intel-macos-stable-release
  cancel-in-progress: false
```

The `discover` job must call
`repos/NousResearch/hermes-agent/releases/latest`, derive
`intel-macos-<upstream-tag>`, and output `should_build=false` when that tag
already has a Release in the fork.

- [ ] **Step 2: Add the Intel build job**

Use `runs-on: macos-15-intel`, check out
`NousResearch/hermes-agent` at the exact discovered tag, set up the Node version
declared by the project, run `npm ci`, and build with:

```bash
CSC_IDENTITY_AUTO_DISCOVERY=false npm run dist:mac -- --x64 --publish never
```

from `apps/desktop`. The explicit publish mode prevents Electron Builder from
inferring an implicit CI publish before the artifacts have been validated.

- [ ] **Step 3: Add fail-closed artifact validation**

Validate the app executable and every `.node`/`spawn-helper` native file with
`file`, require `x86_64`, reject `arm64`, run `hdiutil verify` on the DMG, and
write `SHA256SUMS.txt` for the DMG and ZIP. A missing file or failed assertion
must exit nonzero before the publishing step.

- [ ] **Step 4: Publish only validated outputs**

Use `gh release create` with `--prerelease`, attach the DMG, ZIP, and checksum
file, and include the upstream Release URL, upstream commit, architecture, and
unsigned/unnotarized warning in bilingual Release notes.

- [ ] **Step 5: Validate workflow syntax and policy**

Run:

```bash
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/build-intel-macos-release.yml", aliases: true)'
rg -n 'macos-15-intel|releases/latest|--x64|hdiutil verify|gh release create|--prerelease' .github/workflows/build-intel-macos-release.yml
git diff --check
```

Expected: YAML parses, every required control is present, and the diff check is
clean.

- [ ] **Step 6: Commit the workflow**

```bash
git add .github/workflows/build-intel-macos-release.yml
git commit -m "ci: automate Intel macOS stable release builds"
```

### Task 3: Publish To The Default Branch

**Files:**
- Modify fork branch history by fast-forwarding `main`.

- [ ] **Step 1: Push the implementation branch**

```bash
git push fork codex/intel-mac-release-automation
```

Expected: the branch contains the design, plan, and workflow commits.

- [ ] **Step 2: Fast-forward the fork default branch**

```bash
git push fork codex/intel-mac-release-automation:main
```

Expected: GitHub reports a fast-forward update and the workflow is visible on
the fork's default branch.

- [ ] **Step 3: Verify repository and workflow metadata**

Query the GitHub API and require the renamed full name, public visibility,
default branch `main`, and the workflow path on that branch.

### Task 4: Run End-To-End Verification

**Files:**
- No repository files changed.

- [ ] **Step 1: Dispatch the workflow manually**

Call the Actions workflow dispatch API for `main`, then record the new workflow
run ID.

- [ ] **Step 2: Wait for the run to finish**

Poll the workflow run until it reaches `completed`. Expected conclusion is
`success`; on failure, inspect job logs, correct the workflow, push the fix, and
dispatch again.

- [ ] **Step 3: Verify the public Release**

Require a prerelease with the derived `intel-macos-<upstream-tag>` tag and
exactly one DMG, one ZIP, and `SHA256SUMS.txt`. Send HEAD requests to each asset
URL and require HTTP 200 with nonzero content length.
