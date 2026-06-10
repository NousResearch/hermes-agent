# Monica Stage D/E/F Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Monica trustworthy enough to take a tagged Slack mobile bug report through local fix, simulator proof, and draft PR creation without skipping proof gates.

**Architecture:** Monica runs in its own Hermes profile and owns a separate mobile-agent runtime root. Slack and Linear intake can be live before PR creation, but PR publishing stays disabled until real iOS and Android simulator proof exists. Stage F code may exist before Stage D/E pass, but `approved_pr` is not an operational mode until proof tooling and proof artifacts pass the audit below.

**Tech Stack:** Hermes plugin system, Slack gateway, Linear API, GitHub CLI/API, Codex worker process, React Native/Expo mobile repo, Xcode iOS Simulator, Android SDK emulator.

---

## Non-Negotiable Guardrails

- Monica must use `HERMES_HOME=/Users/ritik/.hermes/profiles/monica`.
- Chandler secrets, logs, runtime state, and Slack identity must not be reused.
- Monica must only act on direct messages or explicit mentions.
- Monica must not silently scan random channel chatter for action.
- Monica must not mark a run `done` without verification and proof artifacts when proof is required.
- Monica must not create a mobile repo PR unless the profile is explicitly moved to `approved_pr`.
- Monica must not create or push a PR to upstream Hermes.
- Monica must open draft PRs only, never ready-for-review PRs, until the user changes the policy.

## Current Truth Snapshot

- Hermes branch: `codex/monica-stage-f`.
- Monica runtime profile: `/Users/ritik/.hermes/profiles/monica/config.yaml`.
- Required current runtime mode before Stage D/E proof: `mobile_bug_agent.rollout_mode: local_fix_only`.
- iOS proof is currently blocked because `xcrun --find simctl` fails.
- Android proof is currently blocked because `adb` and `emulator` are not on `PATH`.
- The Stage F code path is allowed to remain in the codebase only as dormant, gated capability.

## File Map

- Modify: `plugins/mobile_bug_agent/cli.py`
  - CLI setup plan, doctor output, rollout-mode guardrails.
- Modify: `tests/plugins/mobile_bug_agent/test_cli.py`
  - Regression tests for ready-but-proof-warning setup plans.
- Inspect: `plugins/mobile_bug_agent/proof.py`
  - Proof runner and required artifact manifest behavior.
- Inspect: `plugins/mobile_bug_agent/simulator_proof.py`
  - iOS/Android simulator command execution and screenshot manifest.
- Inspect: `plugins/mobile_bug_agent/pr_publisher.py`
  - Draft PR gating and proof-section requirements.
- Inspect: `/Users/ritik/.hermes/profiles/monica/config.yaml`
  - Live Monica runtime mode and proof command.
- Inspect: `/Users/ritik/.hermes/profiles/monica/agents/monica`
  - Monica runtime workspace, state, and mobile worktrees.

---

## Task 1: Lock Runtime Back To Pre-F Mode

**Files:**
- Modify: `/Users/ritik/.hermes/profiles/monica/config.yaml`
- Inspect: `plugins/mobile_bug_agent/config.py`

- [x] **Step 1: Set Monica profile rollout mode to local fix only**

Run:

```bash
perl -0pi -e 's/mobile_bug_agent:\n  enabled: true\n  rollout_mode: approved_pr/mobile_bug_agent:\n  enabled: true\n  rollout_mode: local_fix_only/' /Users/ritik/.hermes/profiles/monica/config.yaml
```

Expected:

```yaml
mobile_bug_agent:
  enabled: true
  rollout_mode: local_fix_only
```

- [x] **Step 2: Verify live profile readiness**

Run:

```bash
HERMES_HOME=/Users/ritik/.hermes/profiles/monica uv run hermes mobile-bug-agent doctor --rollout-mode local_fix_only --json
```

Expected:

```json
{
  "failures": [],
  "ready": true,
  "rollout_mode": "local_fix_only",
  "warnings": [
    {"code": "ios_proof_tooling"},
    {"code": "android_proof_tooling"}
  ]
}
```

- [x] **Step 3: Stop condition**

Do not proceed to `approved_pr` while either warning is present:

```text
ios_proof_tooling
android_proof_tooling
```

---

## Task 2: Make Missing Simulator Proof Visible In Setup Plan

**Files:**
- Modify: `plugins/mobile_bug_agent/cli.py`
- Modify: `tests/plugins/mobile_bug_agent/test_cli.py`

- [x] **Step 1: Add regression test for ready rollout with proof warnings**

Add this test to `tests/plugins/mobile_bug_agent/test_cli.py`:

```python
def test_setup_plan_json_lists_proof_warning_steps_when_rollout_is_ready(capsys):
    exit_code = run_setup_plan_command(
        config=MonicaConfig(
            enabled=True,
            rollout_mode="approved_pr",
            dry_run=False,
            slack=SlackConfig(
                allowed_channels=("C123MOBILE",),
                bot_user_ids=("U123MONICA",),
                approver_user_ids=("U123APPROVER",),
            ),
            linear=LinearConfig(team_id="team-mobile"),
            repo=RepoConfig(url="git@github.com:acme/mobile.git"),
            verification=VerificationConfig(commands=("npm test",)),
            proof=ProofConfig(
                enabled=True,
                required_for_done=True,
                commands=("uv run --project \"$MONICA_HERMES_AGENT_ROOT\" python -m plugins.mobile_bug_agent.simulator_proof",),
                platform_order=("ios", "android"),
            ),
        ),
        target_rollout_mode="approved_pr",
        json_output=True,
        environ={
            "MONICA_SLACK_BOT_TOKEN": "xoxb-token",
            "MONICA_SLACK_APP_TOKEN": "xapp-token",
            "LINEAR_API_KEY": "lin_api_key",
            "GITHUB_TOKEN": "gh_token",
        },
        which=lambda command: None
        if command in {"xcrun", "xcodebuild", "adb", "emulator"}
        else f"/usr/bin/{command}",
        module_available=lambda name: True,
    )

    payload = json.loads(capsys.readouterr().out)
    step_ids = [step["id"] for step in payload["steps"]]
    assert exit_code == 0
    assert payload["ready"] is True
    assert step_ids == ["prepare_ios_simulator", "prepare_android_emulator", "rerun_doctor"]
    assert "Install full Xcode" in payload["steps"][0]["why"]
    assert "Install Android SDK" in payload["steps"][1]["why"]
```

- [x] **Step 2: Run test and verify failure before implementation**

Run:

```bash
scripts/run_tests.sh tests/plugins/mobile_bug_agent/test_cli.py
```

Expected before implementation:

```text
assert step_ids == ["prepare_ios_simulator", "prepare_android_emulator", "rerun_doctor"]
```

- [x] **Step 3: Implement setup-plan warning steps**

Change `_setup_plan_steps(report)` in `plugins/mobile_bug_agent/cli.py` so proof warnings are actionable even when `report.ready` is true:

```python
warning_codes = {str(issue.code) for issue in getattr(report, "warnings", ())}
proof_warning_codes = {
    "proof_commands_empty",
    "ios_proof_tooling",
    "android_proof_tooling",
}
if not failure_codes and not (warning_codes & proof_warning_codes):
    return []
```

Add these step ids:

```text
configure_simulator_proof
prepare_ios_simulator
prepare_android_emulator
rerun_doctor
```

- [x] **Step 4: Verify tests**

Run:

```bash
scripts/run_tests.sh tests/plugins/mobile_bug_agent/test_cli.py
```

Expected:

```text
117 tests passed
```

- [x] **Step 5: Verify live setup plan**

Run:

```bash
HERMES_HOME=/Users/ritik/.hermes/profiles/monica uv run hermes mobile-bug-agent setup-plan --rollout-mode local_fix_only --json
```

Expected step ids:

```json
["prepare_ios_simulator", "prepare_android_emulator", "rerun_doctor"]
```

- [x] **Step 6: Commit and push checkpoint**

Run:

```bash
git add plugins/mobile_bug_agent/cli.py tests/plugins/mobile_bug_agent/test_cli.py
git commit -m "Add Monica simulator setup plan steps"
git push ritikmdn codex/monica-stage-f
```

Expected:

```text
codex/monica-stage-f -> codex/monica-stage-f
```

---

## Task 3: Complete Stage D iOS Simulator Infrastructure

**Files:**
- Inspect: `/Applications/Xcode.app`
- Inspect: `/Library/Developer/CommandLineTools`
- Inspect: `/Users/ritik/.hermes/profiles/monica/config.yaml`
- Inspect: mobile worktree under `/Users/ritik/.hermes/profiles/monica/agents/monica/workspace/worktrees/`

- [ ] **Step 1: Install full Xcode**

Install Xcode so this path exists:

```text
/Applications/Xcode.app/Contents/Developer
```

This cannot be replaced by Command Line Tools. Monica needs `simctl`, iOS runtimes, Simulator.app, and `xcodebuild`.

- [ ] **Step 2: Select full Xcode**

Run:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

Expected:

```bash
xcode-select -p
```

returns:

```text
/Applications/Xcode.app/Contents/Developer
```

- [ ] **Step 3: Accept license and complete first launch**

Run:

```bash
sudo xcodebuild -license accept
xcodebuild -runFirstLaunch
```

Expected:

```bash
xcodebuild -version
```

prints an Xcode version, not a Command Line Tools error.

- [ ] **Step 4: Verify simulator control exists**

Run:

```bash
xcrun --find simctl
xcrun simctl list devices available
```

Expected:

```text
/Applications/Xcode.app/Contents/Developer/usr/bin/simctl
```

and at least one available iPhone simulator.

- [ ] **Step 5: Verify XcodeBuildMCP can see simulators**

Call XcodeBuildMCP:

```text
session_show_defaults
list_sims(enabled=true)
```

Expected:

```json
{"didError": false, "data": {"simulators": ["at least one available iPhone"]}}
```

Stop if `list_sims` still returns:

```text
unable to find utility "simctl"
```

---

## Task 4: Complete Stage D Android Emulator Infrastructure

**Files:**
- Inspect: `/Users/ritik/Library/Android/sdk`
- Inspect: shell profile that exports Android paths
- Inspect: `/Users/ritik/.hermes/profiles/monica/config.yaml`

- [ ] **Step 1: Install Android SDK platform tools and emulator**

Install Android Studio or Android command-line tools so these files exist:

```text
/Users/ritik/Library/Android/sdk/platform-tools/adb
/Users/ritik/Library/Android/sdk/emulator/emulator
/Users/ritik/Library/Android/sdk/cmdline-tools/latest/bin/sdkmanager
/Users/ritik/Library/Android/sdk/cmdline-tools/latest/bin/avdmanager
```

- [ ] **Step 2: Export Android paths**

Add the equivalent of this to the shell environment used by Hermes:

```bash
export ANDROID_HOME=/Users/ritik/Library/Android/sdk
export ANDROID_SDK_ROOT=/Users/ritik/Library/Android/sdk
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
```

- [ ] **Step 3: Install one emulator image**

Run:

```bash
sdkmanager "platform-tools" "emulator" "platforms;android-35" "system-images;android-35;google_apis;arm64-v8a"
```

Expected:

```text
done
```

- [ ] **Step 4: Create Monica AVD**

Run:

```bash
avdmanager create avd --name MonicaPixel --package "system-images;android-35;google_apis;arm64-v8a" --device "pixel_8"
```

Expected:

```bash
emulator -list-avds
```

contains:

```text
MonicaPixel
```

- [ ] **Step 5: Verify emulator boots**

Run:

```bash
emulator -avd MonicaPixel -no-snapshot -no-audio
adb wait-for-device
adb shell getprop sys.boot_completed
```

Expected:

```text
1
```

---

## Task 5: Complete Stage E iOS Proof On Actual Mobile App

**Files:**
- Inspect: mobile worktree `package.json`
- Inspect: `plugins/mobile_bug_agent/simulator_proof.py`
- Inspect: proof output under the run directory

- [ ] **Step 1: Run Monica proof helper for iOS**

Run:

```bash
MONICA_WORKTREE=/Users/ritik/.hermes/profiles/monica/agents/monica/workspace/worktrees/monica-ENG-880-bug-remove-all-hard-coded-tags-from-pdps-of-gyms-and-vouchers-for-now-many-of-th \
MONICA_PROOF_DIR=/private/tmp/monica-ios-proof \
MONICA_PROOF_PLATFORM_ORDER=ios \
uv run --project /Users/ritik/.hermes/hermes-agent python -m plugins.mobile_bug_agent.simulator_proof --timeout-seconds 600
```

Expected:

```text
monica-proof-manifest.json
```

exists in `/private/tmp/monica-ios-proof` and contains an iOS screenshot path.

- [ ] **Step 2: Inspect screenshot**

Open the screenshot path from the manifest.

Acceptance:

```text
The relevant Product Details page renders and the removed tags are absent.
```

- [ ] **Step 3: Attach proof to Monica run**

Expected run state:

```json
{
  "status": "verified",
  "proof": {
    "platform": "ios",
    "artifacts": ["screenshot path or recording path"]
  }
}
```

Stop if the screenshot is blank, on the wrong screen, or cannot prove the bug.

---

## Task 6: Complete Stage E Android Proof On Actual Mobile App

**Files:**
- Inspect: mobile worktree `package.json`
- Inspect: `plugins/mobile_bug_agent/simulator_proof.py`
- Inspect: proof output under the run directory

- [ ] **Step 1: Run Monica proof helper for Android**

Run:

```bash
MONICA_WORKTREE=/Users/ritik/.hermes/profiles/monica/agents/monica/workspace/worktrees/monica-ENG-880-bug-remove-all-hard-coded-tags-from-pdps-of-gyms-and-vouchers-for-now-many-of-th \
MONICA_PROOF_DIR=/private/tmp/monica-android-proof \
MONICA_PROOF_PLATFORM_ORDER=android \
uv run --project /Users/ritik/.hermes/hermes-agent python -m plugins.mobile_bug_agent.simulator_proof --timeout-seconds 600
```

Expected:

```text
monica-proof-manifest.json
```

exists in `/private/tmp/monica-android-proof` and contains an Android screenshot path.

- [ ] **Step 2: Inspect screenshot**

Open the screenshot path from the manifest.

Acceptance:

```text
The relevant Product Details page renders and the removed tags are absent.
```

- [ ] **Step 3: Attach proof to Monica run**

Expected run state:

```json
{
  "status": "verified",
  "proof": {
    "platform": "android",
    "artifacts": ["screenshot path or recording path"]
  }
}
```

Stop if the screenshot is blank, on the wrong screen, or cannot prove the bug.

---

## Task 7: Promote Monica To Stage F Draft PR Mode

**Files:**
- Modify: `/Users/ritik/.hermes/profiles/monica/config.yaml`
- Inspect: `plugins/mobile_bug_agent/pr_publisher.py`
- Inspect: `plugins/mobile_bug_agent/monica_loop.py`
- Inspect: mobile repo remote under Monica worktree

- [ ] **Step 1: Confirm D/E gates are green**

Run:

```bash
HERMES_HOME=/Users/ritik/.hermes/profiles/monica uv run hermes mobile-bug-agent doctor --rollout-mode approved_pr --json
```

Expected:

```json
{
  "failures": [],
  "ready": true,
  "rollout_mode": "approved_pr",
  "warnings": []
}
```

Stop if either proof warning remains.

- [ ] **Step 2: Enable approved PR mode**

Only after Step 1 is green, change Monica profile:

```yaml
mobile_bug_agent:
  rollout_mode: approved_pr
```

- [ ] **Step 3: Run one real tagged Slack dry-live bug flow**

Use a tagged Monica message that describes the mobile bug and includes screenshot context.

Acceptance:

```text
Monica creates or updates Linear, branches the mobile repo, applies the fix, runs verification, captures proof, and opens a draft PR.
```

- [ ] **Step 4: Verify draft PR contents**

The PR body must include:

```text
Linear issue link
Slack source link or channel/thread reference
Code summary
Verification commands and results
iOS proof artifact
Android proof artifact
Explicit "Draft" PR status
```

- [ ] **Step 5: Hard stop condition**

If the PR target repository is not the mobile app repo:

```text
https://github.com/simpleinsure/elixir_card_app.git
```

close the PR immediately and keep Monica in `local_fix_only`.

---

## Verification Suite

Run after every Hermes-side code checkpoint:

```bash
scripts/run_tests.sh tests/plugins/mobile_bug_agent tests/gateway/test_config.py tests/gateway/test_pre_gateway_dispatch.py tests/gateway/test_slack_approval_buttons.py tests/gateway/test_telegram_noise_filter.py tests/hermes_cli/test_plugin_cli_registration.py tests/hermes_cli/test_plugins_cmd.py
uv run ruff check plugins/mobile_bug_agent gateway tests/plugins/mobile_bug_agent
uv run python -m compileall plugins/mobile_bug_agent gateway hermes_cli
git diff --check
```

Expected:

```text
563 tests passed
All checks passed
compileall completes
git diff --check emits no output
```

## Completion Criteria

The full Stage F goal is complete only when all of these are true:

- Monica profile is isolated from Chandler.
- Monica only acts on DM or explicit mention.
- Monica can create Linear issues with context and attachments.
- Monica can create a local mobile worktree and implement a fix.
- Monica can run verification commands.
- Monica can run the actual mobile app in iOS Simulator.
- Monica can run the actual mobile app in Android Emulator.
- Monica captures proof screenshots or recordings for the relevant app screen.
- Monica attaches or references proof in state and Linear/PR output.
- Monica opens a draft PR only against the mobile app repo.
- There is no upstream Hermes PR created by the Monica runtime path.

Until the iOS and Android simulator checks pass, the correct status is:

```text
Stage C complete.
Stage D blocked on host simulator installation.
Stage E blocked on Stage D.
Stage F implemented as dormant gated code, not operationally enabled.
```
