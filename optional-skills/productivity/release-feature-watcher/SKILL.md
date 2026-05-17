---
name: release-feature-watcher
description: Use when the user wants to monitor a technical feature, bugfix, model/API update, changelog, release, package version, or PR and be notified only when a specific actionable condition is met.
version: 1.0.0
author: Abe Diaz (@abe238)
license: MIT
metadata:
  hermes:
    tags: [releases, changelogs, github, packages, models, apis, cron, watchdogs]
    related_skills: [resource-link-decision-framework, github-repo-management, hermes-agent]
    category: productivity
---

# Release Feature Watcher

## Overview

Use this skill to turn “tell me when this ships” into a quiet, verified, self-cleaning watcher for technical changes.

A watcher is not a reminder. A watcher is an automated readiness check. It should stay silent while nothing is actionable, alert only when the condition is met or the check is truly broken, include the next useful action, and remove itself after success when appropriate.

This skill is deliberately broad but not generic. It watches technical dependencies, tools, APIs, models, packages, changelogs, releases, bugs, and features that matter to a user’s workflow. It is not for open-ended “watch the web for interesting stuff” monitoring.

## When to Use

Use when the user says things like:

- “Watch this PR and tell me when it ships.”
- “Let me know when this bug is fixed in a release.”
- “Check daily if this feature has landed.”
- “Hold off installing until the new version includes this.”
- “Watch Anthropic/OpenAI/Gemini/OpenRouter for this model/API update.”
- “Monitor the changelog for this feature.”
- “Tell me when package X reaches version Y.”
- “Kill the cron job once it’s released.”

Do not use for:

- Generic news monitoring.
- Vague curiosity with no action path.
- Marketing/social monitoring unless tied to a technical readiness condition.
- Personal reminders, calendar events, or chores.
- Broad research digests. Use a research/news/watch skill instead.

## Core Doctrine

Every watcher needs six things:

1. **Source**: where to check.
2. **Condition**: the exact state that means “ready.”
3. **Baseline**: what state existed when the watcher was created.
4. **Action**: what the user should do once ready.
5. **Silence rule**: no update unless actionable or broken.
6. **Cleanup rule**: remove when done, expire when stale.

If you cannot state the condition precisely, do not create a watcher yet. Ask one concise clarifying question or park the resource instead.

## Watcher Classes

### 1. GitHub PR or issue watcher

Use for GitHub PRs, issues, releases, and release assets.

Common readiness conditions:

- `pr_merged`: PR `merged_at` is non-null.
- `pr_released`: PR merged and a public release was published after `merged_at`.
- `issue_closed`: issue state is closed.
- `issue_fixed_in_release`: issue closed or linked PR merged, then a release after the fix includes the issue/PR/feature.
- `release_contains_phrase`: release notes mention the requested feature, bug, issue number, PR number, or keyword.
- `asset_available`: a release has an installable asset for the user’s platform.

Default condition for “tell me when this ships”:

```text
PR merged + official non-draft release published after merge.
```

### 2. Changelog watcher

Use for product/app/API changelogs and release-note pages.

Common readiness conditions:

- changelog contains a required phrase.
- changelog mentions a model, feature, bugfix, deprecation, or limit change.
- changelog version is newer than a baseline version.
- changelog has both the feature phrase and a stable release marker.

Avoid alerting on any page diff. Page changes are evidence, not readiness.

### 3. Model/provider watcher

Use for Anthropic, OpenAI, Google/Gemini, xAI, OpenRouter, Nous, Hugging Face, local inference tools, and similar providers.

Common readiness conditions:

- model appears in official API docs.
- model appears in official pricing docs.
- model is available through a router or provider API.
- context window, tool use, structured output, vision/audio, or batch support is documented.
- pricing/limits are published and usable.

Default standard:

```text
Do not alert on marketing-only announcements unless the user asked for early awareness. Prefer official docs, API model lists, pricing pages, changelogs, package releases, or provider dashboards.
```

### 4. Package/version watcher

Use for npm, PyPI, Homebrew, Docker, AUR, GitHub releases, crates.io, and similar package sources.

Common readiness conditions:

- latest version is at least a target semver.
- latest version is newer than baseline and changelog includes a phrase.
- Docker tag is available.
- Homebrew formula/cask updated.
- installable binary exists for platform.

### 5. Docs/API behavior watcher

Use for documentation pages, API references, limit pages, and capability matrices.

Common readiness conditions:

- docs page contains exact field/method/model/limit.
- API reference adds a parameter or endpoint.
- limits page value changes across a numeric threshold.
- docs section changes and contains the desired phrase.

### 6. Security/fix watcher

Use for CVEs, critical bugs, exploit advisories, and fixes in tools the user runs.

Common readiness conditions:

- official advisory published.
- patched version released.
- package registry has version above vulnerable range.
- container/base image has patched digest/tag.

Security watchers may justify more urgent notifications, but still need clear evidence.

## Source Reliability

Grade evidence before alerting.

Strong evidence:

- official API docs
- official changelog/release notes
- GitHub release
- package registry
- provider pricing/limits page
- status/advisory page

Medium evidence:

- official blog post
- employee/founder tweet
- docs PR
- forum announcement by project maintainer

Weak evidence:

- random tweet
- rumor
- screenshot
- third-party blog without primary-source link

Alert only when evidence is strong enough for the requested action. If the user asked to install/update/use something, require install-ready evidence unless they explicitly allow early alerts.

## Default Behavior

When the user gives a watcher request:

1. Inspect the source.
2. Classify the watcher class.
3. Infer the actionable condition if obvious.
4. Ask one concise clarifying question only if the condition materially changes the watcher.
5. Create a quiet cron job.
6. Include the cron job ID in the job prompt/config so it can self-remove.
7. Verify the job exists.
8. Report a compact watcher card.

Default schedule: daily.

Default delivery: origin/current chat.

Default noise mode: quiet, only alert on success or persistent check failure.

Default cleanup: self-remove after success.

Default expiry:

- PR/release watcher: 365 days.
- changelog watcher: 180 days.
- package/model/provider watcher: 180 days.
- security watcher: no expiry unless superseded.

## Clarifying Questions

Prefer acting with a sensible default. Ask only when ambiguity changes the implementation.

Useful Telegram-friendly question:

```text
Watch for what?
1. Any update
2. Specific feature/fix mention
3. Official release after merge/fix
4. Installable version available
5. Model/API availability
```

Default for technical tools:

```text
specific feature/fix + installable or API-usable release
```

Useful noise question:

```text
How noisy?
1. Quiet: only when ready
2. Notify on major intermediate changes
3. Daily digest
```

Default:

```text
Quiet: only when ready.
```

## Cron Prompt Pattern

For LLM-backed cron jobs, make the prompt self-contained:

```text
Watcher: <short name>

Goal: Check whether <target> is ready.

Sources:
- <primary source URL>
- <secondary source URL if needed>

Baseline at creation:
- <state/version/date/hash>

Ready only if:
1. <condition 1>
2. <condition 2>
3. <condition 3>

If not ready:
- Deliver nothing.

If transient check failure:
- Deliver nothing unless failure appears persistent or blocks the watcher completely.

If ready:
- Send a concise message with evidence and next action.
- Remove this cron job. Current job id: <job_id>.

Next action if ready:
- <install/update/test/action>
```

## Script-Backed Watchers

Prefer deterministic `no_agent=true` cron scripts for common checks. They are cheaper, quieter, and less likely to hallucinate.

Recommended pattern:

1. Write a JSON config file under `~/.hermes/watchers/`.
2. Run a script that checks the condition.
3. Script prints nothing if not ready.
4. Script prints the final alert if ready.
5. The LLM-backed cron can then remove itself, or the script can instruct the user to remove by job ID.

This skill includes example scripts/templates for GitHub PR-to-release checks. Treat them as examples; adapt paths and install commands to the current Hermes instance.

## Watcher Card

After creating a watcher, send a compact card:

```text
Watcher created

Target: <project/source>
Watching for: <feature/fix/model/change>
Ready when: <condition>
Sources: <1-3 sources>
Schedule: <human time + timezone>
Quiet: yes/no
Expires: <date or rule>
Self-removes: yes/no
Job ID: <id>
```

## Success Alert

When ready, send:

```text
Ready: <feature/fix/change> appears shipped.

Evidence:
- <source fact 1>
- <source fact 2>
- <source fact 3>

Release/docs/package: <URL>
Next step: <install/update/test/action>
Watcher removed: <job_id>
```

If evidence is not strong enough for action, do not mark ready. Send only if the user requested intermediate updates.

## Failure Handling

Do not notify on every failed check.

Recommended policy:

- One transient network/API failure: silent.
- Two consecutive failures: silent unless the watcher is urgent.
- Three consecutive failures: notify with exact problem and next repair.
- 404/private/auth failure: notify immediately if it cannot self-recover.
- Rate-limit: back off and notify only if it prevents future checks.

## Registry

For public use, keep a local registry when practical:

```text
~/.hermes/watchers/release-feature-watchers.json
```

Minimum fields:

```json
{
  "id": "short-id",
  "name": "Terax Markdown Preview release watcher",
  "kind": "github_pr_release",
  "sources": ["https://github.com/crynta/terax-ai/pull/142"],
  "condition": "PR merged and public release published after merge",
  "created_at": "2026-05-16T00:00:00Z",
  "expires_at": "2027-05-16T00:00:00Z",
  "cron_job_id": "...",
  "status": "active",
  "next_action": "Install/update Terax"
}
```

For users with an Obsidian or notes system, optionally maintain a human-readable watcher registry. Do not assume one exists.

## Install Path Hints

When the watcher fires, include the best next action:

- GitHub release asset: pick the correct OS/architecture asset.
- Homebrew: suggest a Homebrew upgrade command for the specific formula/cask.
- npm: suggest the relevant package-manager update command for the specific package/version.
- PyPI: suggest the relevant Python package update command for the specific package/version.
- Docker: suggest pulling the specific image tag or digest.
- AUR: `yay -Syu <pkg>`.
- AppImage/deb/rpm/dmg/zip: link the exact release asset.
- API/model provider: summarize docs/pricing and ask before changing config.

## Examples

### GitHub PR shipped in release

User:

```text
Watch this PR and tell me when I should install the new version.
```

Condition:

```text
PR merged + latest public release published after merge + release has installable asset.
```

### Model available via API

User:

```text
Tell me when Anthropic releases Sonnet X in the API with pricing.
```

Condition:

```text
Official model/API docs list the model and official pricing page includes it.
```

### Changelog bugfix

User:

```text
Watch the changelog until they fix the markdown preview bug.
```

Condition:

```text
Official changelog includes markdown preview fix phrase in a new version after baseline.
```

### Package version

User:

```text
Tell me when package foo reaches 2.4 and the migration bug is fixed.
```

Condition:

```text
Registry version >= 2.4.0 and changelog/release notes mention the migration fix.
```

## Common Pitfalls

1. **Creating vague reminders.** “Check this PR daily” is not enough. Define readiness.

2. **Alerting on weak evidence.** A founder tweet can be useful context, but it is not install/API-ready evidence unless the user asked for early signals.

3. **Spamming no-change updates.** Quiet is the default. No news means no message.

4. **Forgetting self-removal.** If the watcher is one-shot, include the job ID and removal instruction in the cron prompt.

5. **Ignoring timezone.** Translate the user’s requested local time into the scheduler’s timezone and report both when useful.

6. **Using an LLM for deterministic checks.** Prefer scripts for GitHub/package/version checks; use LLM judgment only for ambiguous changelogs/docs.

7. **No install path.** A watcher that says “released” without the next action is only half done.

8. **Letting stale watchers accumulate.** Add expiry or periodic cleanup.

## Verification Checklist

- [ ] Source inspected.
- [ ] Watcher class identified.
- [ ] Readiness condition explicit and actionable.
- [ ] Baseline recorded.
- [ ] Schedule and timezone correct.
- [ ] Quiet behavior specified.
- [ ] Success alert includes evidence and next action.
- [ ] Self-removal or expiry specified.
- [ ] Cron job created and listed/verified.
- [ ] Job ID included in prompt/config if self-removal is required.
