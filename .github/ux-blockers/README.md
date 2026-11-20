# Hermes UX blocker system

Engineering severity measures the defect. UX priority measures whether a user
can complete a supported core journey. This system proposes UX impact, keeps
fix relationships visible and holds stable promotion for confirmed blockers.
The master proposal is [#157333](https://github.com/NousResearch/hermes-agent/issues/157333).

## Workflow

1. Reporter selects a core surface and journey, describes what stopped and how
   they recovered. Optional structured fields make the proposal reproducible.
2. Actions proposes UX P0 or P1 and adds `ux:needs-review`. Unknown evidence
   remains unclassified. Engineering severity is unchanged.
3. A maintainer checks the supported path, version and evidence. A confirmation
   records an acknowledgement on the policy issue and adds `ux:confirmed` as its visible index.
4. GitHub closing-PR relationships appear on the bot assessment and refresh when
   PRs change. A merged PR earns no automatic recovery or release-clearance claim.
5. The release owner dispositions the exact candidate. Acceptance and the final
   publication controller both check the live ledger.

| UX priority | Recovery boundary | Stable decision |
| --- | --- | --- |
| P0 | Supported core journey blocked or false success, without safe discoverable recovery in its normal surface | Hold affected paths until fix/recovery evidence or an explicit candidate disposition |
| P1 | Serious disruption with a clear safe recovery in the normal surface | Prioritize, with no automatic hold |

An ordinary command belongs to the Terminal experience. Requiring Desktop,
Telegram or Discord users to move to Terminal, edit files, reinstall or seek
another assistant leaves a recovery gap. Supported journeys are setup, provider
sign-in, first reply, returning to a conversation, and update/recovery.

## Maintainer commands

Post a standalone **issue comment**, using the full immutable candidate SHA and
a public-safe evidence URL. Commands are read from live GitHub comments, never
from a PR body or workflow event's claimed permission. Current repository
`write`, `maintain` or `admin` permission and a human account are required.

```text
/ux confirm <40-character SHA> <https evidence URL> <supported path and observed failure>
/ux resolve fixed <SHA> <https evidence URL> <user-visible result and recovery>
/ux resolve recovered <SHA> <https evidence URL> <safe recovery the user can discover>
/ux resolve unaffected <SHA> <https evidence URL> <why the admitted release excludes this path>
/ux resolve exception <SHA> <https evidence URL> <release decision and supported recovery>
/ux duplicate #<confirmed canonical issue number>
```

The latest confirmation supersedes earlier resolutions. Resolution applies only
to its exact candidate and must follow that confirmation. `fixed` additionally
requires at least one GitHub-linked merged PR whose merge commit is an ancestor
of the candidate. A maintainer still decides whether that fix and its evidence
cover the reported failure. A duplicate clears only through its confirmed
canonical issue. Cycles and missing canonical records hold promotion.

Do not edit authority commands to change a decision. Post a new command. An
edited command invalidates its authority. The bot records the source issue, confirmation ID and digest on the canonical
policy issue (#157333), as well as acknowledging the report. The gate always
reads that ledger, so deleting a source command or removing a label leaves the
confirmed report discoverable. Leave the canonical ledger and its bot records
in place; labels are a projection of that ledger.
The assessment labels alone confer no release authority.

For an older report, post `/ux assess` followed by the same headings as the bug
form. The trusted assessment replaces only the classification input.

```text
/ux assess
### UX surface
Desktop
### UX journey
Update / recovery
### UX progress
Blocked or falsely reports success
### UX recovery
Terminal or another surface required
### User Journey and Recovery (optional)
The app offered only an installer. Recovery required a separate CLI update.
```

For Terminal, choose `Clear action in the same surface` when an ordinary safe
command completes recovery. Unknown or incomplete fields never become a
confirmed blocker automatically. Excluded optional features remain outside this
core-journey rubric.

## Operator commands and adoption

All commands accept `--repository owner/repo` or use `GITHUB_REPOSITORY`.
Authenticate `gh` with the intended repository account before running them.

```sh
GITHUB_REPOSITORY=NousResearch/hermes-agent just ux-triage 123737
GITHUB_REPOSITORY=NousResearch/hermes-agent just ux-gate FULL_CANDIDATE_SHA
GITHUB_REPOSITORY=NousResearch/hermes-agent just ux-report
```

Triage needs issues write and PR read. Gate/report are read-only and require
issues/PR reads plus repository metadata permission. JSON output and Actions
summaries retain decisions and fixing-PR identities. Missing API pages or
permission failures stop the gate. A human's evidence URL is an attestation,
not an automatically executed test or a guarantee that a linked run passed.

To adopt, review/merge the workflow and release-controller changes, allow their
standard Actions-token permissions, and nominate the release owner who confirms
and dispositions blockers. The existing policy issue #157333 is the canonical
bot confirmation ledger. Fork adopters create a ledger issue and set repository
variable `UX_LEDGER_ISSUE` to its number; local commands accept `--ledger-issue`
or the same environment variable. Closing the ledger does not remove its
records. Deleting the ledger or editing bot records is unsupported and holds
promotion on API/integrity failure. The first triage run creates four impact/index
labels. Manually assess the first-wave issues before confirming any of them.
The historical dataset does not enroll issues or block releases. No external
provider, extra secret, or bot service is needed. Existing release-signing and
publication approvals remain in force.

Reverting the implementation removes enforcement. Labels and comments remain as
history; reverting does not undo a published release or remove issue metadata.

## Hermes first-wave data

The [readable first-wave report](hermes-first-wave-2026-11-18.md) and
[versioned sample](hermes-first-wave-2026-11-18.json) records the bounded
Oct 1–Nov 18, 2026 Bangkok audit. Public issue/PR metadata was refreshed Nov 18.
It contains **10 open report-evidenced problem clusters** — nine first filed in
the window plus the September macOS carry-in — and **7 qualifying merged
clusters across 8 product PRs**. Two recovery-ambiguous merged candidates are
excluded. All 14 reports underlying the ten open clusters remained open at the
refresh. Six previously audited fix PRs remain open; one additional same-cluster
open fix candidate is listed separately.

This is a lower bound from a bounded screen, not a census. Issue lifecycle,
reported recovery and exact-candidate proof are separate fields. Every cluster
explicitly records current release-blocker assessment as `not_assessed`.
`just ux-report` refreshes public lifecycle/SHAs for this same sample without
silently reclassifying it, counting a process PR as a product fix, or treating
an open issue as proof of current release failure.

The earlier [OpenClaw and Hermes audit](https://github.com/100yenadmin/hermes-agent-for-upstream-PR-only/blob/6ac3291a188161c3f658a0f7f011a9660a2ced7d/ux-release-gate-evidence.md)
found 185 linked OpenClaw fixes reaching preview before stable and 182 linked
reports with ClawSweeper labels before an associated fix merged. These overlap;
they are not additive, causal prevention counts or recorded release holds.
