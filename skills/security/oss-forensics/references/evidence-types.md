# Evidence Types and Sources

This guide defines the taxonomy of evidence used in OSS security investigations.

## Evidence Types

| Type | Description | Examples |
|------|-------------|----------|
| `git_commit` | Raw git data from local repo or remote | Diff, commit message, metadata |
| `gh_api` | Data from current GitHub API | Issue comments, user profile, release notes |
| `web_archive` | Archived snapshots from Wayback Machine | Deleted README, past wiki states |
| `gh_archive` | Immutable event data from BigQuery | PushEvent, PullRequestEvent history |
| `ioc` | Indicators of Compromise | IP addresses, malicious domains, secrets |
| `analysis` | Derived evidence from reasoning or scripts | Discrepancy reports, timeline correlations |

## Observation States

- **Confirmed**: Evidence verified against multiple independent sources.
- **Suspected**: High confidence but single-source evidence.
- **Deleted**: Content found in archive/logs but missing from live environment.
- **Tampered**: Content exists but has been modified after initial creation (e.g., force-pushes).

## Evidence Identifier Format
`EV-[NUM]` - Sequential ID used for cross-referencing in reports.
