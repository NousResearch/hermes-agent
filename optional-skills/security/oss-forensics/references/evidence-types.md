# Evidence Types Reference

Taxonomy of all evidence types, IOC types, GitHub event types, and observation types
used in OSS forensic investigations.

---

## Evidence Source Types

| Type | Description | Example Sources |
|------|-------------|-----------------|
| `git` | Data from local git repository analysis | `git log`, `git fsck`, `git reflog`, `git blame` |
| `gh_api` | Data from GitHub REST API responses | `/repos/.../commits`, `/repos/.../pulls`, `/repos/.../events` |
| `gh_archive` | Data from GitHub Archive (BigQuery) | `githubarchive.month.*` BigQuery tables |
| `web_archive` | Archived web pages from Wayback Machine | CDX API results, `web.archive.org/web/...` snapshots |
| `ioc` | Indicator of Compromise from any source | Extracted from vendor reports, git history, network traces |
| `analysis` | Derived insight from cross-source correlation | "SHA present in archive but absent from API" |
| `vendor_report` | External security vendor or researcher report | CVE advisories, blog posts, NVD records |
| `manual` | Manually recorded observation by investigator | Notes on behavioral patterns, timeline gaps |

---

## IOC Types

| Type | Description | Example |
|------|-------------|---------|
| `COMMIT_SHA` | A git commit hash linked to malicious activity | `abc123def456...` |
| `FILE_PATH` | A suspicious file inside the repository | `src/utils/crypto.js`, `dist/index.min.js` |
| `API_KEY` | An API key accidentally committed | `AKIA...` (AWS), `ghp_...` (GitHub PAT) |
| `SECRET` | A generic secret / credential | Database password, private key blob |
| `IP_ADDRESS` | A C2 server or attacker IP | `192.0.2.1` |
| `DOMAIN` | A malicious or suspicious domain | `evil-cdn.io`, typosquatted package registry domain |
| `PACKAGE_NAME` | A malicious or squatted package name | `colo-rs` (typosquatting `color`), `lodash-utils` |
| `ACTOR_USERNAME` | A GitHub handle linked to the attack | `malicious-bot-account` |
| `MALICIOUS_URL` | A URL to a malicious resource | `https://evil.example.com/payload.sh` |
| `WORKFLOW_FILE` | A suspicious CI/CD workflow file | `.github/workflows/release.yml` |
| `BRANCH_NAME` | A suspicious branch | `refs/heads/temp-fix-do-not-merge` |
| `TAG_NAME` | A suspicious git tag | `v1.0.0-security-patch` |
| `RELEASE_NAME` | A suspicious release | Release with no associated tag or changelog |
| `OTHER` | Catch-all for unclassified IOCs | — |

---

## GitHub Archive Event Types (12 Types)

| Event Type | Forensic Relevance |
|------------|-------------------|
| `PushEvent` | Core: `payload.distinct_size=0` with `payload.size>0` → force push. `payload.before`/`payload.head` shows rewritten history. |
| `PullRequestEvent` | Detects deleted PRs, rapid open→close patterns, PRs from new accounts |
| `IssueEvent` | Detects deleted issues, coordinated labeling, rapid closure of vulnerability reports |
| `IssueCommentEvent` | Deleted comments, rapid activity bursts |
| `WatchEvent` | Star-farming campaigns (coordinated starring from new accounts) |
| `ForkEvent` | Unusual fork patterns before malicious commit |
| `CreateEvent` | Branch/tag creation: signals new release or code injection point |
| `DeleteEvent` | Branch/tag deletion: critical — often used to hide traces |
| `ReleaseEvent` | Unauthorized releases, release artifacts modified post-publish |
| `MemberEvent` | Collaborator added/removed: maintainer compromise indicator |
| `PublicEvent` | Repository made public (sometimes to drop malicious code briefly) |
| `WorkflowRunEvent` | CI/CD pipeline executions: workflow injection, secret exfiltration |

---

## Evidence Verification States

| State | Meaning |
|-------|---------|
| `unverified` | Collected from a single source, not cross-referenced |
| `single_source` | The primary source has been confirmed directly (e.g., SHA resolves on GitHub), but no second source |
| `multi_source_verified` | Confirmed from 2+ independent sources (e.g., GH Archive AND GitHub API both show the same event) |

Only `multi_source_verified` evidence may be cited as fact in validated hypotheses.
`unverified` and `single_source` evidence must be labeled `[UNVERIFIED]` or `[SINGLE-SOURCE]`.

---

## Observation Types (Patterned after RAPTOR)

| Type | Description |
|------|-------------|
| `CommitObservation` | Specific commit SHA with metadata (author, date, files changed) |
| `ForceWashObservation` | Evidence that commits were force-erased from a branch |
| `DanglingCommitObservation` | SHA present in git object store but unreachable from any ref |
| `IssueObservation` | A GitHub issue (current or archived) with title, body, timestamp |
| `PRObservation` | A GitHub PR (current or archived) with diff summary, reviewers |
| `IOC` | A single Indicator of Compromise with context |
| `TimelineGap` | A period with unusual absence of expected activity |
| `ActorAnomalyObservation` | Behavioral anomaly for a specific GitHub actor |
| `WorkflowAnomalyObservation` | Suspicious CI/CD workflow change or unexpected run |
| `CrossSourceDiscrepancy` | Item present in one source but absent in another (strong deletion indicator) |
