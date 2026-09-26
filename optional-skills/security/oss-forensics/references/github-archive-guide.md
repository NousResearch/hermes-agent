# GitHub Archive Query Guide (BigQuery)

GitHub Archive records every public event on GitHub as immutable JSON records. This data is accessible via Google BigQuery and is the most reliable source for forensic investigation — events cannot be deleted or modified after recording.

## Public Dataset

- **Project**: `githubarchive`
- **Tables**: `day.YYYYMMDD`, `month.YYYYMM`, `year.YYYY`
- **Cost**: $6.25 per TiB scanned. Always run dry runs first.
- **Access**: Requires a Google Cloud account with BigQuery enabled. Free tier includes 1 TiB/month of queries.

---

## The 12 GitHub Event Types

| Event Type | What It Records | Forensic Value |
|------------|-----------------|----------------|
| `PushEvent` | Commits pushed to a branch | Force-push detection, commit timeline, author attribution |
| `PullRequestEvent` | PR opened, closed, merged, reopened | Deleted PR recovery, review timeline |
| `IssuesEvent` | Issue opened, closed, reopened, labeled | Deleted issue recovery, social engineering traces |
| `IssueCommentEvent` | Comments on issues and PRs | Deleted comment recovery, communication patterns |
| `CreateEvent` | Branch, tag, or repository creation | Suspicious branch creation, tag timing |
| `DeleteEvent` | Branch or tag deletion | Evidence of cleanup after compromise |
| `MemberEvent` | Collaborator added or removed | Permission changes, access escalation |
| `PublicEvent` | Repository made public | Accidental exposure of private repos |
| `WatchEvent` | User stars a repository | Actor reconnaissance patterns |
| `ForkEvent` | Repository forked | Exfiltration of code before cleanup |
| `ReleaseEvent` | Release published, edited, deleted | Malicious release injection, deleted release recovery |
| `WorkflowRunEvent` | GitHub Actions workflow triggered | CI/CD abuse, unauthorized workflow runs |

---

## Query Templates

### Basic: All Events for a Repository

```sql
SELECT
  created_at,
  type,
  actor.login,
  repo.name,
  payload
FROM
  `githubarchive.day.20240101`  -- Adjust date
WHERE
  repo.name = 'owner/repo'
  AND type IN ('PushEvent', 'DeleteEvent', 'MemberEvent')
ORDER BY
  created_at ASC
```

### Force-Push Detection

Force-pushes produce PushEvents where commits are overwritten. Key indicators:
- `payload.distinct_size = 0` with `payload.size > 0` → commits were erased
- `payload.before` contains the SHA before the rewrite (recoverable)

```sql
SELECT
  created_at,
  actor.login,
  JSON_EXTRACT_SCALAR(payload, '$.before') AS before_sha,
  JSON_EXTRACT_SCALAR(payload, '$.head') AS after_sha,
  JSON_EXTRACT_SCALAR(payload, '$.size') AS total_commits,
  JSON_EXTRACT_SCALAR(payload, '$.distinct_size') AS distinct_commits,
  JSON_EXTRACT_SCALAR(payload, '$.ref') AS branch_ref
FROM
  `githubarchive.month.*`
WHERE
  _TABLE_SUFFIX BETWEEN '202401' AND '202403'
  AND type = 'PushEvent'
  AND repo.name = 'owner/repo'
  AND CAST(JSON_EXTRACT_SCALAR(payload, '$.distinct_size') AS INT64) = 0
ORDER BY
  created_at ASC
```

### Deleted Branch/Tag Detection

```sql
SELECT
  created_at,
  actor.login,
  JSON_EXTRACT_SCALAR(payload, '$.ref') AS deleted_ref,
  JSON_EXTRACT_SCALAR(payload, '$.ref_type') AS ref_type
FROM
  `githubarchive.month.*`
WHERE
  _TABLE_SUFFIX BETWEEN '202401' AND '202403'
  AND type = 'DeleteEvent'
  AND repo.name = 'owner/repo'
ORDER BY
  created_at ASC
```

### Collaborator Permission Changes

```sql
SELECT
  created_at,
  actor.login,
  JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
  JSON_EXTRACT_SCALAR(payload, '$.member.login') AS member
FROM
  `githubarchive.month.*`
WHERE
  _TABLE_SUFFIX BETWEEN '202401' AND '202403'
  AND type = 'MemberEvent'
  AND repo.name = 'owner/repo'
ORDER BY
  created_at ASC
```

### CI/CD Workflow Activity

```sql
SELECT
  created_at,
  actor.login,
  JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
  JSON_EXTRACT_SCALAR(payload, '$.workflow_run.name') AS workflow_name,
  JSON_EXTRACT_SCALAR(payload, '$.workflow_run.conclusion') AS conclusion,
  JSON_EXTRACT_SCALAR(payload, '$.workflow_run.head_sha') AS head_sha
FROM
  `githubarchive.month.*`
WHERE
  _TABLE_SUFFIX BETWEEN '202401' AND '202403'
  AND type = 'WorkflowRunEvent'
  AND repo.name = 'owner/repo'
ORDER BY
  created_at ASC
```

### Actor Activity Profiling

```sql
SELECT
  type,
  COUNT(*) AS event_count,
  MIN(created_at) AS first_event,
  MAX(created_at) AS last_event
FROM
  `githubarchive.month.*`
WHERE
  _TABLE_SUFFIX BETWEEN '202301' AND '202412'
  AND actor.login = 'suspicious-username'
GROUP BY type
ORDER BY event_count DESC
```

---

## Cost Optimization (MANDATORY)

1. **Always dry run first**: Add `--dry_run` flag to `bq query` to see estimated bytes scanned before executing.
2. **Use `_TABLE_SUFFIX`**: Narrow the date range as much as possible. `day.*` tables are cheapest for narrow windows; `month.*` for broader sweeps.
3. **Select only needed columns**: Avoid `SELECT *`. The `payload` column is large — only select specific JSON paths.
4. **Add LIMIT**: Use `LIMIT 1000` during exploration. Remove only for final exhaustive queries.
5. **Column filtering in WHERE**: Filter on indexed columns (`type`, `repo.name`, `actor.login`) before payload extraction.

**Cost estimation**: A single month of GH Archive data is ~1-2 TiB uncompressed. Querying a specific repo + event type with `_TABLE_SUFFIX` typically scans 1-10 GiB ($0.006-$0.06).

---

## Accessing via Hermes

**Option A: BigQuery CLI** (if `gcloud` is installed)
```bash
bq query --use_legacy_sql=false --format=json "YOUR QUERY"
```

**Option B: Python** (via `execute_code`)
```python
from google.cloud import bigquery
client = bigquery.Client()
query = "YOUR QUERY"
results = client.query(query).result()
for row in results:
    print(dict(row))
```

**Option C: No GCP credentials available**
If BigQuery is unavailable, document this limitation in the report. Use the other 4 investigators (Git, GitHub API, Wayback Machine, IOC Enrichment) — they cover most investigation needs without BigQuery.
