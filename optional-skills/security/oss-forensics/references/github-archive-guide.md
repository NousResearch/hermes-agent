# GitHub Archive Query Guide (BigQuery)

GH Archive archives JSON-encoded public events as reported by GitHub's APIs;
from 2015 onward its stated source is the Events API. It is useful historical
evidence, not a guarantee of complete coverage, immutable storage, or the most
reliable account of every action. API visibility, collection gaps and event/payload
changes limit what can be observed. Absence from the archive is not proof that an
action did not occur, and a recorded SHA does not guarantee object recovery.
Preserve the source, query/time window and a hash of acquired evidence; that hash
checks your acquired copy, not upstream completeness or tamper-proof storage.
See the [GH Archive project](https://www.gharchive.org/) and
[GitHub Events API limits](https://docs.github.com/en/rest/activity/events).

## Public Dataset

- **Project**: `githubarchive`
- **Tables**: `day.YYYYMMDD`, `month.YYYYMM`, `year.YYYY`
- **Cost**: Check the applicable [BigQuery pricing](https://cloud.google.com/bigquery/pricing), region, billing model and account allowances; do not infer a bill from a generic example. Dry-run the exact query first.
- **Access**: Requires an authorized Google Cloud project with BigQuery access; public dataset access does not make query execution universally free.

---

## Illustrative GitHub Event Types

These are investigative leads, not a promise that every type/action is emitted
by the Events API or present in the selected archive window. In particular, do not
assume webhook event types such as `WorkflowRunEvent` appear in that feed. Check
the applicable API/event schema and actual archived records before using a template.
`actor.login` identifies the reported platform account, not the natural person
operating it. Git author name/email are configurable commit metadata, not proof
of that person's identity or of who pushed. See
[Git identity configuration](https://git-scm.com/docs/git-config#Documentation/git-config.txt-username).

| Event Type | What It Records | Investigative Lead (not a finding) |
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

### Historical PushEvent Leads for a Possible Ref Rewrite

For historical payloads that contain these fields, `size` is the number of
commits in the push and `distinct_size` the number of distinct commits in the
push, **not a count of deleted commits**. A zero `distinct_size` with positive
`size` is only a lead; previously pushed commits can be pushed to another branch
without erasing anything. Missing fields are unknown, not zero. The versioned
[GHES 3.18 event schema](https://docs.github.com/en/enterprise-server@3.18/rest/using-the-rest-api/github-event-types#pushevent)
documents this historical shape; it is not a claim that all GitHub.com archive
dates or current API versions have the same fields.

To classify an observed update of the **same repository and ref**, use its
`before` and `head` plus an adequate commit graph: if `before` is an ancestor of
`head`, the update is fast-forward; if it is not, the update is non-fast-forward.
Unrelated refs, branch creation/deletion (including zero-SHA sentinels), missing
objects, shallow/incomplete graphs or missing events leave this test inconclusive.
See [Git's fast-forward definition](https://git-scm.com/docs/git-push#_note_about_fast_forwards).
A non-fast-forward ref update still does not prove malicious intent or deletion
of server-side Git objects. Knowing a `before` SHA is an identification lead, not
a guarantee that the commit or its contents remain accessible.

The general lead is to list the same ref's PushEvents with `before`/`head` and
check ancestry as above, without depending on historical commit-count fields.
A rewind to an ancestor can introduce zero new commits; a `size = 0` payload
shape is a semantic/synthetic example here, not an observed GitHub event.
The following historical-field filter is only an optional, incomplete lead:
missing count fields are excluded, so it cannot replace that general route or
prove a force-push.

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
  AND SAFE_CAST(JSON_EXTRACT_SCALAR(payload, '$.distinct_size') AS INT64) = 0
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

1. **Estimate first**: Dry-run the exact final SQL to estimate processed bytes; this does not execute the query. Estimates are not an observed bill. Review any changed query again before execution.
2. **Prune actual tables**: Use a narrow date window and a **constant expression** for `_TABLE_SUFFIX` on wildcard tables (as in the templates). These are date-sharded tables, not proof of time partitioning. Match suffix literals to the wildcard prefix. Compare actual estimates rather than assuming `day.*` or `month.*` is always cheapest.
3. **Select only necessary columns**: Avoid `SELECT *`. Extracting a few paths from a STRING `payload` column does not by itself avoid scanning that column; do not promise path-level byte savings without evidence from the actual storage/schema.
4. **Separate output limits from cost controls**: `LIMIT 1000` can bound returned rows during exploration. On non-clustered tables it does not reduce the bytes read. Clustered tables can sometimes stop scanning earlier with `LIMIT`; do not assume this dataset is clustered or that this is a universal cost ceiling.
5. **Set an execution ceiling**: For on-demand queries, set `--maximum_bytes_billed` (Python: `QueryJobConfig.maximum_bytes_billed`) to the approved byte budget. BigQuery rejects a query whose estimated billed bytes exceed the ceiling without a query charge. Clustered-table estimates can be conservative and reject a query even when actual bytes would be lower. This ceiling is not an account-wide spending limit or a reservation/capacity-billing cap.
6. **Do not invent indexes or scan estimates**: Keep `type`, `repo.name` and `actor.login` predicates for relevance, but neither ordinary field names nor textual WHERE order proves index use or reduced scan cost. Check actual schema/partitioning/clustering and dry-run estimates; do not promise a fixed GiB range or price for a repository filter.

See [BigQuery cost estimation and controls](https://cloud.google.com/bigquery/docs/best-practices-costs)
and [constant `_TABLE_SUFFIX` filtering](https://cloud.google.com/bigquery/docs/querying-wildcard-tables#filter_selected_tables_using_table_suffix).
These controls preserve normal authorized forensic queries; they do not require
an extra source for every fact or prohibit legitimate archive investigation.

---

## Accessing via Hermes

**Option A: BigQuery CLI** (if `gcloud` is installed)
```bash
# Estimate only. Use the same exact SQL for any separately authorized execution.
bq query --use_legacy_sql=false --dry_run "YOUR QUERY"
# Paid execution is disabled below. After reviewing the estimate, replace
# <APPROVED_BYTES> with the approved per-query byte budget and run separately:
# bq query --use_legacy_sql=false --maximum_bytes_billed=<APPROVED_BYTES> --format=json "YOUR QUERY"
```

**Option B: Python** (via `execute_code`)
```python
from google.cloud import bigquery
client = bigquery.Client()
query = "YOUR QUERY"
estimate = client.query(
    query, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
)
print("Estimated processed bytes:", estimate.total_bytes_processed)
# Paid execution is disabled. Review the estimate, then replace <APPROVED_BYTES>
# with the approved integer byte budget before running the following separately:
# max_bytes = <APPROVED_BYTES>
# job_config = bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes)
# results = client.query(query, job_config=job_config).result()
# for row in results:
#     print(dict(row))
```

**Option C: No GCP credentials available**
If BigQuery is unavailable, document this limitation in the report. Use the other 4 investigators (Git, GitHub API, Wayback Machine, IOC Enrichment) — they cover most investigation needs without BigQuery.
