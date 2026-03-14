# GitHub Archive Query Guide (BigQuery)

GitHub Archive records every public event on GitHub. It is accessible via Google BigQuery.

## Public Dataset
- **Project**: `githubarchive`
- **Tables**: `day.*`, `month.*`, `year.*`

## Query Template

```sql
SELECT
  created_at,
  type,
  actor.login,
  repo.name,
  payload
FROM
  `githubarchive.day.20240101` -- Adjust date
WHERE
  repo.name = 'owner/repo'
  AND type IN ('PushEvent', 'DeleteEvent', 'MemberEvent')
ORDER BY
  created_at ASC
```

## Cost Optimization Tips
- **Always use `_TABLE_SUFFIX`** to narrow the date range.
- **Only select columns you need**.
- **Limit results** during initial exploration.

## Accessing via Hermes
Use the `terminal` tool with `bq query` if the BigQuery CLI is installed, or `execute_code` with the Python `google-cloud-bigquery` library.
