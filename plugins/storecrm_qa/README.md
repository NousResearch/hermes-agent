# StoreCRM QA Control Plane

This plugin is the Hermes-owned foundation for StoreCRM QA queue state. It
stores jobs, cases, attempts, leases, and local report summaries in SQLite
under the active Hermes profile:

```bash
python -m plugins.storecrm_qa.cli enqueue \
  --name smoke --tenant tenant-dev --store store-dev \
  --case login --case checkout
```

By default the database is:

```text
{HERMES_HOME}/storecrm_qa/qa_jobs.sqlite3
```

## Boundary

StoreCRM owns the autotest runner adapter. Hermes owns the canonical QA queue:
job/case status, attempts, leases, stale recovery, retries, and redacted local
reports. This plugin does not create StoreCRM database tables, connect to
StoreCRM staging or production, run browser QA, send messages, or call external
providers.

## Commands

```bash
python -m plugins.storecrm_qa.cli list
python -m plugins.storecrm_qa.cli lease --owner worker-1
python -m plugins.storecrm_qa.cli heartbeat --case-id 1 --owner worker-1
python -m plugins.storecrm_qa.cli complete --case-id 1 --owner worker-1 --outcome pass --summary ok
python -m plugins.storecrm_qa.cli fail-retry --case-id 1 --owner worker-1 --summary "transient failure"
python -m plugins.storecrm_qa.cli recover-stale
python -m plugins.storecrm_qa.cli report --job-id 1 --output /tmp/storecrm-qa-report.json
```

Use `--db /path/to/file.sqlite3` for local tests or one-off dry runs. Reports
and evidence summaries pass through redaction helpers that strip
credential-shaped keys and token-like strings before persistence.

## Non-goals

- No StoreCRM DB migrations or queue state in StoreCRM tables.
- No live StoreCRM, browser, email, LINE, Slack, Jira, or provider calls.
- No always-on Hermes core model tool.
- No Kanban or cron ownership of canonical QA case state.
