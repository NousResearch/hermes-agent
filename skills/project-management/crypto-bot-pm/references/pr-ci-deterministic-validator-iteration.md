# Crypto Bot PR CI deterministic validator iteration

Use this reference when an S006/S00x local Gitea PR sync CI run fails on deterministic validators after the PR branch has already been pushed.

## Evidence-first loop

1. Resolve the exact PR branch/head from live Gitea/read-only DB evidence before making claims.
2. Collect job status and logs by run/job id, preserving the commit SHA and event (`pull_request_sync`) in the report.
3. Classify failures as:
   - product/evidence gap: missing docs, examples, JSON fixtures, or validator-required evidence artifacts;
   - validator false-negative: deterministic validator rejects an already-governed safe state;
   - governed blocker: requires runner/workflow/service/Gitea mutation outside approved scope.
4. Repair only the narrowest allowed source/evidence surface, then run the exact failing validator locally before commit.
5. Commit and push only after `git diff --cached --check` and targeted validator evidence pass.
6. Wait for the next PR sync CI run, collect all job statuses/logs, and do not report green unless every relevant job is observed passing.

## Absent historical commit objects in validators

Several AutoResearch design-lane validators check that old design commits are not reachable from current `HEAD`. In local Gitea mirrors, an old short SHA may be absent from the object database. Treat this as unreachable, not as a hard validator failure, when the invariant is specifically “not reachable from HEAD.” Durable helper pattern:

```python
def _run_git_merge_base(commit: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return False, f"{commit} is reachable from HEAD"
    if result.returncode == 1:
        return True, f"{commit} unreachable from HEAD"
    detail = (result.stderr or result.stdout or "git merge-base failed").strip()
    if "Not a valid object name" in detail or "Not a valid commit name" in detail:
        return True, f"{commit} absent from repository object database; treated as unreachable from HEAD"
    return False, detail
```

Apply this only to validators whose acceptance condition is commit unreachability/absence. Do not generalize to cases that require an object to exist.

## Data-platform contract examples

`validate-data-platform-contracts.py` fails closed if `docs/data/examples/` or its two fixture JSON files are missing:

- `historical_manifest_example.json`
- `feature_definition_example.json`

When adding fixtures:

- Keep them deterministic and synthetic-only.
- Avoid sensitive key names such as `credentials`, `token`, `account_id`, `lease`, or `handoff` anywhere in JSON keys; `contains_sensitive_keys()` scans keys recursively, including redaction-policy keys.
- The historical manifest validator rewrites `storage_paths` to a temp safe root during validation, so the fixture can use a harmless relative JSONL path.
- The feature definition must use safe source columns (`open`, `close`, `timestamp`, `symbol`, etc.) and non-executable natural-language transformation text.

## Local Gitea Actions DB polling and logs

When the Gitea UI is not the fastest evidence source, read the local Gitea Actions tables directly and keep the numeric status mapping explicit in reports. Query the run and jobs for the exact commit SHA:

```sql
SELECT ar.id, ar.status, ar.commit_sha, ar.ref, ar.event,
       arj.id, arj.name, arj.status, arj.task_id, arj.started, arj.stopped
FROM action_run ar
JOIN action_run_job arj ON arj.run_id = ar.id
WHERE ar.commit_sha = '<sha>'
ORDER BY arj.id;
```

Observed Gitea status codes in the local setup:

- `1`: success
- `2`: failure
- `5`: waiting/queued
- `6`: running

Poll until every job leaves `{3,4,5,6}` rather than stopping when the first job finishes. Then fetch logs through the Gitea API with a temporary read-only token, save them outside the repo under `/tmp/.../logs/`, and delete the token in a trap. Do not print token material.

## Runtime database metric redaction pitfall

A deterministic Python-quality failure in `tests/observability_service_test.py::test_observability_db_file_size_redacts_runtime_name` exposed a redaction-policy edge case: temp/test paths alone are not sufficient to report DB file size when the basename is the runtime DB name `crypto_trading.db`. The safe policy is conjunctive:

- If `db_path.name == "crypto_trading.db"`, return `(0, "redacted")` even under `/tmp/` or `/tests/`.
- Only non-runtime database filenames under temp/test paths may report actual size.

This protects runtime-name leakage while preserving test/temp observability for synthetic database names. Validate with the targeted test, the observability test file, then `bash scripts/validation/validate-python-quality.sh` before commit/push.

## Reporting discipline

If the tool-call/session cap interrupts mid-repair, report the exact last verified state: latest pushed SHA, latest CI run/job statuses, log paths, uncommitted local changes, and which validation/commit/push steps remain. Never imply committed/pushed/green state before it is verified.
