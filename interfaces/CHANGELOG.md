# Interface Changelog

## score_report.v1 (schema_version: 1) — Frozen
- Added `run_id`, `score` (`[0.0, 1.0]`), `outcome` (`passed|failed|error|timeout`), and `time_seconds`.
- `schema_version` set to integer `1`.
- `failure_reason` strictly required (`minLength: 1`) whenever `outcome != "passed"` (`passed == false`), and `null` when `passed == true`.
- `trajectory_ref` points to repo-relative POSIX path `bench/runs/<run_id>/trajectories/<task_id>.jsonl`.
