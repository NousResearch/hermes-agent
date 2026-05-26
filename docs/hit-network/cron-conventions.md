# Cron Conventions (Hit Network)

1. Register and manage scheduled jobs exclusively via the Hermes `cronjob` tool (structured-state-has-an-api).
2. Store one-shot scripts under ~/.hermes/scripts/; recipe scripts under ~/hermes-workspace/recipes/<recipe>/scripts/.
3. Jobs that touch shared state use atomic writes (ale_atomic primitives) to prevent torn updates.
4. Cron logs live at ~/.hermes/logs/<cron-name>.log.
5. Per-job model/provider/base_url overrides are supported via cron/jobs.py fields; SIE Phase 5 Part B per-job exception isolation is active in cron loops.
