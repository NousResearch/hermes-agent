# Time Trigger Reference

## Event Types

- [`schedule.elapsed`](#scheduleelapsed)

---

## `schedule.elapsed`

### Parameters

- `cron_expression` (string, required): A standard 5-field cron expression (`minute hour day-of-month month day-of-week`). For example `0 9 * * *` fires daily at 9:00, `*/15 * * * *` every 15 minutes, and `0 9 * * 1-5` at 9:00 on weekdays.
- `timezone` (string, optional, default `UTC`): IANA timezone name the cron expression is evaluated in (e.g. `America/Los_Angeles`).

### Sample Payload

```json
{
  "scheduled_for": "2026-06-19T16:00:00Z",
  "cron_expression": "0 9 * * *",
  "timezone": "America/Los_Angeles"
}
```