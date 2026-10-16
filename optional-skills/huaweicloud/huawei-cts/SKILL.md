---
name: huawei-cts
description: 'Use when managing Cloud Trace Service (CTS) audit logs, trackers, and traces on Huawei Cloud. Triggers: CTS, Cloud Trace, audit log, tracker, trace, operation record. NOT for: CES monitoring alarms (use huawei-cloud-eye), log analysis (use LTS).'
version: 1
---

# Huawei Cloud CTS

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud CTS <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for Cloud Trace Service. Covers trackers, traces, notifications, and resource tags.

## Critical Warnings

| Trap                        | Why                                                                            |
| --------------------------- | ------------------------------------------------------------------------------ |
| Tracker required for traces | Must create a tracker before traces are recorded                               |
| OBS bucket prerequisite     | Tracker needs an OBS bucket for log delivery                                   |
| 7-day retention default     | Trace data retained 7 days by default. Create LTS tracker for longer retention |
| Organization tracker        | Cross-account auditing requires organization tracker                           |

## Common Workflows

| Task                 | Operation                                              |
| -------------------- | ------------------------------------------------------ |
| List operations      | `ListOperations --cli-region=<r> --project_id=<p>`     |
| List traces          | `ListTraces --cli-region=<r> --project_id=<p>`         |
| Create tracker       | `CreateTracker --cli-region=<r> --project_id=<p>`      |
| List trackers        | `ListTrackers --cli-region=<r> --project_id=<p>`       |
| Delete tracker       | `DeleteTracker --cli-region=<r> --project_id=<p>`      |
| Create notification  | `CreateNotification --cli-region=<r> --project_id=<p>` |
| List notifications   | `ListNotifications --cli-region=<r> --project_id=<p>`  |
| List trace resources | `ListTraceResources --cli-region=<r> --project_id=<p>` |

Discover exact parameters with `--help` before executing any command.

## Troubleshooting

| Error                  | Fix                                                        |
| ---------------------- | ---------------------------------------------------------- |
| No traces found        | Check tracker is created and enabled. Filter by time range |
| Tracker creation fails | Ensure OBS bucket exists and has correct permissions       |

## Security

- SHOULD enable CTS for all production accounts
- MUST use OBS server-side encryption for trace logs

## Cross-Skill References

- **OBS**: See `huawei-obs` for tracker log delivery bucket
