---
name: huawei-cloud-eye
description: 'Use when setting up monitoring, alarms, dashboards, or event rules on Huawei Cloud Eye (CES). Triggers: Cloud Eye, CES, monitoring, alarm, metrics, dashboard, event monitoring. NOT for: CTS audit logs (use huawei-cts), AAD anti-DDoS (use huawei-waf-aad).'
version: 1
---

# Huawei Cloud Cloud Eye (CES)

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud CES <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for Cloud Eye (CES). Covers metric queries, alarm rules, dashboards, and monitoring agent management.

## Critical Warnings

| Trap                                          | Why                                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Telescope agent required for detailed metrics | Without agent: only basic CPU/network metrics. Memory, disk require agent installation on ECS          |
| Alarm needs SMN topic first                   | Alarm notifications fail silently without a configured SMN topic                                       |
| Cost metrics incur charges                    | Custom metrics and high-frequency alarms have billing implications                                     |
| CES endpoint may vary by region               | Some older regions use different endpoint URLs. Check `https://developer.huaweicloud.com/endpoint?CES` |
| Metric granularity minimum 300s               | Free tier has 5-minute minimum. Higher resolution requires paid tier                                   |

## Prerequisites

- ECS instances need the **Telescope agent** installed for detailed metrics (memory, disk, network)
- Without agent: only SYS.ECS namespace metrics available (CPU, network bytes, disk read/write)
- Alarm notifications require an SMN topic (see `huawei-smn-dms`)

## Common Workflows

| Task             | Operation                                                 |
| ---------------- | --------------------------------------------------------- |
| List metrics     | `ListMetrics --cli-region=<r> --project_id=<p>`           |
| Get metric data  | `BatchListMetricData --cli-region=<r> --project_id=<p>`   |
| List alarms      | `ListAlarms --cli-region=<r> --project_id=<p>`            |
| Create alarm     | `CreateAlarm` (see below)                                 |
| Delete alarm     | `BatchDeleteAlarmRules --cli-region=<r> --project_id=<p>` |
| List dashboards  | `ListDashboardWidgets --cli-region=<r> --project_id=<p>`  |
| Create dashboard | `CreateDashboard --cli-region=<r> --project_id=<p>`       |

## Create Alarm Rule — Param Structure

CES uses **nested object prefixes** for alarm creation. Always verify with `--help`:

```bash
hcloud CES CreateAlarm --help
# Key params: --alarm_name, --metric.metric_name, --metric.namespace,
# --condition.period, --condition.filter, --condition.value,
# --condition.comparison_operator, --condition.count
```

### `--condition.comparison_operator` Legal Values

For metric alarms use **symbols** (NOT words like `gt`/`lt`):

| Value | Meaning               |
| ----- | --------------------- |
| `>`   | Greater than          |
| `>=`  | Greater than or equal |
| `<`   | Less than             |
| `<=`  | Less than or equal    |
| `=`   | Equal                 |
| `!=`  | Not equal             |

For event alarms: `cycle_decrease`, `cycle_increase`, `cycle_wave`.

> **Trap**: Shell may interpret `>` and `<` as redirect operators. Always quote: `--condition.comparison_operator=">="` or use `--cli-jsonInput=<file>`.

## Troubleshooting

| Error                         | Fix                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------- |
| No metric data                | Telescope agent not installed or ECS stopped. Install agent for detailed metrics |
| Alarm not triggering          | Check metric period (minimum 300s), verify condition threshold                   |
| CES network timeout in region | Some regions may have CES endpoint issues. Try a different region                |

## Security

- MUST use role-based alarm notifications
- MUST not expose alarm action endpoints publicly
- SHOULD enable CES alarm history for audit

## Cross-Skill References

- **SMN topics**: See `huawei-smn-dms` for alarm notification setup
- **ECS monitoring**: See `huawei-ecs` for instance creation
