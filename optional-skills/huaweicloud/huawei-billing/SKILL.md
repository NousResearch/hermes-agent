---
name: huawei-billing
description: 'Use when querying bills, costs, resource usage, or billing details on Huawei Cloud (BSS). Triggers: billing, BSS, cost, bill, expense, usage report, resource usage, budget. NOT for: resource management (use huawei-ecs etc.), creating resources.'
version: 1
---

# Huawei Cloud Billing (BSS)

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud BSS <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for billing queries (BSS). Covers cost tracking, bill details, and budget management. Read-only — no resource modifications.

## Critical Warnings

| Trap                       | Why                                                  |
| -------------------------- | ---------------------------------------------------- |
| Bills delayed ~24h         | Yesterday's costs may not appear until the next day  |
| BSS Admin role needed      | IAM user must have BSS Administrator or Finance role |
| Currency conversion varies | Cross-region costs use daily exchange rates          |

## Common Workflows

| Task                 | Operation                                                       |
| -------------------- | --------------------------------------------------------------- |
| List costs           | `ListCosts --cli-region=<r> --project_id=<p>`                   |
| List customer bills  | `ListCustomerBills --cli-region=<r> --project_id=<p>`           |
| List resource usage  | `ListResourceUsage --cli-region=<r> --project_id=<p>`           |
| List sub-customers   | `ListConsumeSubCustomers --cli-region=<r> --project_id=<p>`     |
| Show account balance | `ShowCustomerAccountBalances --cli-region=<r> --project_id=<p>` |
| List conversions     | `ListConversions --cli-region=<r> --project_id=<p>`             |

Discover exact parameters with `--help` before executing any command. All BSS operations are read-only.

## Troubleshooting

| Error                           | Fix                                                         |
| ------------------------------- | ----------------------------------------------------------- |
| Access denied                   | User needs BSS Administrator or Finance role                |
| No data returned                | Check time range (bills have ~24h delay). Verify project_id |
| Enterprise account restrictions | Some APIs require enterprise real-name authentication       |

## Cross-Skill References

- **Resource lifecycle**: See `huawei-ecs`, `huawei-obs` for creating billable resources
