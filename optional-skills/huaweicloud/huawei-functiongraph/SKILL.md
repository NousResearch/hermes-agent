---
name: huawei-functiongraph
description: 'Use when creating, deploying, or managing serverless functions on FunctionGraph. Covers triggers (APIG/OBS/timer/SMN), cold start, reserved concurrency. Triggers: FunctionGraph, serverless, function, Lambda, trigger. NOT for: CCE containers (use huawei-cce).'
version: 1
---

# Huawei Cloud FunctionGraph

**STOP - Do not answer from general knowledge.** Follow the procedure below.

## Overview

Domain expertise for Huawei Cloud FunctionGraph. Covers function lifecycle, code deployment, trigger configuration, and troubleshooting.

## How to Use This Skill

1. This skill tells you the **correct service/operation names** and **non-obvious traps**.
2. **Parameters are discovered via `--help`, not hardcoded.** Always run `hcloud FunctionGraph <Operation> --help` before constructing commands.
3. Detailed examples are in `references/` — load them only when needed.

## Critical Warnings

| Trap                                       | Why                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Service name is `FunctionGraph`            | NOT `FGS`. KooCLI 7.x uses the full service name                                                                                                                                                                                                                                                    |
| CLI requires `project_id`                  | Get it: `hcloud IAM KeystoneListProjects` or extract from URN `urn:fss:<region>:<project_id>:...`                                                                                                                                                                                                   |
| Cold start 100ms-2s                        | Reserve concurrency for latency-sensitive workloads                                                                                                                                                                                                                                                 |
| Max execution 900s                         | Use ECS/CCE for long-running tasks                                                                                                                                                                                                                                                                  |
| Env vars plaintext                         | Use DEW for secrets                                                                                                                                                                                                                                                                                 |
| **ZIP upload is unreliable**               | `CreateFunction --code_type=zip --code_filename=xxx.zip` succeeds even with a nonexistent file — the function is created with empty code. Prefer `code_type=inline` for simple functions. If using zip, verify `code_size` > placeholder after creation and run `InvokeFunction` to confirm output. |
| **DeleteFunction strip `:latest`**         | Passing `urn:fss:...:function:xxx:latest` causes FSS.0400. Strip `:latest` suffix (DeleteFunction only — UpdateApiV2 needs the full URN).                                                                                                                                                           |
| **DeleteFunctionTrigger does NOT cascade** | Deleting a DEDICATEDGATEWAY trigger leaves the API in APIG. Manually clean up: `hcloud APIG DeleteApiV2 --instance_id=<id> --api_id=<api-id>`.                                                                                                                                                      |

## Prerequisites

```bash
hcloud configure list              # confirm a profile exists
hcloud FunctionGraph --help        # confirm service is available
```

## IAM Permissions

If you see `FSS.0403 Forbidden`, the user needs these permissions:

| Operation       | Required Permission                     |
| --------------- | --------------------------------------- |
| Create function | `functiongraph:function:createFunction` |
| List functions  | `functiongraph:function:list`           |
| Delete function | `functiongraph:function:deleteFunction` |
| Invoke function | `functiongraph:function:invoke`         |
| Create trigger  | `functiongraph:trigger:*`               |

Grant via IAM console (`FunctionGraph FullAccess` role) or ask project admin.

## Operations

Always discover parameters with `--help` before executing. These are the correct operation names:

| Task            | Operation               | Gotchas                                                                                                                            |
| --------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| List functions  | `ListFunctions`         |                                                                                                                                    |
| Show function   | `ShowFunctionConfig`    |                                                                                                                                    |
| Create function | `CreateFunction`        | references/create-function.md                                                                                                      |
| Delete function | `DeleteFunction`        | Strip `:latest` from URN                                                                                                           |
| Invoke function | `InvokeFunction`        | Requires body param (`--name=<value>` becomes event body). Use `--x_cff_request_version=v0` for raw output, `v1` for APIG-wrapped. |
| Create trigger  | `CreateFunctionTrigger` | references/triggers.md                                                                                                             |
| List triggers   | `ListFunctionTriggers`  |                                                                                                                                    |
| Delete trigger  | `DeleteFunctionTrigger` |                                                                                                                                    |

Discover exact parameters:

```bash
hcloud FunctionGraph CreateFunction --help
hcloud FunctionGraph CreateFunctionTrigger --help
```

## Deployment Workflow

```
Write code → zip → CreateFunction → InvokeFunction → CreateFunctionTrigger
```

See `references/deploy-workflow.md` for a step-by-step example with code templates.

## Troubleshooting

| Error                             | Root Cause -> Fix                                                        |
| --------------------------------- | ------------------------------------------------------------------------ |
| `不支持的服务名称:FGS`            | Use `FunctionGraph`, not `FGS`                                           |
| `不支持的operation:CreateTrigger` | Use `CreateFunctionTrigger`                                              |
| `FSS.0403` / Forbidden            | Missing IAM permissions — see IAM Permissions above                      |
| `缺少必填参数:{*}` on Invoke      | Add body param: `--name=<value>`                                         |
| APIG/EOM trigger error            | `trigger_type_code=APIG` deprecated — use `DEDICATEDGATEWAY`             |
| `event_data` parse error          | Use dotted format: `--event_data.key=value`, NOT JSON string             |
| FSS.1078 / code upload fails      | `--code_filename` is filename-only, no path. `cd` to zip directory first |
| DeleteFunction with `:latest`     | Strip `:latest` version suffix from URN                                  |
| Code too large                    | Inline limit 10KB — use `zip`/`obs` code type                            |
| Cold start slow                   | Set reserved instances for critical functions                            |
| Auth failure                      | Run `npx huaweicloud-devkit auth init`                                   |

## Security Considerations

- MUST use DEW for secrets, never hardcode in environment variables
- MUST use `--app_xrole` (agency) for cross-service access
- SHOULD enable CTS audit logging for function invocations
- MUST NOT expose AK/SK in function code

## MCP Tools

- `huaweicloud_list_operations` service=FunctionGraph
- `huaweicloud_run_readonly_command` for discovery
- `huaweicloud_run_approved_command` for writes

## Without MCP

Fall back to hcloud CLI. State: "MCP unavailable, using local hcloud CLI."

## Cross-Skill References

- **APIG trigger setup**: See `huawei-apig` for API group creation and publishing
- **OBS trigger setup**: See `huawei-obs` for bucket and object event configuration
- **DEW secrets**: See `huawei-dew` for managing function secrets
- **SMN notifications**: See `huawei-smn-dms` for notification topics
- **VPC configuration**: See `huawei-vpc` for network settings

## References

- FunctionGraph Docs: https://support.huaweicloud.com/functiongraph/
- Create function: references/create-function.md
- Triggers: references/triggers.md
- Deploy workflow: references/deploy-workflow.md
- APIG event format: references/apig-event-format.md
