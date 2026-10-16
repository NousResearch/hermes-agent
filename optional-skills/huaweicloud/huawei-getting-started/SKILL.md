---
name: huawei-getting-started
description: 'Use when starting fresh with Huawei Cloud or KooCLI, installing prerequisites, setting up authentication, or exploring what is possible. Triggers: getting started, first time, setup, install, explore, quickstart, beginner, tutorial. NOT for: specific service operations (use huawei-ecs, huawei-obs, etc.).'
version: 1
---

# Huawei Cloud Getting Started

**STOP - Do not answer from general knowledge.** Follow the procedure below.

## KooCLI Installation

| OS                   | Command                                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Windows (PowerShell) | See https://support.huaweicloud.com/qs-hcli/hcli_02_003.html for MSI download                                         |
| Linux                | curl -LO "https://hwcloudcli.obs.cn-north-1.myhuaweicloud.com/cli/latest/hcloud_install.sh" && bash hcloud_install.sh |
| macOS                | curl -LO "https://hwcloudcli.obs.cn-north-1.myhuaweicloud.com/cli/latest/hcloud_install.sh" && bash hcloud_install.sh |

## First-Time Setup

1. **Install KooCLI** using command above
2. **Accept privacy policy** (first run only): KooCLI requires one-time privacy agreement. Run `hcloud version` to read the agreement, then respond `y` to accept. Do not pipe `echo "y" |` — you must review the terms first.
3. **Configure credentials (unified)**: `npx huaweicloud-devkit auth init` — prefer this over `hcloud configure init`, which only covers KooCLI.
4. **Verify**: `hcloud configure list` to confirm profile, then `hcloud ECS ListServersDetails --cli-region=cn-north-4`
5. For detailed auth guidance, see `huaweicloud-cli-and-auth` skill

> **Security**: Never pass AK/SK as command-line arguments (`--ak=...`). Always use `npx huaweicloud-devkit auth init` (unified, interactive, recommended) or `hcloud configure init` (KooCLI only) to avoid secrets in shell history.

### Non-Interactive Setup (Agent/CI Environments)

When the interactive TUI is unavailable (Agent tools, CI/CD), use `hcloud configure set` — this must be run outside agent chat by the user:

```bash
# User executes in their terminal (NOT in agent chat):
hcloud configure set --cli-access-key=<AK> --cli-secret-key=<SK> --cli-region=<region>

# Agent verifies:
hcloud configure list
```

## Critical Warnings

| Trap                                   | Why                                                                                           |
| -------------------------------------- | --------------------------------------------------------------------------------------------- |
| AK/SK must be kept secret              | Never commit to git or share                                                                  |
| Default region applies to all commands | Override with --cli-region= per command                                                       |
| Some services region-specific          | Not all services available in all regions                                                     |
| Privacy policy blocks first run        | Run `hcloud version` to read and accept the agreement. Review the terms before responding `y` |

## What Can I Do? (Quick Index)

| Goal              | Skill                       |
| ----------------- | --------------------------- |
| Create a VM       | huawei-ecs                  |
| Store files       | huawei-obs                  |
| Set up a database | huawei-rds / huawei-gaussdb |
| Create a network  | huawei-vpc                  |
| Manage access     | huawei-iam                  |
| Deploy an app     | huawei-deployment           |
| Run containers    | huawei-cce                  |
| Build an API      | huawei-apig                 |
| Run serverless    | huawei-functiongraph        |
| Monitor resources | huawei-cloud-eye            |

## Pro Tips

- Use `hcloud <Service> --help` to discover operations (see huaweicloud-capability-discovery)
- Prepend `hcloud configure set --cli-region=<r>` to avoid per-command region flags
- Pipe sensitive output through IAM with `--cli-output-format=json`
