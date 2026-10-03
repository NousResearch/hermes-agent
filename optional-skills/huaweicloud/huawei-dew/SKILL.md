---
name: huawei-dew
description: 'Use when managing secrets, credentials, encryption keys, certificates, or any sensitive data on Huawei Cloud. Covers CSMS (Cloud Secret Management Service) for secrets, KMS for encryption keys, and credential rotation. Triggers on: secret, credential, API key, token, password, encrypt, decrypt, KMS, CSMS, DEW, certificate, CSR, rotation. Activates for ANY task involving secrets or credentials — load this skill first before touching secrets.'
version: 1
---

# Huawei Cloud DEW (Data Encryption Workshop)

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud <Service> <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for Huawei Cloud Data Encryption Workshop (DEW). Covers CSMS for secret lifecycle management, KMS for encryption key management, credential rotation, and secure application integration patterns.

## Critical Warnings

| Trap                                         | Why                                                                |
| -------------------------------------------- | ------------------------------------------------------------------ |
| NEVER fetch secret values into agent context | Use {{resolve:csms:secret-id:SecretString:key}} runtime injection  |
| NEVER echo credentials in conversation       | AK/SK, passwords, tokens must never appear in chat                 |
| KMS key deletion is irreversible             | 7-30 day pending deletion window. Once gone, data is unrecoverable |
| Secret rotation requires automation          | Manual rotation risks stale credentials                            |
| Cross-account KMS needs grants               | KMS keys are regional. Cross-region data needs grant setup         |

## CSMS Operations

### Read-only (safe — metadata only)

| Operation          | Description                                            |
| ------------------ | ------------------------------------------------------ |
| ListSecrets        | List all secrets (names only, no values)               |
| DescribeSecret     | Get secret metadata (rotation config, KMS key, status) |
| ListSecretVersions | List version IDs and stages (no values)                |

### Secret Value (blocked by policy — use runtime injection)

| Operation         | Safe Alternative                              |
| ----------------- | --------------------------------------------- |
| DownloadSecret    | Use {{resolve:csms:secret-id}} in IaC or SDK  |
| ShowSecretVersion | Use runtime injection, never in agent context |
| GetSecretValue    | Blocked. Use MCP proxy resolve pattern        |

## KMS Operations

| Task       | Command                                                | Notes                      |
| ---------- | ------------------------------------------------------ | -------------------------- |
| List keys  | hcloud KMS ListKeys                                    | Read-only                  |
| Create key | hcloud KMS CreateKey --key_alias=<name>                | Requires write approval    |
| Encrypt    | hcloud KMS EncryptData --key_id=<id> --plaintext=<b64> | Prefer SDK offline encrypt |
| Decrypt    | hcloud KMS DecryptData (**BLOCKED**)                   | Use runtime injection      |

## Runtime Injection Pattern

`ash

# Terraform / IaC

resource "huaweicloud_csms_secret" "db_password" {
secret_string = var.db_password # NEVER hardcode
}

# SDK

secret_value = client.get_secret(secret_id="my-secret")

# Use immediately, never print or store in context

# hcloud (for approved automation only)

hcloud CSMS DownloadSecret --secret_name=<name>

# WARNING: Output goes to stdout. Pipe directly, never capture in agent.

`

## Troubleshooting

| Error                | Root Cause -> Fix                                                           |
| -------------------- | --------------------------------------------------------------------------- |
| Secret not found     | Wrong region or project -> Verify secret ARN includes region/project        |
| AccessDenied on CSMS | Missing IAM CSMS policy -> Add csms:DescribeSecret + kms:Decrypt            |
| KMS key disabled     | Key scheduled for deletion or manually disabled -> Enable or create new key |
| Rotation stuck       | Lambda/Python function error -> Check rotation function logs                |

## Security Considerations

- MUST use runtime injection. NEVER fetch secrets into agent context
- MUST rotate credentials every 90 days
- MUST enable automatic rotation for CSMS secrets
- MUST audit KMS key usage via CTS
- SHOULD use customer-managed keys (CMK) for sensitive data
- MUST NOT use default KMS keys for production workloads

## MCP Tools

- huaweicloud_list_operations service=DEW or CSMS or KMS
- huaweicloud_run_readonly_command for ListSecrets/ListKeys (metadata only)
- huaweicloud_safety policy blocks all secret value reads

## References

- DEW Docs: https://support.huaweicloud.com/dew/
- CSMS usage: references/csms-usage.md
- KMS usage: references/kms-usage.md
