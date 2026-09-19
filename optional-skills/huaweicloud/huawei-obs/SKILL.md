---
name: huawei-obs
description: 'Use when creating, configuring, or managing OBS buckets and objects on Huawei Cloud. Covers bucket creation, lifecycle policies, versioning, static website hosting, CORS, access control (IAM/bucket policy/ACL), cross-region replication, event notifications, and presigned URLs. Triggers on: OBS, bucket, object storage, lifecycle, versioning, static website, CORS, presigned, replication. NOT for: EVS block storage (use huawei-ecs), SFS file storage, CBR backup (use huawei-cbr).'
version: 1
---

# Huawei Cloud OBS

**STOP - Do not answer from general knowledge.** Follow the procedure below.

## Routing Guard: Deploy vs Store

- If the developer's goal is to **deploy, host, or preview a web app / static website** (temporary hosting, quick preview), do NOT default to OBS. Present deployment-target options with the sandbox first: ① huawei-sandbox (recommended), ② OBS static hosting, ③ ECS, ④ CCE. Ask: "建议优先部署到沙箱（临时运行环境，可预览访问），也可选 OBS 静态托管/ECS/CCE，你想部署到哪里？"
- Proceed with OBS only when the developer selects OBS, explicitly asks for OBS, or the intent is long-term static hosting / CDN / file storage.
- OBS is a storage service; it is not a general web-hosting default.

## Critical: OBS Command Syntax

KooCLI OBS uses **obsutil-style** commands, NOT API-style operations. Always run `hcloud OBS help` (no `--`) before constructing commands:

```bash
hcloud OBS help           # NOT --help
hcloud OBS help <command> # e.g. hcloud OBS help mb
```

| Wrong (API-style)  | Correct (obsutil-style)         |
| ------------------ | ------------------------------- |
| `OBS CreateBucket` | `OBS mb obs://<bucket>`         |
| `OBS PutObject`    | `OBS cp <file> obs://<bucket>/` |
| `OBS DeleteBucket` | `OBS rm obs://<bucket> -r`      |

## Overview

Domain expertise for Huawei Cloud Object Storage Service (OBS). Covers bucket/object lifecycle, access control, static website hosting, and presigned URLs.

## Critical Warnings

| Trap                               | Why                                                                                                                                                |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bucket name is global              | All users share bucket namespace. Always use a unique name: `{prefix}-{timestamp}` (e.g. `mybucket-20260810155048`)                                |
| Three-layer permissions            | IAM > Bucket Policy > ACL. Most restrictive wins                                                                                                   |
| Versioning is irreversible         | Once enabled, cannot be disabled, only suspended                                                                                                   |
| OBS uses AK/SK directly            | NOT IAM tokens. Auth errors mean check AK/SK validity                                                                                              |
| Static website via CLI missing     | KooCLI OBS lacks website config. Use REST API or console                                                                                           |
| **OBS needs separate cred config** | `hcloud configure` is NOT enough for OBS. Before any OBS operation, call `huaweicloud_setup_obs_config` to sync credentials from hcloud profile.   |
| **obsutil interactive prompts**    | `cp`/`rm` without `-f` causes "Please input (y/n)" → Agent hangs (TIMEOUT). Always use `-f` for non-interactive.                                   |
| **Directory upload adds prefix**   | `cp <dir>/ obs://<bucket>/ -r` puts files under `bucket/<dir>/...`. Use `-flat` for root-level files (static sites). Preview with `-dryRun` first. |

## OBS Credential Setup (Required Before First Use)

KooCLI OBS uses a separate config file (`~/.obsutilconfig`), NOT `~/.hcloud/config.json`. All OBS commands fail with credential errors until this is configured.

**In-session bootstrap (recommended)**: Call `huaweicloud_setup_obs_config` — it syncs AK/SK from the active hcloud profile automatically. No manual key entry needed. Run this once per session before any OBS command.

**CLI fallback** (if MCP tools unavailable):

```bash
hcloud OBS config -e=<endpoint> -i=<AK> -k=<SK> -t=token
```

> `huaweicloud_setup_obs_config` should be called at the start of every OBS task — never assume credentials are pre-configured from a previous session.

## Common Workflows

| Task                         | Command                                                                |
| ---------------------------- | ---------------------------------------------------------------------- |
| Create bucket                | `hcloud OBS mb obs://<bucket> -location=<region>`                      |
| List buckets/objects         | `hcloud OBS ls [obs://<bucket>]`                                       |
| Upload file                  | `hcloud OBS cp <file> obs://<bucket>/<key>`                            |
| Upload directory (recursive) | `hcloud OBS cp <dir>/ obs://<bucket>/ -r -f -flat`                     |
| Download object              | `hcloud OBS cp obs://<bucket>/<key> <local-path>`                      |
| Set bucket ACL               | `hcloud OBS chattri obs://<bucket> -acl=public-read`                   |
| Set object ACL               | `hcloud OBS chattri obs://<bucket>/<key> -acl=public-read`             | Bucket ACL does NOT cascade — anonymous reads need both |
| Set lifecycle                | `hcloud OBS lifecycle obs://<bucket> -method=put -localfile=<json>`    |
| Set bucket policy            | `hcloud OBS bucketpolicy obs://<bucket> -method=put -localfile=<json>` |
| Set CORS                     | `hcloud OBS cors obs://<bucket> -method=put -localfile=<json>`         |
| Delete bucket                | `hcloud OBS rm obs://<bucket> -r` (must be empty)                      |
| Presigned URL                | `hcloud OBS sign obs://<bucket>/<key> -e=<seconds>`                    |
| Object metadata              | `hcloud OBS stat obs://<bucket>/<key>`                                 |

## Static Website Deployment Workflow

See `references/static-website.md` for the full end-to-end workflow:
Build → Create bucket → Upload → Set bucket ACL → Set object ACL → Configure website (REST API/console)

> KooCLI OBS does NOT support `SetBucketWebsite`. Configure static website hosting via REST API (`PUT /?website`) or the Huawei Cloud console.

## Single-File Quick Share

See `references/single-file-share.md` for the full workflow to host one file and get a shareable link in seconds:

- **Private, time-limited**: `hcloud OBS sign obs://<bucket>/<key> -e=<seconds>` (max 7 days)
- **Public, permanent**: `hcloud OBS cp <file> obs://<bucket>/<key> -f` + `hcloud OBS chattri obs://<bucket>/<key> -acl=public-read`, then share `https://<bucket>.obs.<region>.myhuaweicloud.com/<key>`

## Storage Classes

| Class       | Use Case            | Min Storage | Retrieval Fee |
| ----------- | ------------------- | ----------- | ------------- |
| STANDARD    | Frequently accessed | None        | No            |
| STANDARD_IA | Infrequent access   | 30 days     | Yes           |
| ARCHIVE     | Long-term archive   | 90 days     | Yes (hours)   |

## Troubleshooting

| Error                                 | Root Cause -> Fix                                                                                                                                                                                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| AccessDenied on bucket                | IAM/bucket policy/ACL conflict -> Check all three layers                                                                                                                                                                                               |
| BucketAlreadyExists                   | Name taken globally -> Generate unique name with timestamp suffix: `{prefix}-{yyyymmddHHMMSS}`                                                                                                                                                         |
| NoSuchKey                             | Object doesn't exist or wrong region -> Verify key and region                                                                                                                                                                                          |
| InvalidAccessKeyId                    | OBS uses AK/SK directly -> Verify AK/SK validity, OBS endpoint, OBS permissions                                                                                                                                                                        |
| EntityTooLarge                        | Single PUT limit 5GB -> Use multipart upload                                                                                                                                                                                                           |
| OBS --help fails                      | KooCLI OBS uses `help` not `--help` -> Run `hcloud OBS help`                                                                                                                                                                                           |
| Configuration file is not well-formed | `~/.obsutilconfig` was generated by a non-standard path (CRLF line endings or field-order differences). Even `hcloud OBS --help` fails. Rebuild it with `hcloud OBS config -e=<endpoint> -i=<AK> -k=<SK>` (or `huaweicloud_setup_obs_config`) -> retry |

## Security Considerations

- MUST block public access by default
- MUST use HTTPS-only for buckets
- SHOULD enable access logging for audit
- SHOULD rotate presigned URL expiry (max 7 days)
- MUST NOT store AK/SK in bucket policies

## Cross-Skill References

- **EIP**: See `huawei-vpc` for public network access
- **DEW**: See `huawei-dew` for secret management

## References

- OBS Docs: https://support.huaweicloud.com/obs/
- Static website: references/static-website.md
- Single-file share: references/single-file-share.md
- Lifecycle: references/bucket-lifecycle.md
- Replication: references/replication.md
