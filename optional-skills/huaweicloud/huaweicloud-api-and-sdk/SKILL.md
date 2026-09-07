---
name: huaweicloud-api-and-sdk
description: Huawei Cloud API and SDK guidance for application development. Use when writing code that calls Huawei Cloud services, choosing SDKs, building API requests, handling project_id, endpoint, auth, pagination, retries, error codes, or request IDs.
---

# Huawei Cloud API And SDK

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Use this skill when the deliverable is application code, API integration, or precise request construction.

## Critical Warnings

| Trap                                     | Why                                                                                                       |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `project_id` in path for most services   | Many Huawei Cloud APIs require `project_id` in the URL path. Get it from IAM or KooCLI profile            |
| Pagination is not automatic              | APIs return paginated results. Always check for `next_marker` or `page_info` to avoid partial data        |
| AK/SK signing algorithm                  | Huawei Cloud uses HMAC-SHA256 signing with specific header ordering. Use SDK, don't implement manually    |
| Endpoint varies by service and region    | Always verify at https://developer.huaweicloud.com/endpoint                                               |
| Error response in `body` not HTTP status | Many errors return HTTP 200 with error details in the JSON body. Check `error_code` or `error_msg` fields |

## API Workflow

1. Identify service, region, endpoint, API version, and whether `project_id` is required in the path.
2. Verify the exact request body and response schema from official API documentation.
3. Handle pagination explicitly. Do not assume a single page.
4. Preserve `request_id` from errors and responses for troubleshooting.
5. Classify operation risk before running a live call.
6. For write APIs, produce a dry plan and ask the user before execution.

## SDK Workflow

1. Pick the SDK that matches the user's application language and existing dependency style.
2. Use the official SDK client for the target service.
3. Keep credentials out of source code. Use environment, profile, workload identity, or runtime secret injection.
4. Add timeout, retry, and pagination handling.
5. Return typed or structured results instead of unbounded raw logs.

## Troubleshooting

| Error                  | Root Cause -> Fix                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| Invalid project_id     | Wrong or missing project_id in URL path. Use `hcloud IAM KeystoneListProjects` or KooCLI profile |
| 401/AuthFailure        | Expired AK/SK or wrong signing algorithm. Regenerate AK/SK or check SDK version                  |
| 403/Forbidden          | IAM permission missing. Check user/role has required action permissions                          |
| Empty results          | Pagination missing — add `limit` and check `marker`                                              |
| Endpoint not reachable | Wrong region/service endpoint. Verify at developer.huaweicloud.com/endpoint                      |

## Terraform Priority

Terraform Provider is low priority in V1. Mention it when the user needs reviewed, repeatable infrastructure changes. Do not default to Terraform for quick discovery or app-level SDK integration.

## Cross-Skill References

- **CLI setup**: See `huaweicloud-cli-and-auth` for KooCLI credential setup
- **Safety**: See `huaweicloud-safety` for write operation approval flow
