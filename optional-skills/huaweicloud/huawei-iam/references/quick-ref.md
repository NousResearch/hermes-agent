# IAM Quick Reference

## Common Operations

```bash
hcloud IAM KeystoneListProjects                  # list projects
hcloud IAM KeystoneListUsers                     # list users
hcloud IAM KeystoneListGroups                    # list groups
hcloud IAM ListCustomPolicies                    # list custom policies
hcloud IAM ListAgencies                          # list agencies
hcloud IAM KeystoneListAuthDomains               # list domains
hcloud IAM KeystoneShowUser                      # show user detail
```

## Credential Management

```bash
hcloud IAM KeystoneListAccessKeys --user_id=<id> # list AK/SK for user
hcloud IAM CreateLoginToken                      # create login token
hcloud IAM GetAccountSummary                     # account summary
```

## Security Best Practices

1. **NEVER create IAM users** — use IAM Identity Center or temporary STS tokens
2. **NEVER create long-term AK/SK** — use temporary credentials
3. **Least privilege by default** — start empty, add only needed actions
4. **Scope resources** — no `*` wildcards on Resource
5. **Use condition keys** — `g:RequestedRegion`, `g:ResourceTag`, `g:CurrentTime`
6. **Rotate AK/SK every 90 days** — or use agency delegation

## IAM vs AWS Comparison

| AWS                   | Huawei Cloud                         |
| --------------------- | ------------------------------------ |
| `iam:ListUsers`       | `iam:users:list` (lowercase + colon) |
| AWS Managed Policy    | System Policy (系统策略)             |
| Customer Managed      | Custom Policy (自定义策略)           |
| Resource-Based Policy | Project-Level Policy (项目级策略)    |
| IAM Role              | Agency (委托)                        |

> Huawei Cloud IAM action naming: `<service>:<resource>:<action>`. Always verify at https://support.huaweicloud.com/usermanual-iam/iam_01_0001.html

## Confused Deputy Protection

```json
{
  "Effect": "Allow",
  "Action": ["ecs:*"],
  "Resource": ["*"],
  "Condition": {
    "StringEquals": { "g:SourceAccount": "<account-id>" },
    "StringLike": { "g:SourceUrn": "urn:fss:*" }
  }
}
```
