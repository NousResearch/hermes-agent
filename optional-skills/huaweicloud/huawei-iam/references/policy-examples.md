# IAM Policy Examples

## Huawei Cloud IAM Policy Format

Huawei Cloud IAM uses a different action naming convention from AWS. Actions use lowercase service prefixes:

```
<service>:<resource>:<action>
```

Examples:

- `ecs:servers:list` (NOT `ecs:ListInstances`)
- `obs:bucket:GetBucket` (NOT `s3:GetBucket`)
- `vpc:vpcs:list` (NOT `vpc:DescribeVpcs`)

Always verify exact action names via `https://support.huaweicloud.com/usermanual-iam/iam_01_0001.html` or KooCLI `--help`.

## Read-Only Policies

### ECS Read-Only

```json
{
  "Version": "1.1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ecs:servers:list", "ecs:servers:get", "ecs:*:list", "ecs:*:get"],
      "Resource": ["*"]
    }
  ]
}
```

### OBS Read-Only (scoped bucket)

```json
{
  "Version": "1.1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["obs:bucket:Get*", "obs:bucket:List*", "obs:object:Get*"],
      "Resource": ["obs:*:*:bucket:my-bucket", "obs:*:*:bucket:my-bucket/*"]
    }
  ]
}
```

### FunctionGraph Read-Only

```json
{
  "Version": "1.1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "functiongraph:function:getConfig",
        "functiongraph:function:list",
        "functiongraph:function:invoke",
        "functiongraph:trigger:list"
      ],
      "Resource": ["*"]
    }
  ]
}
```

## Full-Access (Scoped)

### RDS Operator

```json
{
  "Version": "1.1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["rds:*"],
      "Resource": ["*"],
      "Condition": { "StringEquals": { "g:ResourceTag/Environment": ["dev", "staging"] } }
    }
  ]
}
```

### FunctionGraph Developer

```json
{
  "Version": "1.1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["functiongraph:function:*", "functiongraph:trigger:*", "functiongraph:runtime:list"],
      "Resource": ["*"]
    }
  ]
}
```

### APIG Operator

```json
{
  "Version": "1.1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["apig:instance:*", "apig:api:*", "apig:group:*", "apig:environment:*"],
      "Resource": ["*"]
    }
  ]
}
```

## Best Practices

1. **Least privilege**: Start with empty policy, add only needed actions
2. **No `*` wildcard on Resource**: Scope to specific resources or use tag conditions
3. **Use Condition keys**: `g:RequestedRegion`, `g:ResourceTag`, `g:CurrentTime` for context-aware policies
4. **Separate policies per role**: Don't combine read and write in one policy

## Confused Deputy Protection

```json
{
  "Version": "1.1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ecs:*"],
      "Resource": ["*"],
      "Condition": {
        "StringEquals": { "g:SourceAccount": "0123456789" },
        "StringLike": { "g:SourceUrn": "urn:fss:*" }
      }
    }
  ]
}
```

## References

- IAM Action Reference: https://support.huaweicloud.com/usermanual-iam/iam_01_0001.html
- Policy Syntax: https://support.huaweicloud.com/usermanual-iam/iam_01_0602.html
