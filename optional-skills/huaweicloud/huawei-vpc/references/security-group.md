# Security Group Rules

## Security Group is NOT bound to VPC

VPC v3 API `CreateSecurityGroup` does NOT accept a `vpc_id` parameter. Security groups are region-level resources, not VPC-bound. They work across any VPC in the same region.

```bash
# CORRECT — no vpc_id needed
hcloud VPC CreateSecurityGroup --security_group.name=<name>

# WRONG
hcloud VPC CreateSecurityGroup --security_group.vpc_id=<vpc-id>
```

## KooCLI 7.x Nested Parameter Format

Security group params use `--security_group.` prefix. Rules use `--security_group_rule.` prefix.

## Create Security Group Rule

Complete example with nested prefix:

```bash
hcloud VPC CreateSecurityGroupRule \
  --security_group_rule.security_group_id=<sg-id> \
  --security_group_rule.direction=<direction> \
  --security_group_rule.protocol=<protocol> \
  --security_group_rule.multiport=<port> \
  --security_group_rule.remote_ip_prefix=<cidr>
```

> Always run `hcloud VPC CreateSecurityGroupRule --help` to verify param names.

## Stateful vs Stateless

| Feature    | Security Group                       | Network ACL                     |
| ---------- | ------------------------------------ | ------------------------------- |
| Stateful?  | Yes — return traffic auto-allowed    | No — must allow both directions |
| Applied to | ECS instance NIC                     | Subnet                          |
| Rules      | Allow only                           | Allow and Deny                  |
| Default    | Deny all inbound, allow all outbound | Allow all                       |

## Common Rules Reference

| Direction | Protocol | Port | Source         | Purpose          |
| --------- | -------- | ---- | -------------- | ---------------- |
| Ingress   | TCP      | 22   | <office-ip>/32 | SSH              |
| Ingress   | TCP      | 80   | 0.0.0.0/0      | HTTP             |
| Ingress   | TCP      | 443  | 0.0.0.0/0      | HTTPS            |
| Ingress   | TCP      | 3306 | <sg-app-id>    | MySQL from app   |
| Ingress   | TCP      | 6379 | <sg-app-id>    | Redis from app   |
| Egress    | ALL      | ALL  | 0.0.0.0/0      | Outbound default |

## Common Errors

| Error                            | Root Cause -> Fix                                    |
| -------------------------------- | ---------------------------------------------------- |
| Cannot SSH                       | Default SG denies all inbound. Add port 22 rule      |
| Rule not effective               | SG assigned to wrong instance. Verify NIC attachment |
| `[USE_ERROR]不正确的参数:vpc_id` | Security groups don't need vpc_id. Remove it         |
