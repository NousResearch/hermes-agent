---
name: huawei-vpc
description: 'Use when creating, configuring, or managing VPC networks, subnets, security groups, EIPs, NAT gateways, VPN connections, or network ACLs on Huawei Cloud. Triggers on: VPC, subnet, security group, EIP, NAT, VPN, network ACL, route table, bandwidth. NOT for: DNS or CDN configuration.'
version: 1
---

# Huawei Cloud VPC

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud <Service> <Operation> --help` before constructing commands to discover exact parameter names and requirements.

> **Multi-version APIs**: KooCLI may print a warning like "ListVpcs有多个版本,默认使用该API版本v3" before the actual response. The text BEFORE the first `{` is the version selection notice — parse JSON starting from `{` only. This is normal behavior, not an error.

## Overview

Domain expertise for Huawei Cloud Virtual Private Cloud (VPC). Covers VPC/subnet lifecycle, security groups, EIP management, NAT gateways, VPN, and network ACLs.

## Critical Warnings

| Trap                                  | Why                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VPC CIDR cannot change                | Once set, VPC CIDR block is immutable                                                                                                                                                                                                                                                                                                                                                              |
| Security group stateful               | SG rules are stateful. Return traffic auto-allowed                                                                                                                                                                                                                                                                                                                                                 |
| Network ACL stateless                 | ACL rules must allow both inbound AND outbound                                                                                                                                                                                                                                                                                                                                                     |
| EIP bills when idle                   | Unbound EIP still charges. Release when unused                                                                                                                                                                                                                                                                                                                                                     |
| Subnet AZ binding                     | Subnet tied to single AZ. Cross-AZ needs multiple subnets                                                                                                                                                                                                                                                                                                                                          |
| EIP PER type needs `--bandwidth.name` | PER bandwidth requires explicit name; `--help` marks it optional but it's required                                                                                                                                                                                                                                                                                                                 |
| **VPC params need nested prefix**     | KooCLI 7.x VPC API uses `--vpc.<param>`, `--subnet.<param>`, `--security_group.<param>`. Example: `--vpc.name=xxx` NOT `--name=xxx`                                                                                                                                                                                                                                                                |
| **Security group needs no vpc_id**    | VPC v3 API `CreateSecurityGroup` does NOT accept `vpc_id`. Security groups are region-level, not VPC-bound                                                                                                                                                                                                                                                                                         |
| Subnet DNS empty → ECS no DNS         | DNS params (`--subnet.primary_dns`, `--subnet.secondary_dns`) marked optional but empty default breaks cloud-init domain resolution — `yum`/`apt` installs fail silently. Always set both. Common DNS IPs: cn-north-4 (100.125.1.250 / 100.125.1.251), cn-north-1 (100.125.1.250 / 100.125.129.250), cn-east-3 (100.125.1.250 / 100.125.129.250), ap-southeast-3 (100.125.1.250 / 100.125.128.250) |
| SCP blocks 0.0.0.0/0 SG rules         | If `CreateSecurityGroupRule` with `--remote_ip_prefix=0.0.0.0/0` fails with `SYS.0403`, an org-level SCP policy is denying wide-open rules. Narrow to a specific CIDR range (e.g., your office IP) instead                                                                                                                                                                                         |
| **VPC tags use `*` separator**        | VPC tag format: `--vpc.tags.1=env*test` (asterisk between key and value). NOT `--vpc.tags.1.key=env` (ECS-style). This is different from ECS `--server.metadata.key=value`                                                                                                                                                                                                                         |

## Common Workflows

| Task            | Command                                                                                                                                                                                                                                                    | Steps                                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Create VPC      | hcloud VPC CreateVpc --vpc.name=<name> --vpc.cidr=<cidr>                                                                                                                                                                                                   | CIDR must not conflict with existing VPCs. Run `hcloud VPC ListVpcs` first                                           |
| Create subnet   | hcloud VPC CreateSubnet --subnet.name=<name> --subnet.vpc_id=<id> --subnet.cidr=<cidr> --subnet.gateway_ip=<gw> --subnet.primary_dns=<dns1> --subnet.secondary_dns=<dns2> --subnet.availability_zone=<az>                                                  | Subnet CIDR must be a subset of the VPC CIDR. DNS addresses vary by region — see Critical Warnings for common values |
| Update subnet   | hcloud VPC UpdateSubnet --subnet_id=<id> --subnet.dnsList.1=<dns1> --subnet.dnsList.2=<dns2>                                                                                                                                                               | Fix DNS after creation. Restart ECS after updating DNS for cloud-init to pick up changes                             |
| Security group  | hcloud VPC CreateSecurityGroup --security_group.name=<name>                                                                                                                                                                                                | references/security-group.md                                                                                         |
| SG rule         | hcloud VPC CreateSecurityGroupRule --security_group_rule.security_group_id=<id> --security_group_rule.direction=<direction> --security_group_rule.protocol=<protocol> --security_group_rule.multiport=<port> --security_group_rule.remote_ip_prefix=<cidr> | references/security-group.md                                                                                         |
| Create EIP      | hcloud EIP CreatePublicip --publicip.type=<type> --bandwidth.size=<size> --bandwidth.share_type=<share-type> --bandwidth.name=<name>                                                                                                                       | Run `hcloud EIP CreatePublicip --help` to confirm valid type values per region                                       |
| Bind EIP to ECS | hcloud EIP AssociatePublicips --publicip_id=<id> --publicip.associate_instance_id=<port-id> --publicip.associate_instance_type=PORT                                                                                                                        | Get port ID from `hcloud ECS ListServersDetails --server_id=<id>` → `OS-EXT-IPS:port_id`                             |
| Unbind EIP      | hcloud EIP DisassociatePublicip --publicip_id=<id>                                                                                                                                                                                                         | references/eip.md                                                                                                    |
| Delete EIP      | hcloud EIP DeletePublicip --publicip_id=<id>                                                                                                                                                                                                               | references/eip.md                                                                                                    |
| List EIPs       | hcloud EIP ListPublicips                                                                                                                                                                                                                                   |                                                                                                                      |
| NAT gateway     | hcloud NAT CreateNatGateway --nat.name=<name> --nat.spec=<spec> --router_id=<vpc-id> --internal_network_id=<subnet-id>                                                                                                                                     | Run `hcloud NAT CreateNatGateway --help` for available spec values                                                   |

## Troubleshooting

| Error                                              | Root Cause -> Fix                                                                                                                                                                                                                            |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cannot reach instance                              | SG missing rule or no EIP -> Add SG rule / Bind EIP                                                                                                                                                                                          |
| Subnet CIDR conflict                               | Overlapping with existing subnets -> Choose non-overlapping CIDR                                                                                                                                                                             |
| NAT gateway no internet                            | Route table missing default route -> Add 0.0.0.0/0 via NAT                                                                                                                                                                                   |
| EIP quota exceeded                                 | Check actual quota: `hcloud EIP ListPublicips --cli-region=<r>` to see current usage. Default varies by account (typically 5-10)                                                                                                             |
| EIP.7905                                           | Run `hcloud EIP ListPublicips --cli-region=<r>` first to check current usage                                                                                                                                                                 |
| VPC.0301: Bandwidth name invalid                   | PER type requires `--bandwidth.name`, even though `--help` marks it optional                                                                                                                                                                 |
| EIP has no public IP after binding                 | May need AddIngressEipV2 for ELB-type resources (see huawei-apig)                                                                                                                                                                            |
| ECS cloud-init fails silently (port 80/443 closed) | Subnet likely has no DNS. Check `hcloud VPC ShowSubnet --subnet_id=<id>` → `dnsList` empty? Rebuild subnet with `--subnet.primary_dns=<dns1> --subnet.secondary_dns=<dns2>`. DNS addresses per region: `hcloud VPC CreateSubnet --help`      |
| VPC.0209: Subnet still used                        | Subnet has dependent resources (ECS/RDS) — delete instances first, then subnet                                                                                                                                                               |
| SYS.0403 / SCP deny                                | Service Control Policy explicitly denies this operation — contact org admin to adjust SCP, or use an account/region without the restriction. If SSH is blocked, bootstrap via cloud-init user_data instead: see `huawei-ecs` — no SSH needed |

## Security Considerations

- MUST use security groups, NOT iptables
- MUST scope SG rules to specific CIDRs, NOT 0.0.0.0/0
- SHOULD use network ACLs as defense-in-depth
- SHOULD place databases in private subnets (no EIP)
- MUST enable VPC flow logs for audit

## MCP Tools

- huaweicloud_list_operations service=VPC
- huaweicloud_run_readonly_command for VPC/subnet discovery
- huaweicloud_run_approved_command for writes

## References

- VPC Docs: https://support.huaweicloud.com/vpc/
- Subnet guide: references/subnet.md
- Security group: references/security-group.md
