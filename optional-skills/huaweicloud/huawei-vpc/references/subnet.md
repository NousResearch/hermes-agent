# VPC Subnet Guide

## Prerequisite: VPC CIDR is Immutable

VPC CIDR block cannot be changed after creation. Plan carefully.

## KooCLI 7.x Nested Parameter Format

VPC API uses nested prefix: `--vpc.name`, `--subnet.vpc_id`, `--subnet.name`. NOT `--name` or `--vpc_id`.

```bash
# CORRECT
hcloud VPC CreateVpc --vpc.name=<name> --vpc.cidr=192.168.0.0/16

hcloud VPC CreateSubnet \
  --subnet.name=<name> \
  --subnet.vpc_id=<vpc-id> \
  --subnet.cidr=192.168.1.0/24 \
  --subnet.gateway_ip=192.168.1.1 \
  --subnet.availability_zone=<az>

# WRONG — will fail with [USE_ERROR]
hcloud VPC CreateVpc --name=xxx
hcloud VPC CreateSubnet --vpc_id=xxx
```

## Key Constraints

| Constraint                | Detail                                                         |
| ------------------------- | -------------------------------------------------------------- |
| Subnet bound to single AZ | Cross-AZ needs multiple subnets. AZ code format: `cn-south-1a` |
| CIDR must not overlap     | Cannot overlap with other subnets in same VPC                  |
| Gateway IP auto-assigned  | Default is first IP in CIDR range (e.g., 192.168.1.1 for /24)  |
| 5 IPs reserved per subnet | First 4 + broadcast reserved by platform                       |

## Common Errors

| Error                          | Root Cause -> Fix                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------- |
| `[USE_ERROR]不正确的参数:name` | Missing nested prefix. Use `--vpc.name=` or `--subnet.name=`                       |
| CIDR overlap                   | Subnet CIDR must be within VPC CIDR and not overlap others                         |
| AZ not found                   | Use `hcloud ECS NovaListAvailabilityZones --cli-region=<r>` to list valid AZ codes |

## CIDR Planning Reference

| Environment | VPC CIDR       | Subnet CIDR                                      |
| ----------- | -------------- | ------------------------------------------------ |
| Dev         | 192.168.0.0/16 | 192.168.1.0/24                                   |
| Staging     | 10.0.0.0/16    | 10.0.1.0/24                                      |
| Production  | 172.16.0.0/16  | 172.16.1.0/24 (public) + 172.16.2.0/24 (private) |
