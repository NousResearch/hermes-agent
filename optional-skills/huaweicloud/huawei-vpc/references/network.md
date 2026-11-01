# VPC Network Management Reference

## DNS Addresses by Region

Subnets without DNS configuration will fail cloud-init. Use these DNS addresses:

| Region         | Primary DNS   | Secondary DNS   |
| -------------- | ------------- | --------------- |
| cn-north-1     | 100.125.1.250 | 100.125.129.250 |
| cn-north-4     | 100.125.1.250 | 100.125.129.250 |
| cn-east-3      | 100.125.1.250 | 100.125.129.250 |
| cn-south-1     | 100.125.1.250 | 100.125.129.250 |
| ap-southeast-3 | 100.125.1.250 | 100.125.129.250 |

**Subnet create with DNS**:

```bash
hcloud VPC CreateSubnet \
  --subnet.name=<name> \
  --subnet.vpc_id=<id> \
  --subnet.cidr=10.50.1.0/24 \
  --subnet.gateway_ip=10.50.1.1 \
  --subnet.availability_zone=<az> \
  --subnet.dnsList.1=100.125.1.250 \
  --subnet.dnsList.2=100.125.129.250
```

## Nested Prefix Summary

KooCLI 7.x VPC API requires nested prefixes:

| Resource       | Create Param Prefix      | Example                                                               |
| -------------- | ------------------------ | --------------------------------------------------------------------- |
| VPC            | `--vpc.`                 | `--vpc.name=my-vpc --vpc.cidr=192.168.0.0/16`                         |
| Subnet         | `--subnet.`              | `--subnet.name=web --subnet.vpc_id=<id> --subnet.cidr=192.168.1.0/24` |
| Security Group | `--security_group.`      | `--security_group.name=sg-web`                                        |
| SG Rule        | `--security_group_rule.` | `--security_group_rule.direction=ingress`                             |

## EIP Management

```bash
# Create EIP (pay-per-use)
hcloud EIP CreatePublicip --publicip.type=<type> \
  --bandwidth.size=<size> --bandwidth.share_type=<share-type> --bandwidth.name=<name>

# Bind to ECS (PORT type, not ECS)
hcloud EIP AssociatePublicips --publicip_id=<id> \
  --publicip.associate_instance_id=<port-id> --publicip.associate_instance_type=PORT

# Unbind
hcloud EIP DisassociatePublicip --publicip_id=<id>

# List
hcloud EIP ListPublicips

# Delete (unbind first)
hcloud EIP DeletePublicip --publicip_id=<id>
```

> EIP bills even when not bound. Release unused EIPs.

## Security Group Quick Reference

```bash
# Create SG (no vpc_id needed)
hcloud VPC CreateSecurityGroup --security_group.name=<name>

# Add SSH rule
hcloud VPC CreateSecurityGroupRule \
  --security_group_rule.security_group_id=<sg-id> \
  --security_group_rule.direction=<direction> \
  --security_group_rule.protocol=<protocol> \
  --security_group_rule.multiport=<port> \
  --security_group_rule.remote_ip_prefix=<cidr>
```

> SG is stateful — return traffic auto-allowed. Network ACL is stateless — need both directions.
