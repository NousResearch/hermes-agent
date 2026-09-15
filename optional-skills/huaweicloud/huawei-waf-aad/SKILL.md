---
name: huawei-waf-aad
description: 'Use when configuring Web Application Firewall (WAF) policies/rules, or Anti-DDoS (AAD) protection on Huawei Cloud. Triggers: WAF, AAD, firewall, DDoS, web protection, IP blacklist, rate limiting, CC attack. NOT for: security groups (use huawei-vpc), CTS audit (use huawei-cts).'
version: 1
---

# Huawei Cloud WAF / AAD

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud WAF <Operation> --help` before constructing commands to discover exact parameter names and requirements. For AAD operations, use `hcloud AAD --help`.

## Overview

Domain expertise for WAF and Anti-DDoS. Covers WAF policy/rules, AAD protection, IP whitelist/blacklist, and rate limiting.

## Critical Warnings

| Trap                       | Why                                                               |
| -------------------------- | ----------------------------------------------------------------- |
| WAF needs CNAME redirect   | DNS must point to WAF endpoint, not server IP                     |
| Premium instance required  | Cloud WAF needs a dedicated/premium WAF instance                  |
| AAD Standard vs Enterprise | Standard protects single IP. Enterprise protects entire IP ranges |
| Rule order matters         | WAF rules evaluated top-to-bottom within a policy                 |

## Common Workflows

### WAF

| Task                | Operation                                                       |
| ------------------- | --------------------------------------------------------------- |
| List WAF instances  | `ListInstances --cli-region=<r> --project_id=<p>`               |
| List policies       | `ListPolicies --cli-region=<r> --project_id=<p>`                |
| Create custom rule  | `BatchCreateCustomRule --cli-region=<r> --project_id=<p>`       |
| Create IP blacklist | `BatchCreateWhiteblackipRule --cli-region=<r> --project_id=<p>` |
| Create CC rule      | `BatchCreateCcRule --cli-region=<r> --project_id=<p>`           |
| Create geo rule     | `BatchCreateGeoIpRule --cli-region=<r> --project_id=<p>`        |

### AAD

```bash
hcloud AAD ListInstances --cli-region=<r> --project_id=<p>
hcloud AAD CreateInstance --cli-region=<r> --project_id=<p>
```

## Troubleshooting

| Error                           | Fix                                                      |
| ------------------------------- | -------------------------------------------------------- |
| Website unreachable after WAF   | Check CNAME record points to WAF endpoint                |
| WAF blocking legitimate traffic | Check rules order, adjust false positive settings        |
| AAD not responding              | Standard AAD only protects single EIP. Check EIP binding |

## Security

- MUST use WAF for all public-facing web applications
- SHOULD enable AAD for DDoS protection
- MUST test WAF rules in report-only mode before blocking

## Cross-Skill References

- **EIP**: See `huawei-vpc` for elastic IP binding
- **DNS/CNAME**: Configure via your DNS provider
