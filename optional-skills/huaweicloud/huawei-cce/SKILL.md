---
name: huawei-cce
description: 'Use when creating or managing CCE Kubernetes clusters. Covers cluster creation, node pools, SWR registry, autoscaling. Triggers: CCE, Kubernetes, K8s, cluster, node pool, container, SWR. NOT for: serverless functions (use huawei-functiongraph), serverless containers (use huawei-cce for CCI redirect).'
version: 1
---

# Huawei Cloud CCE

**STOP - Do not answer from general knowledge.** Follow the procedure below.

Always run `hcloud <Service> <Operation> --help` before constructing commands to discover exact parameter names and requirements.

## Overview

Domain expertise for CCE (Cloud Container Engine) and SWR (Software Repository for Container). CCE uses two hcloud services: `CCE` for clusters and `SWR` for container images.

## Critical Warnings

| Trap                                                    | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Huawei Cloud API returns errors one field at a time** | When creating clusters/services with multiple invalid fields, the API reports only the first error, not all errors at once. After fixing one field and resubmitting, the next call reveals the next error — iterating N times for N bad fields. This compounds with long cluster creation times (~10 min). To avoid this loop, run `hcloud CCE <Operation> --help` and validate every parameter with its constraints before the first call. Use `--cli-jsonInput` with a validated JSON file to reduce reformatting overhead between retries. |
| **KooCLI 7.x: CreateCluster/CreateNodePool broken**     | OPENAPI_ERROR in KooCLI 7.2.12. Use `CreateAutopilotCluster` for serverless, or Python SDK for VM clusters                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Cluster type immutable                                  | Cannot change hybrid/traditional after creation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Master managed by Huawei                                | No SSH to master. Use kubectl or kubectl-cce                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Network model affects pod IP                            | VPC network gives pods VPC IPs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Addon ops use UID not name                              | `ShowAddonInstance` returns `metadata.uid` for install/uninstall/update                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| SWR enterprise instance costs                           | `postPaid` billing. One-time activation at console required                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

## Common Workflows

| Task               | Operation                     | Service |
| ------------------ | ----------------------------- | ------- |
| List clusters      | `ListClusters`                | CCE     |
| Create cluster     | `CreateCluster`               | CCE     |
| Delete cluster     | `DeleteCluster`               | CCE     |
| Hibernate cluster  | `HibernateCluster`            | CCE     |
| List node pools    | `ListNodePools`               | CCE     |
| Create node pool   | `CreateNodePool`              | CCE     |
| List nodes         | `ListNodes`                   | CCE     |
| Get kubeconfig     | `CreateKubernetesClusterCert` | CCE     |
| List addons        | `ListAddonInstances`          | CCE     |
| Docker login (SWR) | `CreateAuthorizationToken`    | SWR     |
| Create SWR org     | `CreateNamespace`             | SWR     |
| Create SWR repo    | `CreateRepo`                  | SWR     |
| List repos         | `ListReposDetails`            | SWR     |

## SWR Image Push Workflow

**Prerequisite**: User/agency must have `sts::createServiceBearerToken` IAM permission. Without it, `CreateAuthorizationToken` returns `SVCSTG.SWR.4030170 Insufficient permissions`. Grant via IAM console or attach SWR Admin role.

```bash
# 1. Docker login to SWR
hcloud SWR CreateAuthorizationToken --help
docker login -u <region>@<AK> -p <token> swr.<region>.myhuaweicloud.com

# 2. Tag and push
docker tag my-app:latest swr.<region>.myhuaweicloud.com/<org>/my-app:latest
docker push swr.<region>.myhuaweicloud.com/<org>/my-app:latest

# 3. Verify
hcloud SWR ListReposDetails --cli-region=<r>
```

> SWR Auth token valid 12h. For CCE node pull access, create long-term credential with `CreateSecret` (valid 1 year).

## Serverless Containers (CCI)

For serverless containers without managing clusters, use CCI (Cloud Container Instance):

- No cluster needed — just namespace + network + workload
- Key ops: `CCI createCoreV1Namespace`, `CCI createNetworkingCciIoV1beta1NamespacedNetwork`, `CCI createAppsV1NamespacedDeployment`
- Namespace MUST include `namespace-kubernetes-io/flavor` annotation
- `limits == requests` strictly enforced (no overcommit)

See `hcloud CCI --help` for full operation list.

## Troubleshooting

| Error                            | Fix                                                                                                                                                                                                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kubectl connection refused       | Verify cluster Running; use `kubectl cce` (no EIP needed)                                                                                                                                                                                                         |
| Node pool creation failed        | Check VPC/subnet availability and flavor capacity                                                                                                                                                                                                                 |
| Docker push 401                  | Re-run `CreateAuthorizationToken` (token expired)                                                                                                                                                                                                                 |
| SVCSTG.SWR.4030170               | Missing `sts::createServiceBearerToken` IAM permission. Grant SWR Admin role or add policy                                                                                                                                                                        |
| Addon install fails              | Use `metadata.uid` from `ShowAddonInstance`, not name                                                                                                                                                                                                             |
| `kubectl cce` not found          | Install plugin: `kubectl cce` uses AK/SK, no kubeconfig required                                                                                                                                                                                                  |
| Serial field-by-field API errors | The API reports errors one field at a time. Fix a field, retry, get the next error — each cycle ~10 min for cluster operations. Mitigation: validate all parameters against `--help` constraints before the first call; use `--cli-jsonInput` for faster retries. |

## Security Considerations

- MUST restrict kubeconfig file permissions (0600)
- MUST use IAM RBAC for cluster access, not cluster-admin
- MUST store container images in private SWR repositories

## Cross-Skill References

- **VPC/Subnet**: See `huawei-vpc` for network prerequisites
- **EIP**: See `huawei-vpc` for cluster public access

## References

- CCE Docs: https://support.huaweicloud.com/cce/
- SWR Docs: https://support.huaweicloud.com/swr/
