---
name: kyverno-policy-engineering
description: Author and roll out Kyverno CEL admission policies.
version: 1.1.0
author: Yasser Alattas (@yalattas), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kyverno, kubernetes, policy-as-code, admission-control, security, gitops]
    category: devops
    requires_toolsets: [terminal]
---

# Kyverno Policy Engineering Skill

Write, review, test, and roll out Kyverno policies on the CEL-based `policies.kyverno.io/v1` APIs
(`ValidatingPolicy`, `MutatingPolicy`, `GeneratingPolicy`, `ImageValidatingPolicy`,
`DeletingPolicy`, and their `Namespaced*` variants). The legacy `kyverno.io/v1` `Policy` /
`ClusterPolicy` APIs are deprecated as of Kyverno 1.17; this skill covers them only for migration.

## When to Use

- Writing Kyverno CEL policies for Kubernetes admission control.
- Migrating legacy `Policy` / `ClusterPolicy` manifests to the new APIs.
- Reviewing policy scope, failure mode, and rollout safety.
- Implementing Pod Security Standards, registry allow-lists, or image signature checks.
- Designing narrow, reviewed policy exceptions.
- Rolling a policy out from `Audit` to `Deny`.
- Debugging admission webhook failures or policy report surprises.

Not for: NetworkPolicy traffic segmentation, OPA Gatekeeper/Rego authoring, or general
Kubernetes troubleshooting unrelated to admission control.

## Prerequisites

- `kubectl` with a kubeconfig context for the target cluster, run through the `terminal` tool.
- Kyverno 1.17 or later installed in the cluster (the `policies.kyverno.io` CRDs must exist).
- The `kyverno` CLI for offline tests (`kyverno test`, `kyverno apply`).
- Read access to the GitOps repository that holds the policies, if one exists.

## How to Run

1. Use `read_file` on existing policy manifests and `search_files` to locate legacy
   `kind: ClusterPolicy` / `apiVersion: kyverno.io/v1` manifests that need migrating.
2. Use `terminal` for `kubectl` and `kyverno` commands against the cluster or local fixtures.
3. Use `patch` (or `write_file` for new files) to edit policy YAML and test fixtures.

## Quick Reference

Use `apiVersion: policies.kyverno.io/v1` for all new policies.

| Kind | Purpose |
|------|---------|
| `ValidatingPolicy` | Validate resources; deny or audit violations. |
| `MutatingPolicy` | Mutate matching resources. |
| `GeneratingPolicy` | Generate resources from policy logic. |
| `ImageValidatingPolicy` | Verify image registry, signatures, attestations, metadata. |
| `DeletingPolicy` | Control cleanup/deletion behaviour. |

Each kind has a `Namespaced*` variant (`NamespacedValidatingPolicy`, ...). Use cluster-wide kinds
when a platform/security team owns the control globally; use namespaced kinds when tenant
namespaces own or stage the policy themselves.

```bash
kubectl get validatingpolicies,mutatingpolicies,generatingpolicies,imagevalidatingpolicies,deletingpolicies -A
kubectl get namespacedvalidatingpolicies,namespacedmutatingpolicies,namespacedgeneratingpolicies,namespacedimagevalidatingpolicies,namespaceddeletingpolicies -A
kubectl get policyreport,clusterpolicyreport -A
kyverno test <policy-dir>
kyverno apply <policy.yaml> --resource <resource.yaml>
```

## Procedure

### 1. Clarify intent

Answer before writing YAML: which risk or compliance control is reduced, which resources are
targeted, cluster-wide or namespace-owned, which namespaces/service accounts/controllers are exempt,
and what breakage is acceptable during rollout.

### 2. Inspect current Kyverno state

```bash
kubectl get pods -n kyverno
kubectl api-resources --api-group=policies.kyverno.io
kubectl get policyreport,clusterpolicyreport -A
kubectl logs -n kyverno deploy/kyverno-admission-controller --tail=200
kubectl logs -n kyverno deploy/kyverno-background-controller --tail=200
```

If `kubectl api-resources` lists no kinds for `policies.kyverno.io`, the installed Kyverno predates
the CEL APIs. Confirm the version (`kyverno version`) before writing policies against them.

### 3. Draft the policy

- One focused control per policy; no large mixed-control policies.
- Explicit `matchConstraints.resourceRules` (groups, versions, resources, operations).
- `validationActions: [Audit]` first; `Deny` only after evidence.
- Readable CEL with a clear `message`; guard optional fields with `?` / `orValue()`.

```yaml
apiVersion: policies.kyverno.io/v1
kind: ValidatingPolicy
metadata:
  name: require-environment-label
spec:
  validationActions:
    - Audit
  matchConstraints:
    resourceRules:
      - apiGroups: [""]
        apiVersions: ["v1"]
        operations: ["CREATE", "UPDATE"]
        resources: ["pods"]
  validations:
    - message: "Pods must set the environment label."
      expression: "'environment' in object.metadata.?labels.orValue({})"
```

Common patterns and their breakage risk:

- **Resource requests/limits:** audit first; exclude one-shot jobs and migrations until inventoried.
- **Privileged containers:** high value; exclude trusted infrastructure namespaces only if unavoidable.
- **hostPath / hostNetwork / hostPID / hostIPC:** high breakage risk for node agents, CNI, and
  storage drivers; build a reviewed exception list.
- **Approved registries:** inventory current images first, including system images and mirrors.
- **Signed images:** `ImageValidatingPolicy` with Sigstore/Cosign; start with critical workloads.
- **Required labels/annotations:** keep the taxonomy small and documented.

### 4. Test before rollout

Add at least one allowed and one denied/audited fixture per policy, then:

```bash
kyverno test <policy-dir>
kyverno apply <policy.yaml> --resource <resource.yaml>
kubectl apply --dry-run=server -f <resource.yaml>
```

### 5. Roll out safely

1. Apply in `Audit` mode.
2. Read `PolicyReport` / `ClusterPolicyReport` results.
3. Fix workloads or add narrow, time-bound, reviewed exceptions.
4. Switch low-risk namespaces to `Deny` first, then expand.
5. Watch admission controller errors and deployment failure rates.

### 6. Report back

When writing or reviewing a policy, state: intent, exact `apiVersion`/`kind`, mode
(`Audit`/`Deny`/staged), matched and excluded scope, breakage risk with reason, test cases, and the
rollout plan.

## Pitfalls

- **Wrong API generation:** new policies must not use `kind: ClusterPolicy` or
  `apiVersion: kyverno.io/v1`.
- **Matching too broadly:** resource rules, namespaces, or selectors wider than intended.
- **Background scans surprise teams:** reports include existing resources, not only new admissions.
- **Webhook timeouts block deploys:** admission controller health and failure policy matter once
  enforcing.
- **Exclusion selectors miss:** labels on the admitted object differ from what the team assumed.
- **Controller-generated objects differ:** the Pod admitted by Kubernetes is not the Deployment
  manifest in git.
- **Namespaced vs cluster-wide mismatch:** tenant-owned controls need a `Namespaced*` kind; platform
  baselines need a cluster-wide kind.
- **Debug commands on old names:** `kubectl get clusterpolicy` does not show CEL policies.

```bash
kubectl describe validatingpolicy <name>
kubectl get clusterpolicyreport -o yaml
kubectl logs -n kyverno deploy/kyverno-admission-controller --tail=300
```

## Verification

- [ ] Policy uses `policies.kyverno.io/v1` and the intended kind.
- [ ] CEL expressions handle missing/null fields.
- [ ] At least one pass and one fail/audit fixture exists.
- [ ] `kyverno test` or a server-side dry run passed.
- [ ] Rollout starts in `Audit` unless the user explicitly asked for immediate enforcement.
- [ ] Exceptions are narrower than the policy scope and documented.
