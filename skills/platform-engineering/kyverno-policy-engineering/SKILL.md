---
name: kyverno-policy-engineering
description: Use when authoring, reviewing, testing, or rolling out Kyverno CEL policy APIs for Kubernetes admission control, including ValidatingPolicy, MutatingPolicy, GeneratingPolicy, ImageValidatingPolicy, DeletingPolicy, namespaced variants, exceptions, and audit-to-enforce rollout. Legacy Policy/ClusterPolicy APIs are deprecated as of Kyverno 1.17.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [kyverno, kubernetes, policy-as-code, admission-control, security, gitops]
    related_skills: [kubernetes-network-policy-with-calico, implementing-opa-gatekeeper-for-policy-enforcement]
---

# Kyverno Policy Engineering

## Overview

Kyverno is a Kubernetes-native policy engine for admission control, mutation, generation, image verification, and policy reporting. This skill focuses on the production-ready CEL-based policy APIs in the `policies.kyverno.io/v1` group.

As of Kyverno 1.17, the legacy `Policy` and `ClusterPolicy` APIs are deprecated. Prefer the new policy kinds for new work and migrations unless you are explicitly maintaining an older Kyverno installation.

## When to Use

Use this skill for:

- Writing Kyverno CEL policies for Kubernetes admission control.
- Migrating legacy Kyverno `Policy` / `ClusterPolicy` manifests to the new APIs.
- Reviewing policy scope, failure mode, and rollout safety.
- Implementing Pod Security Standards with Kyverno.
- Restricting privileged containers, host namespaces, host paths, capabilities, or unsafe volume types.
- Enforcing approved image registries, signatures, attestations, or provenance controls.
- Creating mutation, generation, image validation, or deletion-control policies.
- Designing explicit, narrow, reviewed policy exceptions.
- Rolling policies out gradually from audit/reporting to deny/enforcement.
- Debugging Kyverno admission webhook failures or policy report surprises.

Do not use this skill for:

- Kubernetes NetworkPolicy traffic segmentation; use a networking policy skill instead.
- OPA Gatekeeper/Rego policy authoring unless comparing approaches.
- General Kubernetes troubleshooting unrelated to admission/policy enforcement.

## Current Kyverno Policy APIs

Use `apiVersion: policies.kyverno.io/v1` for new Kyverno CEL policies.

Cluster-wide kinds:

- `ValidatingPolicy` — validate resources and deny or audit violations.
- `MutatingPolicy` — mutate matching resources.
- `GeneratingPolicy` — generate resources from policy logic.
- `ImageValidatingPolicy` — verify image registry, signature, attestation, and metadata rules.
- `DeletingPolicy` — control cleanup/deletion behaviors.

Namespaced ownership variants:

- `NamespacedValidatingPolicy`
- `NamespacedMutatingPolicy`
- `NamespacedGeneratingPolicy`
- `NamespacedImageValidatingPolicy`
- `NamespacedDeletingPolicy`

Prefer cluster-wide policies when a platform/security team owns the control globally. Prefer namespaced variants when tenant namespaces should own or stage the policy independently.

## Core Principles

1. Start with observation: know what workloads currently do before denying anything.
2. Prefer small, focused policies over large mixed-control policies.
3. Scope policies carefully with `matchConstraints`, selectors, and namespace strategy.
4. Use `validationActions: [Audit]` before `Deny` unless the blast radius is known and intentionally small.
5. Exclude system and platform namespaces only when justified; document why.
6. Make exceptions explicit, narrow, time-bound, and reviewed.
7. Treat policies as GitOps-managed production controls: reviewed, tested, versioned, observable.
8. Never enforce a policy until workload impact is understood.

## Standard Workflow

### 1. Clarify intent

Answer these before writing YAML:

- What risk or compliance control is being reduced?
- Which Kubernetes resources are targeted?
- Is this cluster-wide or namespace-owned?
- Which namespaces, service accounts, controllers, or workloads are exempt?
- Should violations be audited, denied, or rolled out namespace-by-namespace?
- What breakage is acceptable during rollout?

### 2. Inspect current Kyverno state

```bash
kubectl get pods -n kyverno
kubectl get validatingpolicies,mutatingpolicies,generatingpolicies,imagevalidatingpolicies,deletingpolicies -A
kubectl get namespacedvalidatingpolicies,namespacedmutatingpolicies,namespacedgeneratingpolicies,namespacedimagevalidatingpolicies,namespaceddeletingpolicies -A
kubectl get policyreport,clusterpolicyreport -A
kubectl logs -n kyverno deploy/kyverno-admission-controller --tail=200
kubectl logs -n kyverno deploy/kyverno-background-controller --tail=200
```

If the new kinds are not recognized, confirm the installed Kyverno version and CRDs before continuing:

```bash
kyverno version
kubectl get crd | grep policies.kyverno.io
```

### 3. Draft policy

For validation policies:

- Use `apiVersion: policies.kyverno.io/v1`.
- Use `kind: ValidatingPolicy` for cluster-wide ownership.
- Use `kind: NamespacedValidatingPolicy` for namespace-owned policy.
- Use `validationActions: [Audit]` first, then `Deny` after evidence.
- Use explicit `matchConstraints.resourceRules`.
- Keep CEL expressions readable and add a clear `message`.

Minimal example:

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

### 4. Test before rollout

Use local tests and server-side dry runs where possible:

```bash
kyverno test <policy-dir>
kyverno apply <policy.yaml> --resource <resource.yaml>
kubectl apply --dry-run=server -f <resource.yaml>
```

For GitOps repositories, add at least one allowed and one denied/audited fixture for every policy.

### 5. Roll out safely

1. Apply in `Audit` mode.
2. Observe `PolicyReport` and `ClusterPolicyReport` results.
3. Fix workloads or add narrow reviewed exceptions.
4. Move low-risk namespaces to `Deny` first.
5. Expand enforcement gradually.
6. Monitor admission controller errors and deployment failure patterns.

## Review Checklist

For every Kyverno policy, verify:

- **Intent:** The risk/control is explicit and worth enforcing.
- **API:** New policies use `policies.kyverno.io/v1`, not deprecated `kyverno.io/v1` `Policy` / `ClusterPolicy`.
- **Correctness:** It matches the intended groups, versions, resources, and operations.
- **Scope:** System/platform namespace handling is intentional and documented.
- **Failure mode:** `Audit` vs `Deny` is appropriate for rollout phase.
- **CEL readability:** Expressions are understandable and produce clear messages.
- **Exceptions:** Exemptions are narrow, justified, and reviewed.
- **GitOps:** Policy is declarative, versioned, and reviewed.
- **Observability:** Policy reports and Kyverno controller logs are monitored.
- **Security value:** The policy materially reduces a real risk.

## Common Policy Patterns

### Require resource requests and limits

Use for production namespaces. Start in audit mode and consider excluding one-shot jobs, migrations, and system workloads until inventory is complete.

### Disallow privileged containers

High-value control. Match pods and workload controllers as needed. Exclude trusted infrastructure namespaces only where unavoidable.

### Restrict hostPath, hostNetwork, hostPID, and hostIPC

High breakage risk for node agents and CNI/storage components. Audit first and build a reviewed exception list.

### Require approved image registries

Inventory current images before enforcement. Avoid breaking system images, private registries, and image mirrors used by cluster components.

### Verify signed images

Use Kyverno image validation policies with Sigstore/Cosign or an approved keyless trust policy. Start with critical workloads, then expand.

### Enforce required labels or annotations

Useful for ownership, cost allocation, environment tagging, and incident response. Keep the label taxonomy simple and documented.

## Debugging

```bash
kubectl describe validatingpolicy <name>
kubectl get validatingpolicies,mutatingpolicies,generatingpolicies,imagevalidatingpolicies,deletingpolicies -A
kubectl get namespacedvalidatingpolicies,namespacedmutatingpolicies,namespacedgeneratingpolicies,namespacedimagevalidatingpolicies,namespaceddeletingpolicies -A
kubectl get policyreport -A
kubectl get clusterpolicyreport -o yaml
kubectl logs -n kyverno deploy/kyverno-admission-controller --tail=300
kubectl logs -n kyverno deploy/kyverno-background-controller --tail=300
```

Common issues:

1. **Wrong API generation:** New CEL policy examples should not use legacy `kind: ClusterPolicy` or `apiVersion: kyverno.io/v1`.
2. **Policy matches too broadly:** Resource rules, namespaces, or selectors are too wide.
3. **Background scans surprise teams:** Reports can include existing resources that were not newly admitted.
4. **Webhook timeout blocks deploys:** Admission controller health and timeout/failure policy matter during enforcement.
5. **Exclusion selectors are wrong:** Label/namespace selectors do not match what the workload actually has.
6. **Controller-generated resources behave differently:** The resource admitted by Kubernetes may differ from the user's source manifest.
7. **Namespaced vs cluster-wide mismatch:** A tenant-owned policy may need a namespaced kind; a platform baseline likely needs a cluster-wide kind.

## Response Format

When reviewing or writing Kyverno policy, respond with:

- **Intent:** what risk/control this implements.
- **API/kind:** exact `apiVersion` and `kind`.
- **Mode:** `Audit`, `Deny`, or staged rollout.
- **Scope:** matched and excluded resources/namespaces.
- **Breakage risk:** low/medium/high with reason.
- **Policy YAML:** if requested.
- **Test cases:** allowed and denied/audited examples.
- **Rollout plan:** audit → fix/exception review → enforce.

## Verification Checklist

- [ ] Policy uses the intended Kyverno API generation.
- [ ] CEL expressions were checked for obvious null/missing-field failures.
- [ ] At least one pass and one fail/audit case exists.
- [ ] Server-side dry run or Kyverno CLI test was run where possible.
- [ ] Rollout starts with audit/reporting unless the user explicitly requested immediate enforcement.
- [ ] Exceptions are narrower than the main policy scope and documented.
- [ ] Debug commands use new policy resource names, not deprecated `clusterpolicy` defaults.
