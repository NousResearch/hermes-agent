# RAG Helm Deployment

If routed here from the deploy workflow, proceed directly to Phase 1.

## Phase 1: Prerequisites Check

Run all checks at once:

```bash
echo "=== KUBECTL ===" && kubectl version --client 2>/dev/null || echo "NO_KUBECTL"; echo "=== HELM ===" && helm version --short 2>/dev/null || echo "NO_HELM"; echo "=== STORAGECLASS ===" && kubectl get storageclass 2>/dev/null || echo "NO_STORAGECLASS"; echo "=== NODES ===" && kubectl get nodes -o wide 2>/dev/null || echo "NO_CLUSTER_ACCESS"; echo "=== GPU_OPERATOR ===" && kubectl get pods -n gpu-operator 2>/dev/null | grep -i running || echo "NO_GPU_OPERATOR"; echo "=== NIM_OPERATOR ===" && kubectl get pods -n nim-operator 2>/dev/null | grep -i running || echo "NO_NIM_OPERATOR"; echo "=== NAMESPACE ===" && kubectl get namespace rag 2>/dev/null && echo "NAMESPACE_EXISTS" || echo "NO_NAMESPACE"; echo "=== HELM_RELEASE ===" && helm list -n rag 2>/dev/null | grep rag || echo "NO_EXISTING_RELEASE"; echo "=== PODS ===" && kubectl get pods -n rag 2>/dev/null | head -10 || echo "NO_PODS"; echo "=== NGC_KEY ===" && [ -n "$NGC_API_KEY" ] && echo "NGC_API_KEY SET" || echo "NGC_API_KEY NOT_SET"; echo "=== GPU_RESOURCES ===" && kubectl get nodes -o json 2>/dev/null | grep -o '"nvidia.com/gpu": "[0-9]*"' || echo "NO_GPU_RESOURCES"
```

Read `docs/support-matrix.md` for current Kubernetes, Helm, and OS version requirements.

| Requirement | Check |
|-------------|-------|
| Kubernetes | Per `docs/support-matrix.md` |
| Helm | Per `docs/support-matrix.md` |
| NVIDIA GPU Operator | Installed and running |
| NVIDIA NIM Operator | Installed and running |
| Default StorageClass | Configured (e.g. local-path-provisioner) |
| Disk space | ≥200 GB per node |
| NGC_API_KEY | Set in environment |

Report all missing prerequisites together so the user can fix everything in one pass.

If NGC_API_KEY is NOT_SET: this is the one thing we must ask the user for.

If an existing Helm release is detected: warn "Existing RAG Helm release found. Proceeding will upgrade it." Continue unless user objects.

## Phase 2: Route to Reference

Auto-detect the GPU variant from cluster nodes (not the local machine):

```bash
echo "=== GPU_LABELS ===" && kubectl get nodes -o json 2>/dev/null | grep -oE '"nvidia.com/gpu.product":\s*"[^"]*"' | sort -u || echo "NO_GPU_LABELS"; echo "=== MIG ===" && kubectl get nodes -o json 2>/dev/null | grep -oE '"nvidia.com/mig.strategy":\s*"[^"]*"' || echo "NO_MIG"
```

Determine variant from node GPU labels:

Route based on detection:

- **MIG enabled** → read and follow `helm-mig.md`
- **RTX PRO 6000** → read and follow `helm-standard.md` (use the RTX values.yaml variant described there)
- **Standard (everything else)** → read and follow `helm-standard.md`

Ask the user only if the variant is genuinely ambiguous. Default to standard deployment.

## Phase 3: Expected Timelines

Set expectations with the user:

| Scenario | Duration |
|----------|----------|
| First deployment | 60–70 min (NIM cache download ~40–50 min, NIMService init ~10–15 min, pod startup ~5–10 min) |
| Subsequent deployments | 10–15 min (model caches already populated) |

Pods in `ContainerCreating` or `Init` state for extended periods is normal — models download in the background without progress indicators.

## Phase 4: Verification

After deployment completes, verify:

```bash
echo "=== PODS ===" && kubectl get pods -n rag; echo "=== NIMCACHE ===" && kubectl get nimcache -n rag; echo "=== NIMSERVICE ===" && kubectl get nimservice -n rag
```

Wait for all pods to reach `Running` status. Poll every 60 seconds for up to 70 minutes (first deployment involves model downloads). Show progress.

Once pods are running, port-forward and verify health:

```bash
kubectl port-forward -n rag service/rag-server 8081:8081 --address 0.0.0.0 & kubectl port-forward -n rag service/rag-frontend 3000:3000 --address 0.0.0.0 & sleep 3 && curl -s http://localhost:8081/v1/health?check_dependencies=true 2>/dev/null || echo "RAG_NOT_READY"
```

## Phase 5: Uninstall

If the user wants to tear down:

```bash
helm uninstall rag -n rag
kubectl delete nimcache --all -n rag
kubectl delete pvc --all -n rag
```

## On Success

Tell the user:
- "RAG Blueprint is running on Kubernetes. Access the UI at http://localhost:3000 (via port-forward)."
- "Ask me to configure features (VLM, query rewriting, guardrails, etc.)"
- "Ask me to shutdown when you're done."

## On Error

1. Check pod status and events: `kubectl describe pod <failing-pod> -n rag` and `kubectl get events -n rag --sort-by='.lastTimestamp' | tail -20`.
2. Read pod logs: `kubectl logs <failing-pod> -n rag --tail 50`.
3. Read `references/troubleshoot.md` to match against common issues (PVC pending, OOM, image pull failure, port conflict).
4. Apply the fix and retry. If the fix requires data deletion (PVCs, namespace), confirm with user first.

## Source Documentation
- `docs/support-matrix.md` — Kubernetes/Helm version requirements, GPU compatibility
- `docs/deploy-helm.md` — standard Helm deployment from NGC
- `docs/deploy-helm-from-repo.md` — Helm deployment from local repo
