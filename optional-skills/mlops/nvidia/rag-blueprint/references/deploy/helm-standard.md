# Helm Deployment

## When to Use
- User wants to deploy RAG Blueprint on Kubernetes
- User asks about Helm chart installation (from NGC or local repo)
- User mentions Kubernetes, k8s, or Helm in deployment context

## Restrictions

Read `docs/support-matrix.md` for current Kubernetes, Helm, and OS version requirements.

- Requires GPU Operator + NIM Operator pre-installed
- Default StorageClass must be configured for PVC provisioning
- Disk space per `docs/support-matrix.md`
- NeMo Guardrails not available in Helm deployment
- Image captioning: on-prem only (requires `values.yaml` changes; see `docs/image_captioning.md`)

## Process

### Option A: Deploy from NGC (Remote Chart)
1. Read `docs/deploy-helm.md` for full commands and values
2. Ensure prerequisites: GPU Operator, NIM Operator, StorageClass, NGC_API_KEY
3. Install chart, monitor pods, port-forward frontend

### Option B: Deploy from Repository (Local Chart)
1. Read `docs/deploy-helm-from-repo.md` for full commands and repo setup
2. Add required Helm repos, run `helm dependency update`, install from local path

### RTX PRO 6000 Variant
1. Uncomment model section under `nimOperator.nim-llm.model` in `values.yaml`
2. See source docs for engine/precision/GPU settings

## Decision Table

| Goal | Option | Key Action |
|------|--------|------------|
| Quick deploy from published chart | NGC (Option A) | `helm upgrade --install` with NGC URL |
| Customized chart | Local repo (Option B) | Clone, modify values, `helm dependency update` |
| RTX PRO 6000 GPUs | Either option | Uncomment model section in values.yaml |
| Retrieval-only (no LLM) | Either option | `--set nimOperator.nim-llm.enabled=false` |

## Agent-Specific Notes
- First deployment: 60–70 min (model cache download); subsequent: 10–15 min
- Pods in `ContainerCreating`/`Init` for extended time is normal during cache download
- PVCs are not removed by `helm uninstall` — delete manually: `kubectl delete nimcache --all -n rag && kubectl delete pvc --all -n rag`
- Port-forwarding may timeout for large file ingestion — not suitable for bulk uploads
- All configurable endpoints documented in `deploy/helm/nvidia-blueprint-rag/endpoints.md`

## Source Documentation
- `docs/deploy-helm.md` — NGC remote chart deployment, prerequisites, monitoring
- `docs/deploy-helm-from-repo.md` — local chart deployment, repo setup, dependency management
