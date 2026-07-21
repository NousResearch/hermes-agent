# MIG GPU Deployment

## When to Use
- User wants fine-grained GPU allocation on Kubernetes using MIG slices
- User has H100 GPUs and wants to share them across RAG services
- User asks about Multi-Instance GPU deployment

## Restrictions
- Requires H100 80GB HBM3 GPUs (MIG-compatible)
- MIG profiles in this guide are specific to H100 80GB — other GPUs need different profiles
- Requires cloned repository (MIG config files in `deploy/helm/`)
- All standard Helm prerequisites apply (GPU Operator, NIM Operator, StorageClass)
- Ingestion profile is scaled down with MIG — large bulk ingestion jobs may fail

## Process
1. Read `docs/mig-deployment.md` for full configuration, commands, and MIG slice definitions
2. Enable MIG with mixed strategy on ClusterPolicy
3. Apply MIG ConfigMap and label the node
4. Verify node labels show `mig.config.state: "success"` before proceeding
5. Install Helm chart with `-f mig-slicing/values-mig.yaml`

## Decision Table

| Goal | Source Doc | Key Action |
|------|-----------|------------|
| Standard MIG on H100 | `docs/mig-deployment.md` | Apply MIG config, label node, install chart |
| RTX PRO 6000 with MIG | `docs/mig-deployment.md` | Also uncomment model section in values.yaml |
| Custom MIG profiles | NVIDIA MIG User Guide | Modify `mig-config.yaml` for different GPU types |

## Agent-Specific Notes
- Must wait for `mig.config.state: "success"` on the node before Helm install — if not present, wait and re-check
- Default H100 MIG layout (see `docs/mig-deployment.md` for current GPU count and slice definitions): GPU 0 → small slices, GPU 1 → mixed slices, GPU 2 → full-GPU slice
- LLM gets the largest slice (`7g.80gb`); embedding/Milvus/ingest share small slices
- RTX PRO 6000 variant: uncomment model section in values.yaml, then use both `-f values.yaml -f mig-slicing/values-mig.yaml`
- Uninstall follows standard Helm procedure (see Helm deployment docs)

## Source Documentation
- `docs/mig-deployment.md` — full MIG config, ClusterPolicy patches, node labeling, verification, Helm install commands
