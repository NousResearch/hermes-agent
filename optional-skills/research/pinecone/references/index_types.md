# Index Types: Serverless vs Pod-Based

Pinecone offers two index architectures. Choose based on workload characteristics.

## Serverless Indexes (Default Recommendation)

```python
from pinecone import Pinecone, ServerlessSpec

pc.create_index(
    name="my-index",
    dimension=1024,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

**Use when:**
- Workload is bursty or unpredictable
- Cost-sensitive (pay per operation + storage, not provisioned capacity)
- Building prototypes or research projects
- Total vector count < 100M

**Available regions** (verify at https://docs.pinecone.io for current list):
- AWS: `us-east-1`, `us-west-2`, `eu-west-1`
- GCP: `us-central1`, `europe-west4`
- Azure: `eastus2`

## Pod-Based Indexes

```python
from pinecone import Pinecone, PodSpec

pc.create_index(
    name="production-index",
    dimension=768,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",   # Pod tier and size
        pods=1,             # Number of pods
        replicas=1          # Replicas per pod for HA
    )
)
```

**Use when:**
- Predictable, sustained high QPS (>100 queries/sec)
- Need dedicated compute for latency SLAs
- Very large indexes (>100M vectors) where serverless cost exceeds pod cost
- Specific pod features (selective metadata indexing, collections/backups)

**Pod tiers:**

| Tier | Best For | Vectors/Pod (~1024d) |
|---|---|---|
| `s1` | Storage-optimized, lower QPS | ~5M |
| `p1` | Balanced performance | ~1M |
| `p2` | Highest QPS, lowest latency | ~1M |

Each tier has sizes `x1`, `x2`, `x4`, `x8` — doubling capacity and cost.

## Dimension Selection by Embedding Model

Match index dimension exactly to your embedding model's output. **You cannot change dimension after creation — only by recreating the index.**

| Embedding Model | Dimension | Domain |
|---|---|---|
| voyage-multimodal-3 | 1024 | Multimodal (text + image) |
| voyage-large-2 | 1536 | General-purpose text |
| voyage-2 | 1024 | General-purpose text (lighter) |
| text-embedding-3-large (OpenAI) | 3072 | General-purpose text |
| text-embedding-3-small (OpenAI) | 1536 | General-purpose text (lighter) |
| all-mpnet-base-v2 | 768 | General-purpose (open source) |
| Bio_ClinicalBERT | 768 | Clinical notes |
| PubMedBERT | 768 | Biomedical literature |
| ESM-2 (650M) | 1280 | Protein sequences |
| ESM-2 (3B) | 2560 | Protein sequences (higher capacity) |
| MolBERT / ChemBERTa | 768 | Molecules (SMILES) |
| Morgan Fingerprints | 1024 or 2048 | Molecules (classical) |

**Pro tip:** If unsure, embed one sample and check: `len(model.embed("test"))`.

## Metric Selection

| Metric | When to Use |
|---|---|
| `cosine` | Default for most embedding models. Vectors typically L2-normalized; measures angle. |
| `dotproduct` | **Required** for hybrid (dense + sparse) search. Also used when magnitude carries meaning. |
| `euclidean` | Geometric features, coordinates, raw pixel embeddings. Rarely used for NLP. |

## Migration Path

1. Start with serverless `cosine` for prototyping
2. If hybrid search needed → recreate with `dotproduct` (cosine ≈ dotproduct on normalized vectors anyway)
3. If sustained QPS or specialized features needed → migrate to pod-based with the same dimension and metric
