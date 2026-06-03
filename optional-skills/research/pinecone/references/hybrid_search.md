# Hybrid Search: Dense + Sparse BM25

Hybrid search combines semantic similarity (dense embeddings) with keyword relevance (sparse BM25). It's especially valuable in scientific retrieval where exact terminology matters — gene names, compound IDs, assay types, drug names — but you also need semantic understanding.

## When to Use Hybrid

**Use hybrid when:**
- Your queries contain exact technical terms (e.g., `BRCA1`, `EGFR-T790M`, `IL-6`, `tofacitinib`)
- Pure dense search misses keyword matches due to embedding ambiguity
- Users mix natural language and identifiers (`"BRCA1 variants in triple-negative breast cancer"`)

**Stick with dense-only when:**
- Queries are purely conceptual (`"mechanisms of immune evasion"`)
- Your corpus has no critical technical identifiers
- Latency budget is tight (hybrid is slightly slower)

## Setup

### 1. Create a dotproduct index

Hybrid search **requires** `metric="dotproduct"`:

```python
from pinecone import Pinecone, ServerlessSpec

pc.create_index(
    name="hybrid-literature",
    dimension=1024,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### 2. Fit a BM25 encoder on your corpus

```python
from pinecone_text.sparse import BM25Encoder

bm25 = BM25Encoder()
bm25.fit(corpus_texts)  # corpus_texts: List[str]
bm25.dump("bm25_model.json")

# Later, reload
bm25 = BM25Encoder().load("bm25_model.json")
```

**Important:** Fit BM25 on a representative sample of your corpus (10k-100k docs is usually enough). Re-fit if your corpus characteristics change substantially.

### 3. Upsert dense + sparse vectors together

```python
from sentence_transformers import SentenceTransformer

dense_model = SentenceTransformer("all-mpnet-base-v2")  # 768-dim

vectors = []
for doc in documents:
    dense_vec  = dense_model.encode(doc["text"], normalize_embeddings=True).tolist()
    sparse_vec = bm25.encode_documents(doc["text"])
    # sparse_vec is dict: {"indices": [...], "values": [...]}

    vectors.append({
        "id":            doc["id"],
        "values":        dense_vec,
        "sparse_values": sparse_vec,
        "metadata":      doc["metadata"]
    })

# Batch upsert
for i in range(0, len(vectors), 100):
    index.upsert(vectors=vectors[i:i + 100])
```

## Querying with Alpha Blending

The `alpha` parameter controls the dense/sparse balance:

- `alpha = 1.0` → pure dense (semantic only)
- `alpha = 0.0` → pure sparse (keyword only)
- `alpha = 0.5` → equal weight

```python
def hybrid_query(query_text: str, alpha: float = 0.5, top_k: int = 10,
                 namespace: str = "", filter: dict = None):
    dense_vec  = dense_model.encode(query_text, normalize_embeddings=True).tolist()
    sparse_vec = bm25.encode_queries(query_text)

    # Scale dense by alpha, sparse by (1 - alpha)
    dense_scaled  = [v * alpha for v in dense_vec]
    sparse_scaled = {
        "indices": sparse_vec["indices"],
        "values":  [v * (1.0 - alpha) for v in sparse_vec["values"]]
    }

    return index.query(
        vector=dense_scaled,
        sparse_vector=sparse_scaled,
        top_k=top_k,
        namespace=namespace,
        filter=filter,
        include_metadata=True
    )
```

**Note** that `bm25.encode_documents()` and `bm25.encode_queries()` use slightly different normalizations — always use `encode_queries()` for the query side.

## Tuning Alpha

Start at `alpha = 0.5` and adjust based on observed retrieval quality:

| If retrieval is missing... | Try |
|---|---|
| Exact gene names, IDs, drug names | Lower alpha (more sparse weight): 0.3 |
| Conceptually related content | Higher alpha (more dense weight): 0.7 |

**Empirical:** for biomedical literature retrieval, `alpha = 0.4` often outperforms pure dense or pure sparse on benchmarks like BEIR/SciFact.

## Limitations

- BM25 is corpus-specific. A BM25 model fit on cancer literature won't transfer well to materials science.
- Sparse vectors increase storage costs marginally (~10-20%).
- Hybrid query latency is slightly higher (typically 10-30ms more than dense-only).
- You cannot retroactively add sparse vectors to an existing dense-only index — you must recreate with `metric="dotproduct"` and re-upsert.

## Alternative: SPLADE

For higher-quality sparse retrieval than BM25, consider SPLADE (Sparse Lexical and Expansion model). Pinecone supports SPLADE vectors with the same `sparse_values` API — see `pinecone-text` documentation for the encoder.

SPLADE typically improves recall by 5-15% over BM25 on scientific benchmarks but requires GPU inference for encoding.
