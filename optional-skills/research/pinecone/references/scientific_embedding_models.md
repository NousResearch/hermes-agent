# Scientific Embedding Models

A map from scientific domain → recommended embedding model → Pinecone index configuration. Use this as a starting point; benchmark on your actual retrieval task before committing to production.

## Quick Decision Table

| Your Data | Recommended Model | Dim | Metric | Notes |
|---|---|---|---|---|
| Biomedical literature (PubMed/PMC) | `voyage-large-2` | 1536 | cosine | Best general scientific recall |
| Clinical notes / EHR | `emilyalsentzer/Bio_ClinicalBERT` | 768 | cosine | Trained on MIMIC-III |
| Biomedical abstracts (lighter) | `pritamdeka/S-PubMedBert-MS-MARCO` | 768 | cosine | Sentence-transformer, fast |
| Protein sequences | `facebook/esm2_t33_650M_UR50D` | 1280 | cosine | Mean-pool token representations |
| Small molecules (SMILES) | `seyonec/PubChem10M_SMILES_BPE_450k` (ChemBERTa) | 768 | cosine | Trained on 10M PubChem SMILES |
| Small molecules (classical) | RDKit Morgan FP | 1024 or 2048 | cosine | Cosine here ≠ Tanimoto; rerank in RDKit |
| Multimodal (text + image) | `voyage-multimodal-3` | 1024 | cosine | Single embedding for text + PIL.Image |
| Radiology images alone | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 512 | cosine | Domain-tuned CLIP |
| Microscopy / cell images | Custom CNN features or DINOv2 | varies | cosine | Domain-fine-tune for best results |
| Genomic sequences (DNA) | `InstaDeepAI/nucleotide-transformer-500m-human-ref` | 1280 | cosine | Use sparingly — large model |

## Usage Patterns

### Voyage AI (recommended for production literature)

```python
import voyageai
voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# CRITICAL: use the right input_type
doc_embeddings = voyage.embed(
    texts=corpus_texts, model="voyage-large-2", input_type="document"
).embeddings

query_embedding = voyage.embed(
    texts=[user_question], model="voyage-large-2", input_type="query"
).embeddings[0]
```

Using `input_type="query"` vs `"document"` materially affects retrieval quality. Don't skip this.

### Bio_ClinicalBERT (clinical notes)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")
embeddings = model.encode(
    notes, batch_size=32, normalize_embeddings=True, show_progress_bar=True
).tolist()
```

`normalize_embeddings=True` is important when using cosine metric in Pinecone — it ensures consistent magnitudes.

### ESM-2 (protein sequences)

```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.training = False  # set to inference mode (PyTorch idiom)

def embed_protein(sequence: str) -> list[float]:
    """Mean-pool ESM-2 token representations into a single sequence embedding."""
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        results = model(tokens, repr_layers=[33])
    # Mean-pool over sequence length, excluding BOS/EOS tokens
    repr_ = results["representations"][33][0, 1:len(sequence) + 1]
    return repr_.mean(0).cpu().numpy().tolist()
```

For batched embedding of many sequences, use `model.forward()` with padded batches; see the ESM repo for the canonical pattern.

### Morgan Fingerprints (molecules, classical)

```python
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_morgan_list(smiles: str, radius: int = 2, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)  # 0/1 list, length n_bits
```

**Important:** Pinecone cosine similarity over binary Morgan fingerprints is **not** Tanimoto coefficient. Use Pinecone for fast top-N recall, then rerank with `rdkit.DataStructs.TanimotoSimilarity` for true Tanimoto ranking.

### voyage-multimodal-3 (text + image)

```python
import voyageai
from PIL import Image

voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

inputs = [
    [report_text_1, Image.open("radiograph_1.png")],
    [report_text_2, Image.open("radiograph_2.png")],
]

result = voyage.multimodal_embed(
    inputs=inputs,
    model="voyage-multimodal-3",
    input_type="document"
)
# result.embeddings is List[List[float]] of dim 1024
```

Inputs can be `[text]`, `[image]`, or `[text, image]` in any order. The model produces a single 1024-dim joint embedding per input.

## Dimension Pitfalls

A common production bug: switching embedding models without recreating the index. Pinecone will silently fail or return garbage if dimensions don't match exactly.

**Defensive pattern:**

```python
index_stats = pc.describe_index("my-index")
expected_dim = index_stats.dimension

emb = model.embed("test")
assert len(emb) == expected_dim, \
    f"Embedding dim {len(emb)} != index dim {expected_dim}"
```

## Cost Considerations

Higher dimensions = higher storage cost + slower queries. If your retrieval quality is acceptable with a 768-dim model, don't upgrade to 3072-dim for marginal gains. Benchmark on your task first.
