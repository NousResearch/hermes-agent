"""
Index PubMed abstracts into Pinecone for scientific RAG.

Pipeline:
    1. Fetch abstracts from NCBI Entrez E-utilities given a list of PMIDs
    2. Embed with voyage-large-2 (input_type="document")
    3. Batch upsert to Pinecone with structured metadata

Usage:
    export PINECONE_API_KEY=...
    export VOYAGE_API_KEY=...
    python index_pubmed.py --index scientific-literature --namespace oncology \\
        --pmids 38291847,38215032,38109284

Environment:
    PINECONE_API_KEY  Pinecone API key
    VOYAGE_API_KEY    Voyage AI API key
"""

import argparse
import os
import sys
import time
from typing import Iterable
from xml.etree import ElementTree as ET

import requests
import voyageai
from pinecone import Pinecone, ServerlessSpec


NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EMBED_MODEL = "voyage-large-2"
EMBED_DIM = 1536


def fetch_pubmed_records(pmids: list[str]) -> list[dict]:
    """Fetch PubMed records via E-utilities. Returns list of dicts with title/abstract/metadata."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    response = requests.get(NCBI_EFETCH, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid_node = article.find(".//PMID")
        title_node = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//AbstractText")
        year_node = article.find(".//PubDate/Year")
        journal_node = article.find(".//Journal/Title")

        if pmid_node is None or title_node is None or not abstract_parts:
            continue

        # Concatenate multi-section abstracts
        abstract_text = " ".join(
            (part.text or "") for part in abstract_parts
        ).strip()
        if not abstract_text:
            continue

        records.append({
            "pmid": pmid_node.text,
            "title": (title_node.text or "").strip(),
            "abstract": abstract_text,
            "year": int(year_node.text) if year_node is not None and year_node.text else 0,
            "journal": (journal_node.text or "").strip() if journal_node is not None else "",
        })

    return records


def chunked(seq: list, size: int) -> Iterable[list]:
    """Yield successive chunks of `seq` of length `size`."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def embed_documents(client: voyageai.Client, texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Embed documents in batches, respecting Voyage AI batch limits."""
    embeddings: list[list[float]] = []
    for batch in chunked(texts, batch_size):
        result = client.embed(batch, model=EMBED_MODEL, input_type="document")
        embeddings.extend(result.embeddings)
    return embeddings


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    """Create the index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        info = pc.describe_index(index_name)
        if info.dimension != dimension:
            sys.exit(
                f"ERROR: index '{index_name}' exists with dimension {info.dimension}, "
                f"but this pipeline requires dimension {dimension}."
            )
        return

    print(f"Creating index '{index_name}' (dim={dimension}, metric=cosine, serverless)")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait for index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def index_pubmed(pmids: list[str], index_name: str, namespace: str) -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    ensure_index(pc, index_name, EMBED_DIM)
    index = pc.Index(index_name)

    print(f"Fetching {len(pmids)} PubMed records...")
    records = fetch_pubmed_records(pmids)
    print(f"  Got {len(records)} records with abstracts")

    if not records:
        print("No records to index. Exiting.")
        return

    print("Embedding abstracts with voyage-large-2...")
    abstracts = [r["abstract"] for r in records]
    embeddings = embed_documents(voyage, abstracts)
    print(f"  Embedded {len(embeddings)} abstracts")

    vectors = [
        {
            "id": f"pmid_{rec['pmid']}",
            "values": emb,
            "metadata": {
                "pmid": rec["pmid"],
                "title": rec["title"],
                "abstract": rec["abstract"][:500],  # truncate for metadata size
                "year": rec["year"],
                "journal": rec["journal"],
                "source": "PubMed",
            },
        }
        for rec, emb in zip(records, embeddings)
    ]

    print(f"Upserting to '{index_name}' namespace='{namespace}'...")
    total_batches = (len(vectors) + 99) // 100
    for i, batch in enumerate(chunked(vectors, 100), start=1):
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  Batch {i}/{total_batches} upserted ({len(batch)} vectors)")

    print(f"Done. Index stats: {index.describe_index_stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--namespace", default="pubmed", help="Pinecone namespace")
    parser.add_argument("--pmids", required=True, help="Comma-separated PMIDs")
    args = parser.parse_args()

    for env_var in ("PINECONE_API_KEY", "VOYAGE_API_KEY"):
        if env_var not in os.environ:
            sys.exit(f"ERROR: environment variable {env_var} is required.")

    pmids = [p.strip() for p in args.pmids.split(",") if p.strip()]
    index_pubmed(pmids, args.index, args.namespace)


if __name__ == "__main__":
    main()
