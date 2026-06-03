"""
Multimodal radiology indexing with voyage-multimodal-3 and Pinecone.

Pipeline:
    1. Load radiology report + image pairs from a manifest CSV
    2. Embed each (text, image) pair into a single 1024-dim vector
    3. Upsert to Pinecone with study metadata

Manifest CSV format (with header):
    case_id,report_path,image_path,modality,body_part,finding,study_uid

Example:
    case_001,reports/001.txt,images/001.png,CT,chest,pneumonia,1.2.3.4.5

Usage:
    export PINECONE_API_KEY=...
    export VOYAGE_API_KEY=...
    python multimodal_radiology.py --manifest cases.csv \\
        --index radiology-cases --namespace ct-chest

Environment:
    PINECONE_API_KEY  Pinecone API key
    VOYAGE_API_KEY    Voyage AI API key
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import voyageai
from PIL import Image
from pinecone import Pinecone, ServerlessSpec


EMBED_MODEL = "voyage-multimodal-3"
EMBED_DIM = 1024
BATCH_SIZE = 8  # voyage-multimodal-3 is heavier; smaller batches


def load_manifest(manifest_path: Path) -> list[dict]:
    """Read the case manifest CSV."""
    cases = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


def load_case(case: dict) -> tuple[str, Image.Image] | None:
    """Load report text and image for a case. Returns None if files are missing."""
    report_path = Path(case["report_path"])
    image_path = Path(case["image_path"])

    if not report_path.exists() or not image_path.exists():
        print(f"  Skipping {case['case_id']}: missing file(s)")
        return None

    report_text = report_path.read_text(encoding="utf-8").strip()
    image = Image.open(image_path).convert("RGB")
    return report_text, image


def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def embed_multimodal_pairs(
    client: voyageai.Client, pairs: list[tuple[str, Image.Image]]
) -> list[list[float]]:
    """Embed (text, image) pairs in batches."""
    embeddings: list[list[float]] = []
    for batch in chunked(pairs, BATCH_SIZE):
        inputs = [[text, img] for text, img in batch]
        result = client.multimodal_embed(
            inputs=inputs, model=EMBED_MODEL, input_type="document"
        )
        embeddings.extend(result.embeddings)
    return embeddings


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        info = pc.describe_index(index_name)
        if info.dimension != dimension:
            sys.exit(
                f"ERROR: index '{index_name}' has dimension {info.dimension}, "
                f"expected {dimension}."
            )
        return

    print(f"Creating index '{index_name}' (dim={dimension}, metric=cosine, serverless)")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def index_radiology(manifest_path: Path, index_name: str, namespace: str) -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    ensure_index(pc, index_name, EMBED_DIM)
    index = pc.Index(index_name)

    cases = load_manifest(manifest_path)
    print(f"Loaded manifest with {len(cases)} cases")

    valid_cases = []
    pairs = []
    for case in cases:
        loaded = load_case(case)
        if loaded is None:
            continue
        valid_cases.append(case)
        pairs.append(loaded)

    if not pairs:
        print("No valid cases to index. Exiting.")
        return

    print(f"Embedding {len(pairs)} (report, image) pairs...")
    embeddings = embed_multimodal_pairs(voyage, pairs)

    vectors = [
        {
            "id": case["case_id"],
            "values": emb,
            "metadata": {
                "modality":  case.get("modality", ""),
                "body_part": case.get("body_part", ""),
                "finding":   case.get("finding", ""),
                "study_uid": case.get("study_uid", ""),
            },
        }
        for case, emb in zip(valid_cases, embeddings)
    ]

    print(f"Upserting to '{index_name}' namespace='{namespace}'...")
    total_batches = (len(vectors) + 99) // 100
    for i, batch in enumerate(chunked(vectors, 100), start=1):
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  Batch {i}/{total_batches} upserted")

    print(f"Done. Index stats: {index.describe_index_stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest CSV")
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--namespace", default="radiology", help="Pinecone namespace")
    args = parser.parse_args()

    for env_var in ("PINECONE_API_KEY", "VOYAGE_API_KEY"):
        if env_var not in os.environ:
            sys.exit(f"ERROR: environment variable {env_var} is required.")

    if not args.manifest.exists():
        sys.exit(f"ERROR: manifest file not found: {args.manifest}")

    index_radiology(args.manifest, args.index, args.namespace)


if __name__ == "__main__":
    main()
