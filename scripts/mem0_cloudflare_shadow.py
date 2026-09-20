#!/usr/bin/env python3
"""Build a Cloudflare-embedded Qdrant shadow collection from an existing one."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests
from qdrant_client import QdrantClient, models


def client_from(config: dict) -> QdrantClient:
    if config.get("url"):
        return QdrantClient(url=config["url"], api_key=config.get("api_key"), https=config.get("https"))
    if config.get("path"):
        return QdrantClient(path=os.path.expanduser(config["path"]))
    return QdrantClient(
        host=config.get("host", "127.0.0.1"),
        port=int(config.get("port", 6333)),
        api_key=config.get("api_key"),
        https=config.get("https", False),
    )


def _embed_batch(endpoint: str, headers: dict[str, str], texts: list[str]) -> list[list[float]]:
    """Embed a batch, bisecting on provider size/token errors."""
    response = requests.post(
        endpoint,
        headers=headers,
        json={"model": "@cf/baai/bge-m3", "input": texts},
        timeout=120,
    )
    if response.status_code in {400, 413} and len(texts) > 1:
        middle = len(texts) // 2
        return _embed_batch(endpoint, headers, texts[:middle]) + _embed_batch(endpoint, headers, texts[middle:])
    response.raise_for_status()
    data = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
    return [row["embedding"] for row in data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--batch", type=int, default=200)
    args = parser.parse_args()

    root = json.loads(Path(args.config).read_text(encoding="utf-8"))
    vector_config = root["oss"]["vector_store"]["config"]
    source = vector_config["collection_name"]
    target = args.target
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_WORKERS_AI_TOKEN"]
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {token}"}

    client = client_from(vector_config)
    if client.collection_exists(target):
        client.delete_collection(target)
    client.create_collection(
        collection_name=target,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )
    source_info = client.get_collection(source)
    payload_schema = getattr(source_info, "payload_schema", {}) or {}
    for field_name, schema in payload_schema.items():
        field_type = getattr(schema, "data_type", schema)
        try:
            client.create_payload_index(target, field_name=field_name, field_schema=field_type)
        except Exception:
            pass

    total = client.count(source, exact=True).count
    offset = None
    done = 0
    started = time.time()
    while True:
        points, offset = client.scroll(
            source,
            limit=args.batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        texts = [str((point.payload or {}).get("data") or (point.payload or {}).get("memory") or "") for point in points]
        # Workers AI limits request body size; keep pathological long memories bounded
        # to the same order as Hermes' memory sync input while preserving their lead.
        texts = [text[:16_000] for text in texts]
        vectors = _embed_batch(endpoint, headers, texts)
        if len(vectors) != len(points):
            raise RuntimeError(f"Cloudflare returned {len(vectors)} vectors for {len(points)} texts")
        client.upsert(
            target,
            points=[
                models.PointStruct(id=point.id, vector=vector, payload=point.payload)
                for point, vector in zip(points, vectors)
            ],
            wait=True,
        )
        done += len(points)
        print(json.dumps({"done": done, "total": total, "elapsed": round(time.time() - started, 1)}), flush=True)
        if offset is None:
            break
    final = client.count(target, exact=True).count
    if final != total:
        raise RuntimeError(f"shadow count mismatch: {final} != {total}")
    print(json.dumps({"status": "ok", "source": source, "target": target, "count": final, "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
