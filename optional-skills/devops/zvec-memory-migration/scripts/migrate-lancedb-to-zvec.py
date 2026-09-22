#!/usr/bin/env python3
"""LanceDB → Zvec memory migration script.

Usage:
    HERMES_PROFILE=<name> python3 migrate-lancedb-to-zvec.py

Reads all rows from the LanceDB memories collection, writes them to a new
Zvec collection with identical schema (HNSW + FTS jieba + InvertIndex),
then optimises all indexes.

Prerequisites:
    - hermes-agent venv activated (or sys.path adjusted)
    - Zvec Python package installed
    - Ollama running with bge-m3:latest (not needed for migration itself, only for post-migration verification)

Environment:
    HERMES_PROFILE  – target profile name (default: default)
"""

import ast
import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-detect hermes-agent venv
# ---------------------------------------------------------------------------
VENV_SITE = Path.home() / ".hermes/hermes-agent/venv/lib/python3.11/site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import lance  # noqa: E402
import zvec   # noqa: E402

PROFILE = os.environ.get("HERMES_PROFILE", "default")
if PROFILE == "default":
    HERMES_HOME = Path.home() / ".hermes"
else:
    HERMES_HOME = Path.home() / ".hermes/profiles" / PROFILE

LANCEDB_PATH = str(HERMES_HOME / "记忆数据库/lance_memory/memories.lance")
ZVEC_PARENT = str(HERMES_HOME / "记忆数据库/zvec_memory")
ZVEC_PATH = ZVEC_PARENT + "/memories"
VECTOR_DIM = 1024


def build_schema() -> zvec.CollectionSchema:
    """Build Zvec CollectionSchema matching memory-zvec plugin conventions."""
    return zvec.CollectionSchema(
        name="memories",
        fields=[
            zvec.FieldSchema(
                "content", zvec.DataType.STRING,
                index_param=zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"]),
            ),
            zvec.FieldSchema(
                "role", zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),
            zvec.FieldSchema(
                "session_id", zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),
            zvec.FieldSchema(
                "created_at", zvec.DataType.DOUBLE,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema("metadata", zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, VECTOR_DIM,
                index_param=zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.COSINE,
                    m=16,
                    ef_construction=100,
                ),
            ),
        ],
    )


def migrate() -> None:
    # --- 1. Read LanceDB ----------------------------------------------------
    print(f"[1/4] Reading LanceDB from {LANCEDB_PATH}")
    ds = lance.dataset(LANCEDB_PATH)
    df = ds.to_table().to_pandas()
    total = len(df)
    print(f"      Found {total} rows, columns: {list(df.columns)}")
    if total == 0:
        print("      Nothing to migrate, exiting.")
        return

    # --- 2. Prepare Zvec directory -----------------------------------------
    if os.path.exists(ZVEC_PATH):
        print(f"[2/4] Removing existing Zvec data at {ZVEC_PATH}")
        shutil.rmtree(ZVEC_PATH)
    os.makedirs(ZVEC_PARENT, exist_ok=True)
    assert not os.path.exists(ZVEC_PATH), f"Target path must not exist: {ZVEC_PATH}"

    # --- 3. Create collection & batch insert --------------------------------
    schema = build_schema()
    coll = zvec.create_and_open(ZVEC_PATH, schema)
    print(f"[3/4] Collection created, inserting {total} rows …")

    batch_size = 50
    inserted = 0
    start = time.time()

    for i in range(0, total, batch_size):
        batch_df = df.iloc[i : i + batch_size]
        docs = []
        for _, row in batch_df.iterrows():
            vec = row["vector"]
            if isinstance(vec, str):
                vec = ast.literal_eval(vec)
            docs.append(
                zvec.Doc(
                    id=str(row["id"]),
                    vectors={"vector": vec},
                    fields={
                        "content": str(row["content"]),
                        "role": str(row["role"]),
                        "session_id": str(row["session_id"]),
                        "created_at": float(row["created_at"]),
                        "metadata": str(row.get("metadata", "{}")),
                    },
                )
            )
        coll.insert(docs)
        inserted += len(docs)
        elapsed = time.time() - start
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"      {inserted}/{total} ({rate:.0f} docs/s)")

    coll.flush()
    elapsed = time.time() - start
    print(f"      Insert complete: {inserted} rows in {elapsed:.1f}s ({inserted/elapsed:.0f} docs/s)")

    # --- 4. Optimise indexes -----------------------------------------------
    print("[4/4] Optimising indexes …")
    coll.optimize()
    stats = coll.stats
    print(f"      doc_count: {stats.doc_count}")
    print(f"      index_completeness: {stats.index_completeness}")

    print(f"\n✅ Migration complete for profile '{PROFILE}' ({total} rows).")
    print(f"   Zvec data: {ZVEC_PATH}")
    print(f"   LanceDB preserved at: {LANCEDB_PATH}")


if __name__ == "__main__":
    migrate()
