#!/usr/bin/env python3
"""Post-migration index repair for Zvec memory collection.

Usage:
    python3 post-migration-repair.py [collection_path]

If no path is given, uses default HERMES_HOME/记忆数据库/zvec_memory/memories.

Requires: zvec installed in the Python environment running this script.
"""
import sys
import os

def main():
    # Resolve collection path
    if len(sys.argv) > 1:
        collection_path = sys.argv[1]
    else:
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        collection_path = os.path.join(hermes_home, "记忆数据库", "zvec_memory", "memories")

    if not os.path.isdir(collection_path):
        print(f"ERROR: Collection directory not found: {collection_path}")
        sys.exit(1)

    import zvec

    print(f"Opening collection: {collection_path}")
    try:
        coll = zvec.open(collection_path)
    except Exception as e:
        print(f"ERROR: Cannot open collection (is a gateway holding the lock?): {e}")
        sys.exit(1)

    doc_count = coll.stats.doc_count
    print(f"Total docs: {doc_count}")

    # Step 1: HNSW optimize
    print("\n[1/3] HNSW optimize...")
    coll.optimize()
    completeness = coll.stats.index_completeness
    print(f"  index_completeness: {completeness}")
    ok = "✅" if completeness.get("vector", 0) >= 1.0 else "❌"
    print(f"  {ok} {'HNSW complete' if completeness.get('vector', 0) >= 1.0 else 'HNSW still incomplete'}")

    # Step 2: FTS rebuild with jieba
    print("\n[2/3] FTS rebuild (jieba tokenizer)...")
    try:
        coll.drop_index("content")
        print("  Dropped old FTS index")
    except Exception:
        print("  No existing FTS index to drop")

    coll.create_index("content", zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"]))
    print("  Created new FTS index with jieba")

    # Verify with English
    en_hits = coll.query(zvec.Query(field_name="content", fts=zvec.Fts(match_string="user")), topk=1)
    print(f"  FTS 'user' hits: {len(en_hits)} {'✅' if len(en_hits) > 0 else '❌'}")

    # Verify with Chinese
    zh_hits = coll.query(zvec.Query(field_name="content", fts=zvec.Fts(match_string="芯片")), topk=1)
    print(f"  FTS '芯片' hits: {len(zh_hits)} {'✅' if len(zh_hits) > 0 else '⚠️ (no Chinese content?)'}")

    # Step 3: Scalar InvertIndex rebuild
    print("\n[3/3] Scalar InvertIndex rebuild...")
    for field_name in ["role", "session_id", "created_at"]:
        try:
            coll.drop_index(field_name)
        except Exception:
            pass
        coll.create_index(field_name, zvec.InvertIndexParam(enable_range_optimization=True))
        print(f"  ✅ {field_name}")

    print(f"\n{'='*50}")
    print(f"Repair complete. Total docs: {doc_count}")
    print(f"index_completeness: {coll.stats.index_completeness}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
