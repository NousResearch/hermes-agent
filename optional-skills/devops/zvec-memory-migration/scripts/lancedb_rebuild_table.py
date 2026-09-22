#!/usr/bin/env python3
"""
lancedb_rebuild_table.py — 扫描所有 profile，识别"Table 'memories' not found"故障，
自动用 lance.file.LanceFileReader 读出 data/*.lance，重建 memories 表 + FTS 索引。

Usage:
    ~/.hermes/venv/bin/python lancedb_rebuild_table.py [--dry-run] [--profile <name>]

行为:
    --dry-run:    只扫描报告，不修改
    --profile X:  只处理指定 profile (默认所有)
    不带参数:     扫描 + 报告 + 询问确认后修复
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import lancedb
import lance.file as lf
import pyarrow as pa


PROFILES = ["default"]  # add your profile names here, or run per profile
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("role", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 1024)),
    pa.field("created_at", pa.float64()),
    pa.field("metadata", pa.string()),
])


def profile_home(name):
    if name == "default":
        return Path.home() / ".hermes"
    return Path.home() / ".hermes/profiles" / name


def diagnose(profile):
    """返回 (ok: bool, lance_dir, row_count_or_None, error_msg)"""
    home = profile_home(profile)
    lance_dir = home / "lance_memory"
    if not lance_dir.exists():
        return False, lance_dir, None, "no lance_memory dir"
    try:
        db = lancedb.connect(str(lance_dir))
        if "memories" not in db.table_names():
            return False, lance_dir, None, "no memories table"
        tbl = db.open_table("memories")
        return True, lance_dir, tbl.count_rows(), "ok"
    except Exception as e:
        return False, lance_dir, None, f"{type(e).__name__}: {str(e)[:80]}"


def read_lance_files(data_dir):
    """读 data/ 下所有 .lance 文件，返回合并的 pyarrow Table"""
    if not data_dir.exists():
        return None
    all_tables = []
    errors = []
    for f in sorted(data_dir.iterdir()):
        if not f.name.endswith(".lance"):
            continue
        try:
            tbl = lf.LanceFileReader(str(f)).read_all().to_table()
            if tbl.num_rows > 0:
                all_tables.append(tbl)
        except Exception as e:
            errors.append((f.name, str(e)[:60]))
    if errors:
        print(f"  ⚠️ {len(errors)} files unreadable: {errors[:3]}")
    if not all_tables:
        return None
    return pa.concat_tables(all_tables)


def rebuild(profile):
    """重建一个 profile 的 memories 表。返回重建后的 row count，失败返回 None"""
    home = profile_home(profile)
    lance_dir = home / "lance_memory"
    memories_dir = lance_dir / "memories.lance"
    data_dir = memories_dir / "data"

    if not data_dir.exists():
        print(f"  ❌ no data dir")
        return None

    # 1. 读出
    print(f"  步骤 1: 读 {data_dir}/*.lance")
    t0 = time.time()
    combined = read_lance_files(data_dir)
    if combined is None or combined.num_rows == 0:
        print(f"  ❌ no data to recover")
        return None
    print(f"    → {combined.num_rows} rows, {time.time()-t0:.2f}s")

    # 2. 备份
    backup = home / "lance_memory_PRE_REBUILD"
    if backup.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = home / f"lance_memory_PRE_REBUILD_{ts}"
    shutil.copytree(memories_dir, backup)
    print(f"  步骤 2: 备份到 {backup}")

    # 3. 删旧表
    shutil.rmtree(memories_dir)
    print(f"  步骤 3: 删旧 memories.lance/")

    # 4. 重建
    db = lancedb.connect(str(lance_dir))
    rows = combined.to_pylist()
    tbl = db.create_table("memories", data=rows, schema=SCHEMA, mode="overwrite")
    print(f"  步骤 4: create_table 写入 {tbl.count_rows()} rows")

    # 5. FTS 索引
    t0 = time.time()
    try:
        tbl.create_fts_index("content", replace=True)
        print(f"  步骤 5: FTS 索引 ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"  ⚠️ FTS 失败: {e}")

    # 6. 验证
    db2 = lancedb.connect(str(lance_dir))
    tbl2 = db2.open_table("memories")
    final = tbl2.count_rows()
    idx = tbl2.list_indices()
    print(f"  验证: {final} rows, indices={idx}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只扫描报告")
    ap.add_argument("--profile", help="只处理指定 profile")
    args = ap.parse_args()

    profiles = [args.profile] if args.profile else PROFILES

    print("=" * 60)
    print("LanceDB 表健康扫描")
    print("=" * 60)

    bad = []
    for prof in profiles:
        ok, lance_dir, rows, msg = diagnose(prof)
        status = "✅ OK" if ok else "❌ FAULT"
        rows_str = f"{rows} rows" if rows else "—"
        print(f"  [{prof:20}] {status}  {rows_str:12}  {msg}")
        if not ok:
            bad.append((prof, lance_dir, msg))

    if not bad:
        print("\n所有 profile 健康。")
        return 0

    print(f"\n发现 {len(bad)} 个故障 profile")
    if args.dry_run:
        print("(dry-run 模式，跳过修复)")
        return 1

    # 确认
    print()
    for prof, lance_dir, msg in bad:
        print(f"  • {prof}: {msg}")
    ans = input(f"\n确认要修复这 {len(bad)} 个 profile？(yes/no): ")
    if ans.lower() != "yes":
        print("取消")
        return 2

    # 修复
    print()
    for prof, lance_dir, msg in bad:
        print(f"\n========== 修复 {prof} ==========")
        try:
            rows = rebuild(prof)
            if rows:
                print(f"✅ {prof}: 修复完成, {rows} rows")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ {prof}: 修复失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
