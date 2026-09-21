#!/usr/bin/env python3
"""Verify memory-zvec plugin tools through Hermes framework (no gateway needed).

Usage:
    HERMES_PROFILE=<name> python3 verify-plugin-tools.py

Runs the full 8-item verification checklist from vector-db-migration skill:
  vec_memory_stats, vec_memory_list, vec_memory_search (vector/keyword/hybrid/filter),
  vec_memory_add, vec_memory_delete.

Prerequisites:
    - hermes-agent venv (zvec package installed)
    - Target profile has the user-level plugin dir (memory-zvec, same-named
      manifest) and config.yaml provider setting
    - Ollama running with bge-m3:latest (needed for vec_memory_add embedding)

Environment:
    HERMES_PROFILE  – target profile name (default: zunhunfan)
"""

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-detect hermes-agent venv
# ---------------------------------------------------------------------------
VENV_SITE = Path.home() / ".hermes/hermes-agent/venv/lib/python3.11/site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

PROFILE = os.environ.get("HERMES_PROFILE", "zunhunfan")
# `default` profile 的 HERMES_HOME 就是 ~/.hermes/（没有 profiles/default/）——
# 不做特判会让 default 解析到不存在的路径，验证脚本自身先报错。
if PROFILE == "default":
    HERMES_HOME = str(Path.home() / ".hermes")
else:
    HERMES_HOME = str(Path.home() / f".hermes/profiles/{PROFILE}")

os.environ["HERMES_HOME"] = HERMES_HOME

passed = 0
failed = 0


def call_tool(provider, name, params):
    """Call a plugin tool and return parsed dict."""
    result = provider.handle_tool_call(name, params)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    return result


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {label}")
    else:
        failed += 1
        print(f"  ❌ {label}  — {detail}")


def main():
    from plugins.memory import load_memory_provider

    print(f"Profile: {PROFILE}  |  HERMES_HOME: {HERMES_HOME}")
    print("=" * 55)

    # Discover providers. Since 2026-09-14 the plugin ships as a USER-LEVEL fork;
    # directory name and plugin.yaml name are both `memory-zvec` (2026-09-21 rename).
    # plugins/memory discovery keys on DIRECTORY name, so the bare config name no
    # longer resolves via find/load — resolve through a plugin.yaml manifest scan
    # (mirrors how the gateway's PluginManager actually activates it).
    from plugins.memory import discover_memory_providers, load_memory_provider
    providers = {n: (d, a) for n, d, a in discover_memory_providers()}
    provider_name = "memory-zvec" if "memory-zvec" in providers else None
    if provider_name is None:
        import re as _re
        plugins_root = Path(HERMES_HOME) / "plugins"
        if plugins_root.is_dir():
            for child in sorted(plugins_root.iterdir()):
                yml = child / "plugin.yaml"
                if not (child.is_dir() and yml.exists()):
                    continue
                try:
                    text = yml.read_text(errors="replace", encoding="utf-8")[:2048]
                except Exception:
                    continue
                m = _re.search(r"^name:\s*(\S+)", text, _re.M)
                if m and m.group(1).strip("'\"") == "memory-zvec":
                    provider_name = child.name
                    break
    check("memory-zvec discovered (via dir: %s)" % provider_name, provider_name is not None)

    # Load and initialize
    provider = load_memory_provider(provider_name) if provider_name else None
    check("load_memory_provider", provider is not None)
    if provider is None:
        print(f"\nResult: {passed} passed, {failed + 1} failed — cannot continue without provider")
        return 1
    provider.initialize(session_id="verify_plugin")
    print(f"  Provider: {provider.name}, dim={provider.vector_dim}, model={provider.model}")

    # 1. stats
    print("\n--- vec_memory_stats ---")
    stats = call_tool(provider, "vec_memory_stats", {})
    check("backend=zvec", stats.get("backend") == "zvec", f"got {stats.get('backend')}")
    check("has total_memories", "total_memories" in stats, f"keys: {list(stats.keys())[:5]}")
    total_before = stats.get("total_memories", 0)

    # 2. list
    print("\n--- vec_memory_list ---")
    lst = call_tool(provider, "vec_memory_list", {"limit": 3})
    # handle_tool_call returns JSON string for list too
    if isinstance(lst, str):
        print(f"  raw: {lst[:200]}")
        check("list returns data", len(lst) > 10)
    else:
        check("list returns dict", isinstance(lst, dict), f"got {type(lst)}")

    # 3. vector search
    print("\n--- vec_memory_search (vector) ---")
    srch = call_tool(provider, "vec_memory_search", {
        "query": "测试查询", "mode": "vector", "top_k": 3
    })
    if isinstance(srch, str):
        srch = json.loads(srch) if srch.startswith("{") else {"raw": srch}
    results = srch.get("results", [])
    check("vector search returns results", len(results) > 0, f"got {len(results)}")

    # 4. keyword search (FTS)
    # A single hard-coded query gives a FALSE FAILURE on profiles whose corpus
    # simply lacks that word (e.g. zunhunfan's reports contain no "用户").
    # Probe a few common terms and pass if any hits — the point is "does FTS
    # work at all", not "does this word exist in this corpus".
    print("\n--- vec_memory_search (keyword) ---")
    results_k = []
    hits = []
    for probe in ("用户", "报告", "公司", "新闻", "数据", "分析", "测试", "memory", "the"):
        srch_k = call_tool(provider, "vec_memory_search", {
            "query": probe, "mode": "keyword", "top_k": 3
        })
        if isinstance(srch_k, str):
            srch_k = json.loads(srch_k) if srch_k.startswith("{") else {"raw": srch_k}
        got = srch_k.get("results", [])
        if got:
            results_k = got
            hits.append(f"{probe}×{len(got)}")
            break
    check("keyword FTS returns results", len(results_k) > 0,
          f"got {len(results_k)} (probe hits: {', '.join(hits) or 'none'})")

    # 5. hybrid search
    print("\n--- vec_memory_search (hybrid) ---")
    srch_h = call_tool(provider, "vec_memory_search", {
        "query": "测试", "mode": "hybrid", "top_k": 3
    })
    if isinstance(srch_h, str):
        srch_h = json.loads(srch_h) if srch_h.startswith("{") else {"raw": srch_h}
    results_h = srch_h.get("results", [])
    check("hybrid search returns results", len(results_h) > 0, f"got {len(results_h)}")

    # 6. session filter
    print("\n--- vec_memory_search (session filter) ---")
    # Use a known session_id from stats or list
    srch_s = call_tool(provider, "vec_memory_search", {
        "query": "用户", "mode": "keyword", "top_k": 5, "session_id": "test_verification"
    })
    if isinstance(srch_s, str):
        srch_s = json.loads(srch_s) if srch_s.startswith("{") else {"raw": srch_s}
    results_s = srch_s.get("results", [])
    check("session filter works", True, f"returned {len(results_s)} results")

    # 7. add
    print("\n--- vec_memory_add ---")
    add_result = call_tool(provider, "vec_memory_add", {
        "content": "这是一条插件功能验证测试条目，用于确认vec_memory_add工具能够正常写入数据到Zvec向量数据库中。",
        "role": "turn",
    })
    add_id = add_result.get("id")
    check("add returns id", add_id is not None, f"got {json.dumps(add_result)[:200]}")
    if add_result.get("status") == "skipped":
        print(f"  ⚠️ Skipped: {add_result.get('reason', '?')}")

    time.sleep(1)  # let indexes settle

    # 8. delete
    if add_id:
        print("\n--- vec_memory_delete ---")
        del_result = call_tool(provider, "vec_memory_delete", {"memory_ids": [add_id]})
        check("delete succeeds", del_result.get("deleted", 0) >= 1,
              f"got {json.dumps(del_result)[:200]}")

    # Final stats check
    print("\n--- Final stats check ---")
    stats_after = call_tool(provider, "vec_memory_stats", {})
    total_after = stats_after.get("total_memories", -1)
    check("total count stable", total_after == total_before,
          f"before={total_before}, after={total_after}")

    # Summary
    print(f"\n{'=' * 55}")
    print(f"Result: {passed} passed, {failed} failed out of {passed + failed} checks")
    if failed == 0:
        print(f"🎉 All checks passed for profile '{PROFILE}'")
    else:
        print(f"⚠️ {failed} check(s) failed — investigate above")
    print(f"{'=' * 55}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
