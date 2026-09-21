#!/usr/bin/env python3
"""Multi-profile Zvec memory health check.

Checks all Hermes profiles' zvec_memory collections for:
- Collection existence
- LOCK file status (locked by gateway?)
- HNSW index completeness
- FTS index (jieba tokenizer)
- Quick FTS search test (Chinese + English)

Usage:
    python3 check_all_profiles.py
    HERMES_PROFILE=chip_expert python3 check_all_profiles.py  # single profile

Requires zvec installed in the Python environment running this script.
"""
import os
import sys
import subprocess

def get_profiles():
    """Return list of profile names."""
    single = os.environ.get("HERMES_PROFILE", "")
    if single:
        return [single]
    base = os.path.expanduser("~/.hermes")
    profiles = ["default"]
    profiles_dir = os.path.join(base, "profiles")
    if os.path.isdir(profiles_dir):
        for entry in sorted(os.listdir(profiles_dir)):
            config = os.path.join(profiles_dir, entry, "config.yaml")
            if os.path.isfile(config):
                profiles.append(entry)
    return profiles

def get_collection_path(profile):
    if profile == "default":
        return os.path.expanduser("~/.hermes/记忆数据库/zvec_memory/memories")
    return os.path.expanduser(f"~/.hermes/profiles/{profile}/记忆数据库/zvec_memory/memories")

def check_ollama():
    """Check if Ollama is reachable."""
    try:
        result = subprocess.run(
            ["timeout", "3", "bash", "-c", "echo > /dev/tcp/localhost/11434 && echo OPEN"],
            capture_output=True, text=True, timeout=5
        )
        return "OPEN" in result.stdout
    except Exception:
        return False

def main():
    try:
        import zvec
    except ImportError:
        print("ERROR: zvec not installed. Run: pip install zvec")
        sys.exit(1)

    profiles = get_profiles()
    ollama_ok = check_ollama()
    print(f"Ollama: {'✅ reachable' if ollama_ok else '❌ not reachable'}")
    print(f"Profiles: {', '.join(profiles)}")
    print("=" * 60)

    for profile in profiles:
        path = get_collection_path(profile)
        print(f"\n--- {profile} ---")

        if not os.path.isdir(path):
            print("  ❌ Collection directory not found")
            continue

        # Check LOCK
        lock_file = os.path.join(path, "LOCK")
        locked = False
        if os.path.exists(lock_file):
            # Check if any process has fd pointing to LOCK
            try:
                result = subprocess.run(
                    ["lsof", lock_file],
                    capture_output=True, text=True, timeout=5
                )
                locked = bool(result.stdout.strip())
            except Exception:
                locked = True  # assume locked if lsof fails

        # Try to open (may fail if locked)
        try:
            coll = zvec.open(path)
        except Exception as e:
            print(f"  ⚠️  LOCKED (cannot open rw): {e}")
            # Try read-only
            try:
                coll = zvec.open(path, option=zvec.CollectionOption(read_only=True))
                print(f"  ✅ Opened read-only")
            except Exception as e2:
                print(f"  ❌ Cannot open even read-only: {e2}")
                continue

        stats = coll.stats
        doc_count = stats.doc_count
        ic = stats.index_completeness
        hnsw_pct = ic.get("vector", 0) * 100

        # Field indexes
        schema = coll.schema
        fts_ok = False
        fts_tokenizer = "N/A"
        invert_count = 0
        for f in schema.fields:
            if f.name == "content":
                if f.index_param:
                    fts_ok = True
                    fts_tokenizer = getattr(f.index_param, "tokenizer_name", "?")
            if f.index_param and "InvertIndex" in type(f.index_param).__name__:
                invert_count += 1

        # Summary
        print(f"  Docs: {doc_count}")
        print(f"  HNSW: {hnsw_pct:.1f}% {'✅' if hnsw_pct >= 99.9 else '⚠️ needs optimize()'}")
        print(f"  FTS: {'✅' if fts_ok else '❌ no index'} (tokenizer: {fts_tokenizer})")
        print(f"  InvertIndex fields: {invert_count} {'✅' if invert_count >= 3 else '⚠️ < 3'}")
        print(f"  LOCK held: {'⚠️ yes' if locked else 'no'}")

        # Quick FTS test
        if fts_ok and ollama_ok and doc_count > 0:
            for q in ["测试", "memory"]:
                try:
                    r = coll.query(zvec.Query(field_name="content", fts=zvec.Fts(match_string=q)), topk=1)
                    icon = "✅" if r else "⚠️ 0 hits"
                    print(f"  FTS test '{q}': {icon}")
                except Exception as e:
                    print(f"  FTS test '{q}': ❌ {e}")

if __name__ == "__main__":
    main()
