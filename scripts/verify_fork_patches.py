#!/usr/bin/env python3
"""Verify that all custom fork patches are present in the codebase.

Reads PATCHES.yaml from the repo root and checks each marker string
exists in the corresponding file. Also runs ast.parse() on all patched
files to catch syntax errors.

Usage:
    python3 scripts/verify_fork_patches.py [--repo /path/to/repo]

Exit code: 0 if all markers present, 1 if any missing.
"""

import ast
import os
import sys
import yaml  # pyyaml

def find_repo_root():
    """Walk up from this script to find the repo root (contains PATCHES.yaml)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, 'PATCHES.yaml')):
            return d
        d = os.path.dirname(d)
    # Fallback: use argument or cwd
    return os.getcwd()

def verify_patches(repo_root):
    """Check all patches defined in PATCHES.yaml. Returns (ok, results)."""
    patches_path = os.path.join(repo_root, 'PATCHES.yaml')
    if not os.path.exists(patches_path):
        print(f"ERROR: {patches_path} not found")
        return False, []

    with open(patches_path) as f:
        data = yaml.safe_load(f)

    all_ok = True
    results = []

    for patch in data.get('patches', []):
        name = patch['name']
        print(f"\n📋 Patch: {name}")
        patch_ok = True

        seen_files = set()
        for file_spec in patch.get('files', []):
            fpath = file_spec['path']
            full_path = os.path.join(repo_root, fpath)

            # Deduplicate (same file can appear twice with different markers)
            if fpath in seen_files:
                # Just check the additional markers
                for marker in file_spec.get('markers', []):
                    if not os.path.exists(full_path):
                        print(f"  ❌ {fpath}: FILE NOT FOUND")
                        patch_ok = False
                        continue
                    with open(full_path) as fh:
                        content = fh.read()
                    if marker not in content:
                        print(f"  ❌ {fpath}: missing marker '{marker}'")
                        patch_ok = False
                continue
            seen_files.add(fpath)

            if not os.path.exists(full_path):
                print(f"  ❌ {fpath}: FILE NOT FOUND")
                patch_ok = False
                continue

            with open(full_path) as fh:
                content = fh.read()

            # Syntax check
            try:
                ast.parse(content)
            except SyntaxError as e:
                print(f"  ❌ {fpath}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
                patch_ok = False

            # Marker check
            for marker in file_spec.get('markers', []):
                if marker not in content:
                    print(f"  ❌ {fpath}: missing marker '{marker}'")
                    patch_ok = False

            if patch_ok:
                marker_count = len(file_spec.get('markers', []))
                print(f"  ✅ {fpath}: {marker_count} markers OK, syntax valid")

        status = "✅ PASS" if patch_ok else "❌ FAIL"
        results.append((name, patch_ok))
        print(f"  {status}")
        if not patch_ok:
            all_ok = False

    return all_ok, results

def main():
    repo_root = find_repo_root()
    if '--repo' in sys.argv:
        idx = sys.argv.index('--repo')
        repo_root = sys.argv[idx + 1]

    print(f"Verifying fork patches in: {repo_root}")
    ok, results = verify_patches(repo_root)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")

    if ok:
        print("\n✅ All patches verified.")
        sys.exit(0)
    else:
        print("\n❌ Some patches are missing or broken!")
        print("   → Check the output above for missing markers")
        print("   → Re-apply from spec or original commits")
        sys.exit(1)

if __name__ == '__main__':
    main()
