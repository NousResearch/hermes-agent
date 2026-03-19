from cosmos.core.evolution.codebase_context import CodebaseContext
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

try:
    ctx = CodebaseContext()
    print("--- SCANNING FILE TREE ---")
    tree = ctx.scan_file_tree(max_depth=2)
    print(tree)
    print("--- END TREE ---")
    
    if len(tree) < 50:
        print("WARNING: Tree seems textually too short!")
    else:
        print(f"Tree length: {len(tree)} chars")

except Exception as e:
    print(f"ERROR: {e}")
