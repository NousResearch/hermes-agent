#!/usr/bin/env python3
"""
Extract worktree-related functions from original PR commit (864d03c3f3)
and apply them to the current upstream-based server.py.
"""
import subprocess
import sys

WORKTREE = "/Users/mikedemott/.hermes/worktrees/hermes-pr-116"
ORIGINAL_PR = "864d03c3f3"

def run(cmd, **kwargs):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WORKTREE, **kwargs)

# Step 1: Extract the worktree functions block from original PR
# We need lines 50-78 (globals + retry helpers), 124-130 (pending dicts), 146-150 (lock helper),
# 486-506+ (worktree metadata/lease/policy), 619-760 (manager + bind functions), 777+ (prompt fragment)
# and 2956+ (prewarm)

# Get the full original file
result = run(f"git show {ORIGINAL_PR}:tui_gateway/server.py")
original_lines = result.stdout.splitlines(keepends=True)

# Extract the blocks we need
blocks = []

# Block 1: Lines 50-78 (globals + retry lease functions, 0-indexed: 49-77)
blocks.append(("globals_retry", original_lines[49:78]))

# Block 2: Lines 124-130 (pending dicts)
blocks.append(("pending_dicts", original_lines[123:130]))

# Block 3: Lines 130-133 (batch clarify comment)
blocks.append(("batch_clarify", original_lines[129:133]))

# Block 4: Lines 146-150 (session prompt submit lock)
blocks.append(("prompt_submit_lock", original_lines[145:150]))

# Block 5: Lines 486-543 (worktree metadata, lease, remove, policy functions)
blocks.append(("worktree_helpers", original_lines[485:543]))

# Block 6: Lines 619-760 (conversation_worktree_manager + bind functions)
blocks.append(("worktree_manager", original_lines[618:760]))

# Block 7: Lines 777-810 (prompt fragment)
blocks.append(("prompt_fragment", original_lines[776:810]))

# Block 8: Lines 2956-3020 (prewarm pending)
blocks.append(("prewarm", original_lines[2955:3020]))

# Now build the patch file
patch_content = []
patch_content.append("=== BLOCK 1: globals + retry lease (insert after 'logger = logging.getLogger(__name__)') ===")
for line in blocks[0][1]:
    patch_content.append(line.rstrip())

patch_content.append("\n=== BLOCK 2-3: pending dicts + batch clarify (insert after '_methods' dict) ===")
for line in blocks[1][1]:
    patch_content.append(line.rstrip())
for line in blocks[2][1]:
    patch_content.append(line.rstrip())

patch_content.append("\n=== BLOCK 4: prompt_submit_lock (insert before '_ws_orphan_setting') ===")
for line in blocks[3][1]:
    patch_content.append(line.rstrip())

patch_content.append("\n=== BLOCK 5: worktree helpers (insert before _conversation_worktree_manager) ===")
for line in blocks[4][1]:
    patch_content.append(line.rstrip())

patch_content.append("\n=== BLOCK 6: worktree manager + bind functions ===")
for line in blocks[5][1]:
    patch_content.append(line.rstrip())

patch_content.append("\n=== BLOCK 7: prompt fragment ===")
for line in blocks[6][1]:
    patch_content.append(line.rstrip())

patch_content.append("\n=== BLOCK 8: prewarm pending ===")
for line in blocks[7][1]:
    patch_content.append(line.rstrip())

# Write the extracted functions to a file
with open(f"{WORKTREE}/tui_gateway/_worktree_functions.py", "w") as f:
    f.write("# Extracted worktree functions from original PR commit\n")
    f.write("# These need to be merged into server.py\n\n")
    for name, lines in blocks:
        f.write(f"\n# --- {name} ---\n")
        for line in lines:
            f.write(line)

print(f"Extracted {len(blocks)} function blocks to tui_gateway/_worktree_functions.py")
print(f"Total lines: {sum(len(b[1]) for b in blocks)}")
