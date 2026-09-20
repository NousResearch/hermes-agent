#!/usr/bin/env python3
"""Insert missing worktree functions from original PR into upstream server.py."""
import subprocess

WORKTREE = "/Users/mikedemott/.hermes/worktrees/hermes-pr-116"

def get_original_file(path, commit="864d03c3f3"):
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        capture_output=True, text=True, cwd=WORKTREE
    )
    return result.stdout

def extract_functions(source, func_names):
    """Extract complete function bodies from source code."""
    lines = source.split('\n')
    extracted = {}
    
    for func_name in func_names:
        # Find the function definition
        start = None
        for i, line in enumerate(lines):
            if f'def {func_name}(' in line or f'def {func_name}(' in line:
                start = i
                break
        
        if start is None:
            continue
        
        # Find the end of the function (next def at same indentation or end of file)
        func_indent = len(lines[start]) - len(lines[start].lstrip())
        end = len(lines)
        
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= func_indent and (line.startswith('def ') or line.startswith('class ')):
                end = i
                break
        
        extracted[func_name] = '\n'.join(lines[start:end])
    
    return extracted

# Get original PR's server.py
original_server = get_original_file("tui_gateway/server.py")

# Extract the globals and functions we need
globals_block = """_failed_conversation_root_leases: list[object] = []
_failed_conversation_root_leases_lock = threading.Lock()
_failed_conversation_root_lease_retry_timer = None

"""

functions = extract_functions(original_server, [
    '_remember_failed_conversation_root_lease',
    '_retry_failed_conversation_root_leases',
    '_session_prompt_submit_lock',
    '_conversation_worktree_metadata',
    '_acquire_conversation_root_lease',
    '_remove_failed_conversation_worktree',
    '_conversation_worktree_policy_for_session',
    '_conversation_worktree_manager',
    '_resolve_existing_conversation_worktree',
    '_bind_conversation_worktree_for_new_root',
    '_bind_conversation_worktree_on_submit',
    '_conversation_worktree_prompt_fragment',
    '_conversation_worktree_prewarm_pending',
])

# Also extract global dicts
global_dicts = """_pending: dict[str, tuple[str, threading.Event]] = {}
_pending_prompt_payloads: dict[str, tuple[str, dict]] = {}
_answers: dict[str, str] = {}
_batch_clarify: dict[str, dict] = {}

"""

# Now insert into upstream server.py
server_path = f"{WORKTREE}/tui_gateway/server.py"
with open(server_path) as f:
    content = f.read()

# 1. Insert globals after `logger = logging.getLogger(__name)`
insert_after = "logger = logging.getLogger(__name__)"
if "_failed_conversation_root_leases" not in content:
    content = content.replace(
        insert_after,
        insert_after + "\n\n" + globals_block + global_dicts,
        1  # Only first occurrence
    )

# 2. Insert functions at end of file
functions_block = "\n\n# ── Conversation worktree (fork-specific) ──\n\n"
for func_name in [
    '_remember_failed_conversation_root_lease',
    '_retry_failed_conversation_root_leases',
    '_session_prompt_submit_lock',
    '_conversation_worktree_metadata',
    '_acquire_conversation_root_lease',
    '_remove_failed_conversation_worktree',
    '_conversation_worktree_policy_for_session',
    '_conversation_worktree_manager',
    '_resolve_existing_conversation_worktree',
    '_bind_conversation_worktree_for_new_root',
    '_bind_conversation_worktree_on_submit',
    '_conversation_worktree_prompt_fragment',
    '_conversation_worktree_prewarm_pending',
]:
    if func_name in functions:
        functions_block += functions[func_name] + "\n\n"

content += functions_block

with open(server_path, 'w') as f:
    f.write(content)

print(f"Inserted {len(functions)} functions into server.py")
