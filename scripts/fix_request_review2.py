#!/usr/bin/env python3
"""Fix request_review - restore from upstream."""
import subprocess

# Get upstream version
result = subprocess.run(
    ['git', 'show', '30b6b4d99c:hermes_cli/kanban_db.py'],
    cwd='/Users/mikedemott/hermes-fork-work/fork-repo',
    capture_output=True, text=True
)
upstream = result.stdout

# Find the request_review function in upstream
start = upstream.find('def request_review(')
if start == -1:
    print("Could not find request_review in upstream")
    exit(1)

# Find the end of request_review (next def at same level)
end = upstream.find('\ndef ', start + 1)
if end == -1:
    end = len(upstream)

upstream_func = upstream[start:end]

# Read current file
with open('/Users/mikedemott/hermes-fork-work/fork-repo/hermes_cli/kanban_db.py', 'r') as f:
    current = f.read()

# Find the request_review function in current
start_curr = current.find('def request_review(')
if start_curr == -1:
    print("Could not find request_review in current")
    exit(1)

# Find the end of request_review in current
end_curr = current.find('\ndef ', start_curr + 1)
if end_curr == -1:
    end_curr = len(current)

# Replace
new_content = current[:start_curr] + upstream_func + current[end_curr:]

with open('/Users/mikedemott/hermes-fork-work/fork-repo/hermes_cli/kanban_db.py', 'w') as f:
    f.write(new_content)

print("Replaced request_review from upstream!")