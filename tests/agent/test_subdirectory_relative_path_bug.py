# Regression test for subdirectory hints relative path bug

import os
from pathlib import Path

# Simulate the bug scenario
working_dir = Path.home() / '.hermes' / 'hermes-agent'
directory = Path('agent')
filename = 'AGENTS.md'

# Bug reproduction: hint_path constructed without anchoring to working_dir
hint_path = directory / filename
print(f"Bug path: {hint_path}")
print(f"Resolved: {hint_path.resolve()}")

# Expected: working_dir / directory / filename
expected_path = working_dir / directory / filename
print(f"Expected: {expected_path}")

# Verify they match
assert hint_path.resolve() == expected_path, f"Path mismatch: {hint_path.resolve()} != {expected_path}"

print("✓ Test passed: relative paths are correctly anchored to working_dir")
