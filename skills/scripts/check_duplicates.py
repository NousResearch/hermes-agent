#!/usr/bin/env python3
"""Check for duplicate skill names in registry."""
import json
import sys

idx = json.load(open('skills/index-cache/registry.json'))
names = [s['name'] for s in idx['skills']]
dupes = [n for n in names if names.count(n) > 1]
if dupes:
    print(f"DUPLICATE SKILL NAMES: {dupes}")
    sys.exit(1)
print("No duplicate skill names")