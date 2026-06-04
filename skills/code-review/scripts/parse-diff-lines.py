#!/usr/bin/env python3
"""Parse a unified diff and return (filepath, new_line_number) for every added line.

Usage:
    gh pr diff 123 --repo owner/repo | python3 parse-diff-lines.py
    python3 parse-diff-lines.py < diff.txt

Output: one JSON array of [filepath, line_number] pairs.
Useful for anchoring inline review comments to the correct new-file line.
"""

import json, re, sys

def parse_diff(diff_text):
    """Parse unified diff, return list of (filepath, new_line_number) for each '+' line."""
    results = []
    current_file = None
    new_pos = 0
    in_hunk = False

    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            m = re.match(r'diff --git a/(.*) b/(.*)', line)
            if m:
                current_file = m.group(2)
            in_hunk = False
        elif line.startswith('@@'):
            m = re.search(r'\+(\d+)', line)
            if m:
                new_pos = int(m.group(1))
            in_hunk = True
        elif in_hunk and current_file:
            if line.startswith('+'):
                results.append((current_file, new_pos))
                new_pos += 1
            elif line.startswith('-'):
                pass  # removed line — only old-pos advances
            else:
                new_pos += 1  # context line
    return results

if __name__ == '__main__':
    diff_text = sys.stdin.read()
    results = parse_diff(diff_text)
    print(json.dumps(results, indent=2))
