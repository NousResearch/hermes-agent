#!/usr/bin/env python3
"""
traceback_pruner — Programmatic python tool to sanitize and compact stack traces.
Filters out third-party site-packages and standard library frames, leaving only 
first-party application development frames. Saves up to 80% context tokens for agents.
"""

import sys
import re

# Lines to ignore in pytest or general tracebacks
IGNORE_PATTERNS = [
    r"site-packages/pytest",
    r"site-packages/pluggy",
    r"site-packages/xdist",
    r"bin/pytest",
    r"/usr/lib/python",
    r"/Cellar/python",
    r"/venv/lib/",
    r"\.venv/lib/",
]

def compact_traceback(stream_text: str) -> str:
    lines = stream_text.splitlines()
    output_lines = []
    
    in_traceback = False
    skipped_frames = 0
    
    # Process line-by-line
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect traceback block
        if "Traceback (most recent call last)" in line or "Traceback:" in line:
            in_traceback = True
            output_lines.append(line)
            i += 1
            continue
            
        if in_traceback:
            # Check if this line describes a file frame link (Standard Python format)
            # e.g.: '  File "/path/to/file.py", line X, in function'
            if line.startswith("  File ") and 'line ' in line:
                # Retrieve the next line too (the code statement) as it is coupled
                stmt_line = lines[i+1] if i+1 < len(lines) else ""
                
                # Check match against ignores
                should_ignore = any(re.search(pat, line) for pat in IGNORE_PATTERNS)
                
                if should_ignore:
                    skipped_frames += 1
                    i += 2  # skip both frame and code description statements
                    continue
                else:
                    if skipped_frames > 0:
                        output_lines.append(f"  [... {skipped_frames} external library frames omitted for token conservation ...]")
                        skipped_frames = 0
                    output_lines.append(line)
                    if stmt_line:
                        output_lines.append(stmt_line)
                    i += 2
                    continue
            
            # Detect end of traceback: non-indented lines that describe exception values
            # e.g.: 'TypeError: ...' or 'AssertionError: ...'
            if line and not line.startswith(" ") and ":" in line:
                if skipped_frames > 0:
                    output_lines.append(f"  [... {skipped_frames} external library frames omitted for token conservation ...]")
                    skipped_frames = 0
                in_traceback = False
                output_lines.append(line)
                i += 1
                continue
                
        output_lines.append(line)
        i += 1
        
    return "\n".join(output_lines)

if __name__ == "__main__":
    # If piped via stdin (e.g., pytest tests.py | python compact_trace.py)
    if not sys.stdin.isatty():
        raw_text = sys.stdin.read()
        print(compact_traceback(raw_text))
    else:
        # Help details
        print("Usage: python compact_trace.py < traceback_file.txt")
        print("Or pipe directly: pytest tests/test_abc.py | python compact_trace.py")
