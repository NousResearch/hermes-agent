---
name: remote-file-patch-via-scp
description: Patch files on a remote host (e.g. Pluto over SSH) without shell-escaping hell. Write a Python patch script locally, scp it, then execute it remotely. Use when inline SSH string manipulation fails due to quotes, special chars, or multi-line replacements.
tags: [ssh, scp, pluto, remote, patch, file-edit]
---

# Remote File Patch via SCP

## When to Use

- You need to edit a file on Pluto (or any SSH host) with complex string replacements
- Inline `ssh host 'sed ...'` or `ssh host \"python3 -c '...'\"` fails due to escaping
- The replacement contains quotes, newlines, special characters, or is multi-line
- You want reliable, auditable, idempotent edits on remote files

## Two Patterns

### Pattern A: Inline SSH heredoc (faster, no temp files)

Use when the replacement logic is moderate and doesn't need local file references.

```bash
ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 "
python3 << 'EOF'
with open('/remote/path/file.js') as f:
    content = f.read()

block = '''
// new block to insert
const FOO = 'bar';
'''

marker = 'const PORT = ...'  # exact string from file
if 'const FOO' not in content and marker in content:
    content = content.replace(marker, marker + block)
    with open('/remote/path/file.js', 'w') as f:
        f.write(content)
    print('Inserted block')
else:
    print('Skipped: already present or marker not found')
EOF
"
```

**When heredoc fails** (special chars in the block or exit code 255 from SSH), fall back to Pattern B.

### Pattern B: SCP then execute (most reliable, use for complex cases)

**Do NOT try to inline Python or sed over SSH for complex replacements — it always fails.**

1. Write a `.py` patch script locally using `write_file`
2. `scp` it to the remote host
3. `ssh` to execute it
4. Print result and check exit code

## Template

```python
# /tmp/patch_<target>.py
path = '/remote/path/to/file.md'
with open(path, 'r') as f:
    content = f.read()

old = "exact string to find"
new = "replacement string"

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('Patched successfully')
else:
    print('String not found — no changes made')
    print('Expected:', repr(old[:80]))
```

Then execute:

```bash
scp -i ~/.ssh/hermes_access_ed25519 /tmp/patch_target.py sander@100.108.223.25:/tmp/patch_target.py \
  && ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 'python3 /tmp/patch_target.py'
```

Or combined via execute_code:

```python
from hermes_tools import terminal

result = terminal(
    "scp -i ~/.ssh/hermes_access_ed25519 /tmp/patch_target.py sander@100.108.223.25:/tmp/patch_target.py "
    "&& ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 'python3 /tmp/patch_target.py'"
)
print(result['output'])
```

## Pluto SSH Details

- Host: sander@100.108.223.25 (Tailscale)
- Key: ~/.ssh/hermes_access_ed25519

## Multi-block Replacement

For multiple replacements in one file, use a list of (old, new) tuples:

```python
path = '/remote/path/to/file.md'
with open(path, 'r') as f:
    content = f.read()

patches = [
    ("old string 1", "new string 1"),
    ("old string 2", "new string 2"),
]

applied = []
for old, new in patches:
    if old in content:
        content = content.replace(old, new)
        applied.append(old[:40])
    else:
        print(f'NOT FOUND: {repr(old[:60])}')

with open(path, 'w') as f:
    f.write(content)

print(f'Applied {len(applied)} patches:', applied)
```

## Pitfalls

- Always print the first 80 chars of the expected string when not found — helps debug whitespace/indent mismatches
- Use `repr()` in debug output so hidden characters (tabs, \\r, trailing spaces) are visible
- For YAML/JSON files, prefer loading and re-serializing instead of string replacement — avoids corrupting structure
- Clean up /tmp/*.py after use if the script contained sensitive data
- `scp` and `ssh` must be in separate steps if you need the exit code of the patch script itself
- If the file uses Windows line endings (\\r\\n), `old` must include \\r or the match will fail — check with `repr(content[:200])`

## Idempotency Guard: Be Specific

When checking if a block is already present, use the **declaration string**, not a substring that may appear elsewhere:

```python
# WRONG: 'OLLAMA_FALLBACK' appears in references even before the const is declared
if 'OLLAMA_FALLBACK' not in content:
    ...  # skipped even though const block was never inserted

# CORRECT: check for the exact declaration
if 'const OLLAMA_FALLBACK' not in content:
    ...  # only skipped if the actual declaration exists
```

This catches the case where a variable name is used in the file (e.g. in function calls) before the declaration block is inserted.

## SSH Heredoc: exit code 255

If `ssh ... "python3 << 'EOF' ... EOF"` returns exit_code 255, the SSH connection itself failed (timeout, key issue, or the heredoc confuses the shell quoting). Switch to Pattern B (SCP) immediately — don't retry the heredoc.
