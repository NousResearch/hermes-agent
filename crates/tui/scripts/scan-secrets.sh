#!/usr/bin/env bash
# Fail if high-confidence secret shapes appear outside redaction fixtures.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$root" <<'PY'
import os, re, sys
root = sys.argv[1]
pat = re.compile(
    r"AKIA[0-9A-Z]{16}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|gho_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|sk-ant-[A-Za-z0-9\-]{8,}"
    r"|BEGIN (OPENSSH |RSA )?PRIVATE KEY"
)
skip_dir = {".git", "target", "upstream"}
hits = []
for dirpath, dirnames, files in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip_dir]
    for name in files:
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root)
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "redact_gateway_line(" in line:
                continue
            if pat.search(line):
                hits.append(f"{rel}:{i}:{line.strip()}")
if hits:
    print("scan-secrets: possible secret material:", file=sys.stderr)
    print("\n".join(hits), file=sys.stderr)
    sys.exit(1)
print("scan-secrets: clean")
PY
