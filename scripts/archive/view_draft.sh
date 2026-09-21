#!/bin/bash
# view_draft.sh <draft_id> — print the full draft body for Telegram view callback.
# Uses database.get_draft() with parameterised SQL via Python (no shell-interpolation
# of the draft_id into a SQL string — was C1 SQL injection in earlier version).

set -euo pipefail

DRAFT_ID="${1:?usage: view_draft.sh <draft_id>}"

exec python3 -c '
import sys
sys.path.insert(0, "/home/kensei/repos/KenseiAgent/content_engine")
from database import get_draft
d = get_draft(sys.argv[1])
if not d:
    print("(draft not found)")
    sys.exit(1)
print("Brand:", d.get("brand") or "?")
print("Platform:", d.get("platform") or "?")
print("Pillar:", d.get("pillar") or "?")
print("Status:", d.get("status") or "?")
print("Created:", d.get("created_at") or "?")
print("---")
print(d.get("body_text") or "(no body)")
' "$DRAFT_ID"
