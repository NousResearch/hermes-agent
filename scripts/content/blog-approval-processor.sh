#!/bin/bash
# P13 isolation: BLOG_APPROVAL_DRY_RUN=1 short-circuits before any Python
# invocation that flips MDX, builds, or pushes.
set -euo pipefail

if [[ ${BLOG_APPROVAL_DRY_RUN:-0} == "1" ]]; then
    echo "dry-run: would process approvals"
    exit 0
fi

# Cron supplies the canonical runtime checkout as workdir.  An explicit
# override supports controlled relocation without reintroducing a hard-coded
# checkout path.
RUNTIME_ROOT="${HERMES_AGENT_ROOT:-$PWD}"
CE="$RUNTIME_ROOT/content_engine"
[[ -d "$CE" ]] || { echo "ERROR: content engine not found under runtime root: $RUNTIME_ROOT" >&2; exit 1; }
cd "$CE"
export PYTHONPATH="${CE}${PYTHONPATH:+:$PYTHONPATH}"

python3 <<'PY'
from blog.blog_approval import _read_tracker, publish

entries = _read_tracker()
approved = [e for e in entries if e.get("status") == "approved"]
if not approved:
    raise SystemExit(0)

results = []
for entry in approved:
    slug = entry.get("slug", "")
    title = entry.get("title", slug)
    result = publish(slug)
    results.append({
        "slug": slug,
        "title": title,
        "status": result.get("status", "unknown"),
        "detail": result,
    })

published = [r for r in results if r["status"] == "ok"]
failed = [r for r in results if r["status"] != "ok"]

lines = [
    "[Blog Publish]",
    "━━━ **🚀 Publish Status** ━━━",
]

if published:
    lines.append("**Published**")
    for r in published:
        lines.append(f"- `{r['slug']}` — {r['title']}")

if failed:
    if published:
        lines.append("")
    lines.append("**Not published**")
    for r in failed:
        detail = r["detail"]
        extra = detail.get("error") or detail.get("message") or ""
        extra = f" — {extra}" if extra else ""
        lines.append(f"- `{r['slug']}` — {r['title']} (`{r['status']}`{extra})")

print("\n".join(lines))
PY
