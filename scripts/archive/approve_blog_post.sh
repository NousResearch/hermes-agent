#!/bin/bash
# Approve a SahilBlog draft post and publish it (build + push).
# Usage:
#   approve_blog_post.sh <slug>
# Examples:
#   approve_blog_post.sh token-maxing-at-the-edge
#
# Side effects:
#   1. Flips approved:false -> approved:true in the MDX frontmatter
#   2. Runs pnpm build in ~/repos/SahilBlog (must exit 0)
#   3. Commits "publish: <title>"
#   4. Pushes to origin main
#
# This is the manual approval bridge. The cron stages drafts with approved:false;
# this script flips them when Sahil is ready to publish.

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: approve_blog_post.sh <slug>" >&2
    echo "  Example: approve_blog_post.sh token-maxing-at-the-edge" >&2
    echo "" >&2
    echo "  To see staged drafts:" >&2
    echo "    cd ~/repos/SahilBlog && git log --oneline | grep 'draft:'" >&2
    exit 2
fi

SLUG="$1"
CE_ROOT=/home/kensei/repos/KenseiAgent/content_engine
VENV_PY=/home/kensei/repos/KenseiAgent/.venv/bin/python

cd "$CE_ROOT"
export PYTHONPATH="$CE_ROOT:${PYTHONPATH:-}"

echo "$(date -Iseconds) - Approving blog post: $SLUG"
"$VENV_PY" -c "
import json
import blog.blog_publisher as bp
result = bp.approve('$SLUG')
print(json.dumps(result, indent=2))
if result['status'] != 'ok':
    import sys
    sys.exit(1)
"

echo "$(date -Iseconds) - Done. Post $SLUG published to origin/main."