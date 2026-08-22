#!/usr/bin/env bash
# discover-repos.sh — Automatically discover all Git repos under a workspace root
# Usage: ./discover-repos.sh [workspace_root=/home/deploy]
#
# Scans for all .git directories, extracts metadata, and outputs JSON
# ready for repo_mappings.json schema v2.

set -euo pipefail

WORKSPACE="${1:-/home/deploy}"

echo '[' > /tmp/repos-dump.json
first=true

find "$WORKSPACE" -maxdepth 5 -name '.git' -type d 2>/dev/null |
  while read -r gitdir; do
    repodir="$(dirname "$gitdir")"
    
    # Skip hidden dirs and node_modules
    case "$repodir" in
      */node_modules/*|*/__pycache__/*|*/venv/*|*/.venv/*) continue ;;
    esac

    # Skip if it's a worktree (gitdir is a file, not a dir)
    [ -f "$gitdir" ] && continue

    name="$(basename "$repodir")"
    branch="$(cd "$repodir" && git branch --show-current 2>/dev/null || echo '(detached)')"
    head="$(cd "$repodir" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
    remote="$(cd "$repodir" && git remote -v 2>/dev/null | head -1 | awk '{print $2}')"
    dirty="$(cd "$repodir" && git status --porcelain 2>/dev/null | wc -l)"
    remote_count="$(cd "$repodir" && git remote 2>/dev/null | wc -l)"
    
    # Detect if worktree
    git_dir_path="$(cd "$repodir" && git rev-parse --git-dir 2>/dev/null || echo '')"
    parent=$(echo "$git_dir_path" | grep -oP 'worktrees/\K[^/]+' || echo '')
    parent_repo="${parent:-}"
    git_root="$(cd "$repodir" && git rev-parse --show-toplevel 2>/dev/null || echo '')"

    # Check for AGENTS.md, CLAUDE.md, README.md
    has_agents=$([ -f "$repodir/AGENTS.md" ] && echo true || echo false)
    has_claude=$([ -f "$repodir/CLAUDE.md" ] && echo true || echo false)
    has_readme=$([ -f "$repodir/README.md" ] && echo true || echo false)

    if [ "$first" = true ]; then
      first=false
    else
      echo ',' >> /tmp/repos-dump.json
    fi

    cat >> /tmp/repos-dump.json <<JSONDATA
    {
      "name": "$name",
      "path": "$repodir",
      "remote": "${remote:-null}",
      "remote_count": $remote_count,
      "branch": "$branch",
      "HEAD": "$head",
      "dirty": $dirty,
      "has_AGENTS_md": $has_agents,
      "has_CLAUDE_md": $has_claude,
      "has_README_md": $has_readme,
      "git_root": "$git_root",
      "parent_repo": "${parent_repo:-null}"
    }
JSONDATA
  done

echo ']' >> /tmp/repos-dump.json

# Pretty-print the JSON
python3 -c "
import json, sys
with open('/tmp/repos-dump.json') as f:
    data = json.load(f)
# Sort by name
data.sort(key=lambda x: x['name'])
with open('/tmp/repos-dump.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'✅ {len(data)} repos découverts dans $WORKSPACE')
print(f'   Fichier: /tmp/repos-dump.json')
for r in data:
    print(f'   {r[\"name\"]:35s} {r[\"branch\"]:25s} {r.get(\"remote\",\"(local)\") or \"(local)\"}')
" 2>&1
