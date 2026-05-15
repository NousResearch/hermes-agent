#!/bin/bash
# autolycus-upstream-sync.sh
# Fetches upstream, rebases our fork, resolves known conflicts, and pushes.

set -e

REPO_DIR="$HOME/compiled/autolycus-agent"
cd "$REPO_DIR"

# Fetch latest upstream
git fetch upstream

# Check if upstream has new commits
AHEAD=$(git rev-list HEAD..upstream/main --count 2>/dev/null || echo "0")

if [ "$AHEAD" -eq 0 ]; then
    echo "Already up to date with upstream/main."
    exit 0
fi

echo "Found $AHEAD new upstream commits. Rebasing..."

# Rebase onto upstream
# Use --autostash to handle uncommitted changes, --autostash to skip empty commits
git rebase --autostash --rebase-merges upstream/main 2>&1 || {
    echo "Rebase had issues, attempting to auto-resolve..."
    
    # Resolve known conflict patterns
    # README conflicts: always keep our (Autolycus) version
    while git status --porcelain | grep -q "UU README.md"; do
        git checkout --ours README.md
        git add README.md
        GIT_EDITOR=true git rebase --continue 2>&1 || true
    done
    
    # main.py conflicts: take upstream (theirs) - our rebranding is in earlier commits
    while git status --porcelain | grep -q "UU hermes_cli/main.py"; do
        git checkout --theirs hermes_cli/main.py
        git add hermes_cli/main.py
        GIT_EDITOR=true git rebase --continue 2>&1 || true
    done
    
    # cli.py conflicts: take ours (our custom additions)
    while git status --porcelain | grep -q "UU cli.py"; do
        git checkout --ours cli.py
        git add cli.py
        GIT_EDITOR=true git rebase --continue 2>&1 || true
    done
    
    # General fallback: take ours for any remaining conflicts
    while git status --porcelain | grep -q "UU"; do
        git checkout --ours $(git status --porcelain | grep "UU" | awk '{print $2}')
        git add $(git status --porcelain | grep "UU" | awk '{print $2}')
        GIT_EDITOR=true git rebase --continue 2>&1 || true
    done
    
    # Handle stuck "editing commit" states (empty commits during rebase)
    if git status | grep -q "currently editing a commit"; then
        echo "Stuck on empty commit edit, continuing..."
        GIT_EDITOR=true git rebase --continue 2>&1 || true
    fi
}

# Clean up any leftover conflict markers in all Python files
for f in $(grep -rl "^<<<<<<<\|^=======\|^>>>>>>>" --include="*.py" . 2>/dev/null); do
    sed -i '/^<<<<<<<\|^=======\|^>>>>>>>.*$/d' "$f"
    echo "Cleaned conflict markers from: $f"
done

# Verify no conflict markers remain
REMAINING=$(grep -r "^<<<<<<<\|^=======\|^>>>>>>>" --include="*.py" . 2>/dev/null | wc -l || echo "0")
if [ "$REMAINING" -gt 0 ]; then
    echo "WARNING: $REMAINING conflict markers still remain!"
    exit 1
fi

# === Fork-specific file preservation ===
# Reference commits with current fork-specific content:
#   ed5bbc826  - README.md with Autolycus branding
#   b2c98986f  - scripts/install.sh with FreeBSD detection
#   f6159a2d2  - scripts/install-autolycus.sh (dedicated installer)
#   2b2d51f79  - pyproject.toml with LGPL license + lycus entry point

# Restore Autolycus README branding if lost
if ! grep -q "Autolycus" README.md 2>/dev/null; then
    echo "Restoring Autolycus README branding..."
    git show ed5bbc826:README.md > README.md
fi

# Preserve FreeBSD OS detection in install.sh
if ! grep -q "freebsd" scripts/install.sh 2>/dev/null; then
    echo "Restoring FreeBSD detection in install.sh..."
    git show b2c98986f:scripts/install.sh > scripts/install.sh
fi

# Preserve install-autolycus.sh if lost
if [ ! -f scripts/install-autolycus.sh ]; then
    echo "Restoring scripts/install-autolycus.sh..."
    git show f6159a2d2:scripts/install-autolycus.sh > scripts/install-autolycus.sh
    chmod +x scripts/install-autolycus.sh
fi

# Preserve pyproject.toml fork-specific settings (LGPL + lycus entry point)
if ! grep -q "lycus" pyproject.toml 2>/dev/null; then
    echo "Restoring lycus entry point in pyproject.toml..."
    git show 2b2d51f79:pyproject.toml > pyproject.toml
fi
if ! grep -q "LGPL" pyproject.toml 2>/dev/null; then
    echo "Restoring LGPL license in pyproject.toml..."
    git show 2b2d51f79:pyproject.toml > pyproject.toml
fi

# Preserve our custom build_plan_path function if it was lost during rebase
# (upstream removed it but our cli.py TUI still needs it)
if ! grep -q "def build_plan_path" agent/skill_commands.py 2>/dev/null; then
    echo "Restoring build_plan_path function to agent/skill_commands.py..."
    # Add datetime import if missing
    if ! grep -q "from datetime import datetime" agent/skill_commands.py; then
        sed -i 's/^from pathlib import Path$/from datetime import datetime\nfrom pathlib import Path/' agent/skill_commands.py
    fi
    # Add the function after _SKILL_MULTI_HYPHEN if not present
    if ! grep -q "_PLAN_SLUG_RE" agent/skill_commands.py; then
        sed -i '/_SKILL_MULTI_HYPHEN = re.compile/a\
\
# Plan path helpers (removed in upstream, kept for TUI /plan command compatibility)\
_PLAN_SLUG_RE = re.compile(r"[^a-z0-9-]")\
\
def build_plan_path(\
    user_instruction: str = "",\
    *,\
    now: datetime | None = None,\
) -> Path:\
    """Return the default workspace-relative markdown path for a /plan invocation."""\
    slug_source = (user_instruction or "").strip().splitlines()[0] if user_instruction else ""\
    slug = _PLAN_SLUG_RE.sub("-", slug_source.lower()).strip("-")\
    if slug:\
        slug = "-".join(part for part in slug.split("-")[:8] if part)[:48].strip("-")\
    slug = slug or "conversation-plan"\
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")\
    return Path(".hermes") / "plans" / f"{timestamp}-{slug}.md"
' agent/skill_commands.py
    fi
fi

# Preserve custom_provider_slug function if it was lost
# (upstream removed it but our model_switch.py still needs it)
if ! grep -q "def custom_provider_slug" hermes_cli/providers.py 2>/dev/null; then
    echo "Restoring custom_provider_slug to hermes_cli/providers.py..."
    cat >> hermes_cli/providers.py << 'PYEOF'


def custom_provider_slug(display_name: str) -> str:
    """Build a canonical slug for a custom_providers entry.

    Matches the convention used by runtime_provider and credential_pool
    (``custom:<normalized-name>``).  Centralised here so all call-sites
    produce identical slugs.
    """
    return "custom:" + display_name.strip().lower().replace(" ", "-")
PYEOF
fi

# === Post-sync patches: remove Windows-specific files and CI jobs ===
# These files are not needed on our FreeBSD/Linux-focused fork
if [ -f scripts/check-windows-footguns.py ]; then
    echo "Removing scripts/check-windows-footguns.py..."
    rm scripts/check-windows-footguns.py
fi
if [ -f scripts/install.cmd ]; then
    echo "Removing scripts/install.cmd..."
    rm scripts/install.cmd
fi
if [ -f scripts/install.ps1 ]; then
    echo "Removing scripts/install.ps1..."
    rm scripts/install.ps1
fi

# Remove windows-footguns job from lint workflow if present
if grep -q "windows-footguns:" .github/workflows/lint.yml 2>/dev/null; then
    echo "Removing windows-footguns job from lint.yml..."
    python3 -c "
import re
path = '.github/workflows/lint.yml'
with open(path) as f:
    content = f.read()
# Remove the windows-footguns job block (indented under jobs:)
content = re.sub(r'\n  windows-footguns:.*?(?=\n  [a-z]|\n\njobs:|\Z)', '', content, flags=re.DOTALL)
with open(path, 'w') as f:
    f.write(content)
"
fi

# Push to origin
# Safety check: ensure rebase is complete before committing
if git status | grep -q "rebase in progress\|rebase-merge\|rebase-apply"; then
    echo "ERROR: Rebase appears incomplete, aborting sync!"
    exit 1
fi

git add -A
git commit -m "upstream-sync: auto-rebase cleanup $(date -u +%Y-%m-%dT%H:%M:%SZ)" --allow-empty
git push origin main --force-with-lease

echo "Sync complete. Fork is now on top of upstream/main."
