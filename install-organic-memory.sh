#!/usr/bin/env bash
# Organic Memory Architecture — One-Line Installer for Linux/macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/20231118185SSPU/hermes-agent/feat/organic-memory-architecture/install-organic-memory.sh | bash

set -euo pipefail

echo ""
echo "🧬 Organic Memory Architecture Installer"
echo ""

# Find Hermes installation
HERMES_DIR=""
for dir in "$HOME/.hermes/hermes-agent" "$HOME/.hermes" "$HOME/hermes-agent"; do
    if [ -f "$dir/agent/memory_manager.py" ]; then
        HERMES_DIR="$dir"
        break
    fi
done

if [ -z "$HERMES_DIR" ]; then
    echo "❌ Could not find Hermes installation."
    echo "   Install Hermes first: curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash"
    exit 1
fi

echo "✅ Found Hermes at: $HERMES_DIR"

# Backup
echo "📦 Creating backup..."
cd "$HERMES_DIR"
git stash push -m "organic-memory-installer-backup" 2>/dev/null || true

# Download files
echo "📥 Downloading organic memory files..."
BASE_URL="https://raw.githubusercontent.com/20231118185SSPU/hermes-agent/feat/organic-memory-architecture"

FILES=(
    "agent/memory_pipeline.py:agent/memory_pipeline.py"
    "plugins/memory/holographic/episodic.py:plugins/memory/holographic/episodic.py"
    "plugins/memory/holographic/dreaming.py:plugins/memory/holographic/dreaming.py"
    "ORGANIC_MEMORY.md:ORGANIC_MEMORY.md"
)

for entry in "${FILES[@]}"; do
    src="${entry%%:*}"
    dst="${entry##*:}"
    dst_dir=$(dirname "$HERMES_DIR/$dst")
    mkdir -p "$dst_dir"
    if curl -fsSL "$BASE_URL/$src" -o "$HERMES_DIR/$dst" 2>/dev/null; then
        echo "  ✅ $src"
    else
        echo "  ❌ $src (download failed)"
    fi
done

# Patch memory_manager.py
echo "🔧 Patching memory_manager.py..."
MANAGER="$HERMES_DIR/agent/memory_manager.py"

if ! grep -q "from agent.memory_pipeline import" "$MANAGER" 2>/dev/null; then
    # Add import
    sed -i.bak 's/from agent.memory_provider import MemoryProvider/from agent.memory_provider import MemoryProvider\nfrom agent.memory_pipeline import MemoryPipeline, _load_pipeline_config/' "$MANAGER"

    # Add _pipeline field
    sed -i.bak 's/self._has_external: bool = False/self._has_external: bool = False\n        self._pipeline: MemoryPipeline | None = None/' "$MANAGER"

    rm -f "$MANAGER.bak"
    echo "  ✅ memory_manager.py patched (import + field)"
    echo "  ⚠️  Full merge recommended for lifecycle wiring."
else
    echo "  ⚠️  memory_manager.py already has pipeline import"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Installation complete!"
echo ""
echo "To enable, add to your config.yaml:"
echo ""
echo "  memory:"
echo "    provider: holographic"
echo "    pipeline:"
echo "      enabled: true"
echo "      episodic:"
echo "        enabled: true"
echo "      dreaming:"
echo "        enabled: true"
echo ""
echo "Then restart: hermes"
echo ""
echo "Full merge (recommended for all features):"
echo "  cd $HERMES_DIR"
echo "  git remote add fork https://github.com/20231118185SSPU/hermes-agent.git"
echo "  git fetch fork feat/organic-memory-architecture"
echo "  git merge fork/feat/organic-memory-architecture --no-edit"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
