#!/usr/bin/env bash
# gitlawb quick setup — install CLI, create identity, register with node
set -euo pipefail

NODE_URL="${GITLAWB_NODE:-https://node.gitlawb.com}"

echo "=== gitlawb setup ==="

# Install gl CLI if not present
if ! command -v gl &>/dev/null; then
  echo "Installing gl CLI..."
  curl -sSf https://gitlawb.com/install.sh | sh
else
  echo "gl CLI already installed: $(which gl)"
fi

# Create identity if needed
if gl identity show &>/dev/null; then
  echo "Identity: $(gl identity show)"
else
  echo "Creating new identity..."
  gl identity new
  echo "Identity: $(gl identity show)"
fi

# Register with node
echo "Registering with $NODE_URL..."
export GITLAWB_NODE="$NODE_URL"
gl register

# Health check
echo ""
echo "Running health check..."
gl doctor

echo ""
echo "=== Setup complete ==="
echo "DID: $(gl identity show)"
echo "Node: $NODE_URL"
echo ""
echo "Next steps:"
echo "  gl repo create my-project --description \"...\""
echo "  gl quickstart  (interactive wizard)"
