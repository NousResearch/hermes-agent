#!/usr/bin/env bash
set -euo pipefail

echo "=== Pixeltable Setup ==="

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VERSION"

echo ""
echo "--- Installing Pixeltable ---"
pip install --quiet pixeltable
echo "Installed pixeltable"

echo ""
echo "--- Verifying import ---"
python3 -c "import pixeltable as pxt; print(f'Pixeltable {pxt.__version__} ready')"

echo ""
echo "--- Installing MCP server (optional) ---"
if pip install --quiet mcp-server-pixeltable-developer 2>/dev/null; then
    echo "MCP server installed"
    echo ""
    echo "Add to ~/.hermes/config.yaml:"
    echo ""
    echo "  mcpServers:"
    echo "    pixeltable:"
    echo "      command: uvx"
    echo "      args: [mcp-server-pixeltable-developer]"
else
    echo "MCP server skipped (install manually: pip install mcp-server-pixeltable-developer)"
fi

echo ""
echo "--- Running verification ---"
python3 -c "
import pixeltable as pxt
pxt.create_dir('pxtverify', if_exists='ignore')
t = pxt.create_table('pxtverify.test', {'text': pxt.String}, if_exists='replace')
t.insert([{'text': 'hello'}, {'text': 'world'}])
assert len(t.collect()) == 2
pxt.drop_table('pxtverify.test')
pxt.drop_dir('pxtverify')
print('Verification passed')
"

echo ""
echo "=== Setup complete ==="
