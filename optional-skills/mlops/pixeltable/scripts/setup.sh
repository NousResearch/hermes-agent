#!/usr/bin/env bash
set -euo pipefail

echo "=== Pixeltable Setup ==="

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11 or newer first."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VERSION"

PY_OK=$(python3 -c "import sys; print(int(sys.version_info >= (3, 11)))")
if [[ "$PY_OK" != "1" ]]; then
    echo "ERROR: Pixeltable requires Python 3.11 or newer. Found $PY_VERSION."
    exit 1
fi

echo ""
echo "--- Installing Pixeltable ---"
python3 -m pip install --quiet pixeltable
echo "Installed pixeltable"
echo "For HTTP serving, also run: python3 -m pip install 'pixeltable[serve]'"

echo ""
echo "--- Verifying import ---"
python3 -c "import pixeltable as pxt; print(f'Pixeltable {pxt.__version__} ready')"

echo ""
echo "--- Checking MCP server (optional) ---"
if command -v uvx &>/dev/null; then
    echo "uvx found. Developer MCP is the PyPI package mcp-server-pixeltable-developer."
    echo "Add to ~/.hermes/config.yaml (paths must exist; Hermes does not inherit PIXELTABLE_* or API keys):"
    echo ""
    echo "  mcp_servers:"
    echo "    pixeltable:"
    echo "      command: uvx"
    echo "      args:"
    echo "        - mcp-server-pixeltable-developer"
    echo "      env:"
    echo "        PIXELTABLE_MCP_PROJECT_ROOT: /absolute/path/to/app"
    echo "        PIXELTABLE_HOME: /absolute/path/to/.pixeltable"
else
    echo "uvx not found. Install uv (https://docs.astral.sh/uv/) for MCP server support."
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
