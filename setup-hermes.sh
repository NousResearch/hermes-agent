#!/bin/bash

# Hermes Agent Setup Script
# Automated setup for all dependencies and configuration

set -e

echo "========================================="
echo "Hermes Agent Setup"
echo "========================================="
echo ""

# Change to hermes-agent directory
cd /home/teknium/hermes-agent

# Check Python version
echo "[1/10] Checking Python version..."
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python $python_version detected"
echo ""

# Install uv
echo "[2/10] Installing uv (fast Python package installer)..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✓ uv installed"
else
    echo "✓ uv already installed: $(uv --version)"
fi
echo ""

# Install Node.js 20 using NodeSource
echo "[3/10] Installing Node.js 20..."
if ! command -v node &> /dev/null || [[ $(node --version | cut -d'v' -f2 | cut -d'.' -f1) -lt 20 ]]; then
    echo "Installing Node.js 20 LTS..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    echo "✓ Node.js installed"
else
    echo "✓ Node.js 20+ already installed: $(node --version)"
fi
echo ""

# Initialize git submodules
echo "[4/10] Initializing git submodules..."
git submodule update --init --recursive
echo "✓ Submodules initialized"
echo ""

# Create Python virtual environment with uv
echo "[5/10] Creating Python virtual environment with uv..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    uv venv venv
    echo "✓ Virtual environment created with uv"
fi
echo ""

# Activate virtual environment and install Python packages with uv
echo "[6/10] Installing Python dependencies with uv..."
source venv/bin/activate
uv pip install -r requirements.txt
echo "✓ Python packages installed"
echo ""

# Install mini-swe-agent with uv
echo "[7/10] Installing mini-swe-agent..."
uv pip install -e ./mini-swe-agent
echo "✓ mini-swe-agent installed"
echo ""

# Install Node.js dependencies
echo "[8/10] Installing Node.js dependencies..."
npm install
echo "✓ Node.js packages installed"
echo ""

# Set up environment file
echo "[9/10] Setting up environment configuration..."
if [ -f ".env" ]; then
    echo ".env file already exists, creating backup..."
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi
cp .env.example .env
echo "✓ .env file created from .env.example"
echo ""

# Set up CLI config
echo "[10/10] Setting up CLI configuration..."
if [ ! -f "cli-config.yaml" ]; then
    cp cli-config.yaml.example cli-config.yaml
    echo "✓ cli-config.yaml created from example"
else
    echo "cli-config.yaml already exists, skipping..."
fi
echo ""

# Show Node.js and Python versions
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Installed versions:"
echo "  Node.js: $(node --version)"
echo "  npm: $(npm --version)"
echo "  Python: $(python3 --version)"
echo "  uv: $(uv --version)"
echo ""

echo "========================================="
echo "Next Steps:"
echo "========================================="
echo ""
echo "1. Configure API Keys in .env file:"
echo "   nano .env"
echo ""
echo "   Required API keys:"
echo "   - OPENROUTER_API_KEY (https://openrouter.ai/keys)"
echo "   - FIRECRAWL_API_KEY (https://firecrawl.dev/)"
echo "   - NOUS_API_KEY (https://inference-api.nousresearch.com/)"
echo "   - FAL_KEY (https://fal.ai/)"
echo ""
echo "   Optional API keys:"
echo "   - BROWSERBASE_API_KEY (https://browserbase.com/)"
echo "   - BROWSERBASE_PROJECT_ID"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the CLI:"
echo "   ./hermes"
echo ""
echo "4. Or run a single query:"
echo "   python run_agent.py --query \"your question here\""
echo ""
echo "5. List available tools:"
echo "   python run_agent.py --list_tools"
echo ""
echo "========================================="
echo "Configuration Files:"
echo "========================================="
echo "  .env - API keys and environment variables"
echo "  cli-config.yaml - CLI settings and preferences"
echo ""
echo "For more information, see README.md"
echo ""
