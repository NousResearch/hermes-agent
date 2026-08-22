#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Installing dependencies for Security Recon Assistant..."

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Check for Go
if ! command -v go &>/dev/null; then
    echo "⚠️  Go is not installed. Please install Go 1.21+ first:"
    echo "   https://go.dev/dl/"
    exit 1
fi

# System packages
install_system_pkg() {
    local pkg=$1
    if command -v "$pkg" &>/dev/null; then
        echo "✅ $pkg already installed"
        return
    fi

    echo "📦 Installing $pkg..."
    case "$OS" in
        linux)
            if command -v apt-get &>/dev/null; then
                sudo apt-get update
                sudo apt-get install -y "$pkg"
            elif command -v yum &>/dev/null; then
                sudo yum install -y "$pkg"
            else
                echo "❌ Unsupported Linux package manager. Install $pkg manually."
                exit 1
            fi
            ;;
        darwin)
            if command -v brew &>/dev/null; then
                brew install "$pkg"
            else
                echo "❌ Homebrew not found. Install $pkg manually."
                exit 1
            fi
            ;;
        *)
            echo "❌ Unsupported OS: $OS"
            exit 1
            ;;
    esac
}

# Install system tools
for pkg in nmap whatweb sslscan; do
    install_system_pkg "$pkg"
done

# Go tools
install_go_tool() {
    local import_path=$1
    local bin_name=$(basename "$import_path" | sed 's/@.*//')
    if command -v "$bin_name" &>/dev/null; then
        echo "✅ $bin_name already installed"
        return
    fi
    echo "📦 Installing $bin_name via go install..."
    GOBIN="${GOBIN:-$HOME/go/bin}" go install -v "$import_path"
    echo "   → $GOBIN/$bin_name"
}

install_go_tool "github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest"
install_go_tool "github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest"
install_go_tool "github.com/ffuf/ffuf/v2@latest"
install_go_tool "github.com/sensepost/gowitness/v3@latest"

# Ensure GOBIN is in PATH
GOBIN="${GOBIN:-$HOME/go/bin}"
if [[ ":$PATH:" != *":$GOBIN:"* ]]; then
    echo "⚠️  Add Go bin to PATH: export PATH=\$PATH:$GOBIN"
fi

# Update nuclei templates
if command -v nuclei &>/dev/null; then
    echo "🔄 Updating nuclei templates..."
    nuclei -update-templates 2>/dev/null || true
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v python3 &>/dev/null; then
    python3 -m pip install -e . --break-system-packages
    echo "✅ Python dependencies installed"
else
    echo "❌ python3 not found. Install Python dependencies manually:"
    echo "   python3 -m pip install -e ."
fi

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "📝 Next steps:"
echo "1. Copy templates/scope.example.yaml to scope.yaml"
echo "2. Edit scope.yaml with authorized targets"
echo "3. Run: python -m security_recon_assistant --scope scope.yaml --targets \"scanme.nmap.org\""
