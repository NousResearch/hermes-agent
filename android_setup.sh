#!/bin/bash
# Hermes Agent Android Setup Script for Termux
# This script prepares the Termux environment for running Hermes Agent in lightweight mode.

set -e

echo "=== Hermes Agent Android Setup ==="
echo "Preparing Termux environment..."

# Update and upgrade packages
echo "Updating package lists..."
pkg update -y && pkg upgrade -y

# Install essential packages
echo "Installing essential packages..."
pkg install -y python git openssh termux-api

# Install Python packages in a virtual environment
echo "Setting up Python virtual environment..."
python -m venv ~/.hermes-android-venv
source ~/.hermes-android-venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core Python packages needed for lightweight operation
echo "Installing core Python packages..."
pip install \
    python-dotenv \
    requests \
    websocket-client \
    pyyaml \
    sqlite3 \
    gitpython \
    termux-api \
    rich \
    prompt_toolkit

# Clone Hermes agent repository if not present
if [ ! -d "$HOME/hermes-agent" ]; then
    echo "Cloning Hermes Agent repository..."
    git clone https://github.com/AHMADSYAF21/hermes-agent.git "$HOME/hermes-agent"
else
    echo "Updating existing Hermes Agent repository..."
    cd "$HOME/hermes-agent"
    git pull
fi

# Create Android-specific config directory
mkdir -p ~/.hermes-android

# Create a basic config.yaml for Android
cat > ~/.hermes-android/config.yaml << 'EOF'
# Hermes Agent Android Configuration
# Lightweight configuration for Termux/Android

# Core settings
max_iterations: 30
save_trajectories: false
quiet_mode: false

# Toolset configuration - only enable lightweight tools on Android
enabled_toolsets:
  - file
  - terminal
  - web  # for HTTP requests, but not heavy browsing
  - skills
  - session_search
  - todo

disabled_toolsets:
  - browser          # Heavy - use bridge instead
  - computer_use     # Not available on Android without root
  - vision           # Heavy - use bridge if needed
  - cronjob          # Limited on Android, use bridge
  - delegation       # Subagents can be heavy
  - image_gen        # Heavy ML models
  - voice            # Heavy processing
  - mcp              # May be heavy

# Logging
log_level: "INFO"

# Bridge settings for heavy operations
bridge:
  enabled: true
  server_url: "ws://your-vps-ip:2999/socket"  # Change to your VPS
  auth_token: ""  # Set if using JWT auth
  timeout: 30
EOF

# Create a simple startup script
cat > ~/.hermes-agent/start_android.sh << 'EOF'
#!/bin/bash
# Start Hermes Agent in Android lightweight mode

# Activate virtual environment
source "$HOME/.hermes-android-venv/bin/activate"

# Set environment
export HERMES_HOME="$HOME/.hermes-android"
export PYTHONPATH="$HOME/hermes-agent:$PYTHONPATH"

# Run the lightweight agent
echo "Starting Hermes Agent (Android Lightweight Mode)..."
python "$HOME/hermes-agent/run_agent.py" \
    --config "$HOME/.hermes-android/config.yaml" \
    "$@"
EOF

chmod +x ~/.hermes-agent/start_android.sh

# Make the hermes executable
cat > ~/.hermes-agent/hermes << 'EOF'
#!/bin/bash
# Hermes wrapper for Android
source "$HOME/.hermes-agent/start_android.sh" "$@"
EOF

chmod +x ~/.hermes-agent/hermes
ln -sf "$HOME/.hermes-agent/hermes" "$HOME/bin/hermes" 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo "To start Hermes Agent on Android:"
echo "  source ~/.hermes-android-venv/bin/activate"
echo "  hermes"
echo ""
echo "For heavy operations, the agent will automatically bridge to your VPS"
echo "if bridge is configured in ~/.hermes-android/config.yaml"
echo ""
echo "Make sure to set your VPS IP and auth token in the config file."