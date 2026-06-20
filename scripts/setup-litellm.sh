#!/bin/bash
set -euo pipefail
echo "🚀 Setting up LiteLLM..."
[[ ! -d venv ]] && python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
pip install litellm python-dotenv --quiet
echo "✅ LiteLLM ready"
