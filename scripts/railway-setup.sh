#!/bin/bash
set -eu

echo "🚀 Hermes Agent — Railway Deployment Setup"
echo ""

if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install from: https://railway.app/"
    exit 1
fi

echo "📦 Initializing Railway project..."
railway init

echo "🔧 Creating services..."
railway service add hermes-gateway
railway service add hermes-dashboard

echo "💾 Creating persistent volume..."
railway volume create hermes-data 10240

echo "🔗 Mounting volume to services..."
railway service select hermes-gateway
railway volume mount hermes-data /opt/data
railway service select hermes-dashboard
railway volume mount hermes-data /opt/data

echo ""
echo "📝 Configure environment variables:"
echo "1. Open Railway dashboard: https://railway.app/"
echo "2. Go to Variables for each service"
echo "3. Add required variables from .env.railway.example"
echo ""
echo "Minimum required:"
echo "  - OPENROUTER_API_KEY (or other LLM provider key)"
echo "  - Messaging platform tokens (if using)"
echo ""

echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment complete!"
echo "Next steps:"
echo "1. Configure environment variables in Railway dashboard"
echo "2. Access dashboard: https://hermes-dashboard-xxxx.up.railway.app"
echo "3. Set up messaging platforms (Telegram, Discord, etc.)"
echo ""
echo "Documentation: docs/railway-deployment.md"
