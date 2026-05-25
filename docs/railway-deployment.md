# Deploying Hermes Agent to Railway

Deploy Hermes Agent to Railway as two services:
- **hermes-gateway**: Messaging platforms (Telegram, Discord, Slack, Teams, Google Chat)
- **hermes-dashboard**: Web UI on port 9119

Both services share a persistent volume at `/opt/data`.

## Prerequisites

1. Railway account (https://railway.app)
2. GitHub account
3. API keys for LLM provider (OpenRouter, Anthropic, OpenAI, etc.)

## Quick Start

### 1. Create Railway Project

```bash
git clone https://github.com/bubg61/hermes-variant.git
cd hermes-variant
railway init
