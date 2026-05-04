# Hermes Agent — Setup Guide

This directory contains the config template and setup instructions for reproducing this hermes-agent setup on a new machine.

## What's in this branch

- **3 bug fixes** in the hermes-agent Python code (see PR description):
  - `cli.py`: `model.provider` dict no longer crashes provider selection
  - `hermes_cli/auth.py`: same dict-vs-string guard
  - `hermes_cli/runtime_provider.py`: same guard in runtime resolver
- **`setup/config-template.yaml`**: hermes config with NVIDIA NIM free API as fallback
- **`setup/.env.template`**: environment variable template

## Setup on a new laptop

### 1. Clone this fork
```bash
git clone https://github.com/namangoyal3/hermes-agent.git ~/.hermes-agent
cd ~/.hermes-agent
pip install -e .
```

### 2. Create hermes config directory
```bash
mkdir -p ~/.hermes
cp ~/.hermes-agent/setup/config-template.yaml ~/.hermes/config.yaml
cp ~/.hermes-agent/setup/.env.template ~/.hermes/.env
```

### 3. Fill in your API keys
Edit `~/.hermes/.env` with your real keys:
- **NVIDIA NIM** (free): https://build.nvidia.com — sign up, generate API key
- **Cloudflare Workers AI** (free tier): https://dash.cloudflare.com/
- **OpenRouter** (free models): https://openrouter.ai/keys

Also update `~/.hermes/config.yaml`:
- Replace `YOUR_CF_ACCOUNT_ID` with your Cloudflare account ID
- Replace `YOUR_CLOUDFLARE_API_TOKEN` and `YOUR_NVIDIA_API_KEY` in `custom_providers`

### 4. Start the hermes gateway
```bash
hermes gateway start
```

## Free model stack

| Priority | Provider | Model | Context |
|----------|----------|-------|---------|
| Primary | Cloudflare Workers AI | `kimi-k2.6` | 128K |
| Fallback 1 | NVIDIA NIM (free) | `meta/llama-3.3-70b-instruct` | 128K |
| Fallback 2 | NVIDIA NIM (free) | `nvidia/llama-3.3-nemotron-super-49b-v1` | 128K |
| Fallback 3 | OpenRouter (free) | `minimax/minimax-m2.5:free` | 1M |

## Why these models?

NVIDIA NIM's `meta/llama-3.3-70b-instruct` and `nvidia/llama-3.3-nemotron-super-49b-v1` return proper OpenAI-format JSON tool calls. Other NVIDIA models (MiniMax M2.7, Nemotron 120B) use a custom XML format that breaks hermes tool use.

## Troubleshooting

- **HTTP 402**: API account has no credits — hermes will automatically fall through to next fallback
- **HTTP 429**: Rate limited — hermes will automatically fall through to next fallback  
- **`'dict' object has no attribute 'strip'`**: Fixed in this branch — `model.provider` is a dict in config but code assumed string
