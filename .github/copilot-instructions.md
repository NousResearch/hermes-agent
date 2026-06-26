# Copilot Instructions — hermes-agent

NousResearch Hermes AI agent infrastructure. Local AI agent with tool use and memory.

## Stack
- NousResearch Hermes-3 models
- Python-based agent framework
- Connects to local LLM via Ollama (port 11434) or LM Studio
- API: port 8642 | Dashboard: port 9119

## Key Rules
- Services bind to `0.0.0.0` for Tailnet access — never `127.0.0.1`
- Secrets in `.env` only — never committed
- Models go on AI TOP NVMe — never C: drive
- Python: async preferred, type hints required

## Token Budget
See conductor-brain repo (CONTEXT_CONTROL.md) for agent delegation rules.
