# Hermes Agent

Multi-provider LLM agent framework with pluggable memory, credential management, and multi-platform gateway support.

## Tech Stack

- Runtime: Python 3.10+
- Key Dependencies: httpx, pydantic, pyyaml
- Gateway Platforms: Discord, Matrix, Slack, Telegram, WhatsApp, API server
- Memory Plugins: Honcho, Mem0, Holographic, Byterover, RetainDB, OpenViking, Hindsight

## Structure

```
agent/               # Core agent logic
  - credential_pool.py     # Multi-credential failover
  - memory_manager.py      # Memory provider orchestration
  - redact.py              # Secret redaction
  - anthropic_adapter.py   # LLM provider adaptation
gateway/             # Multi-platform message ingestion
hermes_cli/          # CLI and setup
plugins/memory/      # Memory backend implementations
```

## Conventions

- **Commit style**: Conventional commits (`fix(scope):`, `feat(scope):`, etc.)
- **Tests**: pytest in `tests/`
- **Secrets**: Always redacted before logging; never committed
- **Config**: YAML-based, `hermes_cli/config.py` provides interface
- **Type hints**: Required for new code

## Scripts

- `pytest tests/ -q` — run tests
- `python run_agent.py` — run agent
- `python -m hermes_cli.main` — CLI entry point

## Setup

```bash
pip install -r requirements.txt
python -m hermes_cli.main auth login
```

## Key Decisions

- **Credential pool**: Supports multiple credentials per provider with round-robin/least-used strategies
- **Memory plugins**: Pluggable architecture with standardized interface
- **Redaction**: Snapshots env at import time to prevent runtime bypass
- **Gateway delivery**: Event-driven, platform-agnostic message handling
