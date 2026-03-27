# Plan: Fork & Refactor hermes-agent (Security-Hardened)

## Context

hermes-agent (NousResearch) is a capable self-improving AI agent with native MCP support, but has significant security issues: unsafe installer (curl|bash, passwordless sudo, unsigned downloads), 40+ unpinned deps, no security policy, and bloated optional surfaces. This fork will produce a hardened, slimmed-down version for a local-only stack (hermes + obsidian-semantic-mcp + Ollama on 8GB M2).

---

## Security Audit Summary

### Trust Signals
- NousResearch — established AI research lab, 10K+ stars, MIT licensed
- Very active development (last push: today), 63 contributors
- Single lead maintainer (teknium1 = 95% of commits) — bus factor risk

### Critical Findings

| Issue | Severity | Detail |
|-------|----------|--------|
| curl\|bash installer | HIGH | Downloads unsigned binaries, runs passwordless sudo |
| No checksum verification | HIGH | uv and Node.js downloaded without SHA verification |
| Floating dependencies | HIGH | 40+ deps, most unpinned (openai, httpx, fire) |
| Git deps without pinning | HIGH | atroposlib, tinker pull latest main |
| No SECURITY.md | MEDIUM | No vulnerability disclosure process |
| Branch protection not enforced | MEDIUM | CI can be bypassed |
| Aggressive release cycle | LOW | 5-day intervals, limited testing window |

### Verdict
**MEDIUM TRUST, HIGH IMPLEMENTATION RISK.** Legitimate project, unsafe installer. Safe to use with manual installation and hardening.

---

## Phase 1 — Clone Existing Fork (DONE)

Fork: `celstnblacc/hermes-agent` (from `NousResearch/hermes-agent`).

```bash
git clone https://github.com/celstnblacc/hermes-agent.git ~/DevOpsSec/hermes-agent
cd ~/DevOpsSec/hermes-agent
git remote add upstream https://github.com/NousResearch/hermes-agent.git
```

---

## Phase 2 — Installer Rewrite (`scripts/install.sh`)

**Remove:**
- `sudo` calls entirely — no silent system package installation
- Unsigned binary downloads without checksums
- Auto-modification of shell rc files

**Add:**
- SHA256 checksum verification for all downloaded binaries (uv, Node.js)
- Explicit user confirmation before each system change
- `--dry-run` flag to preview all actions
- Prerequisite check that fails fast with clear instructions instead of auto-installing

**Model after:** `osm init` wizard pattern (confirmation prompts, dry-run, explicit actions).

---

## Phase 3 — Dependency Lockdown (`pyproject.toml`)

**Pin all deps** with upper bounds:
```
openai>=1.50,<2.0
anthropic>=0.39.0,<1.0
httpx>=0.27,<1.0
pydantic>=2.0,<3.0
```

**Remove Git deps** (`atroposlib`, `tinker`) from default install — move to a separate `[rl]` extra that requires explicit opt-in.

**Generate lockfile:** `uv lock` to produce reproducible builds.

---

## Phase 4 — Strip Unnecessary Extras

**Remove from default install** (keep as opt-in extras only):
- 6 messaging gateways (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email)
- 5 remote terminal backends (Docker, Singularity, Modal, Daytona, SSH) — keep `local` only
- RL training tools (`atroposlib`, `tinker`, W&B)
- Browser automation (`browserbase`)
- Image generation (`fal`)
- Home Assistant integration

**Keep in core:**
- MCP support (essential for obsidian-semantic integration)
- Skills system (the differentiator)
- Memory & sessions
- Local terminal backend
- OpenAI-compatible LLM client (for Ollama)

**Result:** Core install goes from 40+ deps to ~15.

---

## Phase 5 — Security Hardening

1. **Add `SECURITY.md`** — vulnerability disclosure policy
2. **Add `.shipguard.yml`** — SAST config for the fork
3. **Credential handling audit:**
   - Verify `~/.hermes/.env` is created with mode `0600`
   - Verify no credentials logged or transmitted beyond configured LLM endpoint
   - Audit the MCP environment filtering (already has safe-var allowlist — verify it's correct)
4. **Add pre-commit hook** with shipguard scan
5. **Pin GitHub Actions** in CI workflows to SHA (not floating tags)

---

## Phase 6 — Tests for Security-Critical Paths

Add tests for:
- Installer dry-run produces correct output without side effects
- MCP environment filtering blocks credential leakage
- Config file permissions (0600 for .env)
- Dependency resolution produces reproducible lockfile
- No network calls during offline mode

---

## Phase 7 — Ollama-First Configuration

**Default config changes:**
- Default model: `qwen3:4b` (not `anthropic/claude-opus-4.6`)
- Default provider: `openai` pointing at `localhost:11434/v1`
- Add `OLLAMA_MAX_LOADED_MODELS=1` guidance in default `.env.example`
- Pre-configure obsidian-semantic MCP in example config

---

## Phase 8 — Verify

| Check | Method |
|-------|--------|
| Clean install | `uv pip install -e .` in fresh venv — no errors, no sudo |
| Dep count | `pip list` shows ~15 core packages |
| shipguard clean | `shipguard scan .` — 0 findings |
| Tests pass | `pytest -q` |
| MCP works | `hermes` → `/tools` → obsidian-semantic tools discovered |
| E2E | "Search my vault for architecture decisions" returns results |
| Offline | Disconnect network, verify local Ollama inference still works |

---

## Files to modify

| File | Action |
|------|--------|
| `scripts/install.sh` | Rewrite — safe installer with checksums, dry-run, no sudo |
| `pyproject.toml` | Pin deps, restructure extras, slim core |
| `SECURITY.md` | Create |
| `.shipguard.yml` | Create |
| `.project-hooks/pre-commit` | Create |
| `.env.example` | Update defaults for Ollama-first |
| `config.example.yaml` | Update defaults, add obsidian-semantic MCP example |
| `.github/workflows/*.yml` | Pin actions to SHA |
| `tests/test_security.py` | Create — security-critical path tests |

---

## Upstream Compatibility

The refactor strips extras but doesn't change core APIs (`agent/core.py`, `tools/mcp_tool.py`, `tools/skills_tool.py`). Upstream improvements to the agent loop, MCP handling, and skills system can be cherry-picked without conflicts. Divergence is in packaging, installer, and default config — not runtime behavior.

---

## Integration with obsidian-semantic-mcp

**One-way (recommended to start):** hermes reads your vault via MCP tools (search, get_file). No osm changes needed.

**Bidirectional (deferred):** hermes writing to vault creates noise pollution and dual-memory ambiguity. Revisit only if hermes's built-in memory proves insufficient after real usage.

---

*Created: 2026-03-22*
